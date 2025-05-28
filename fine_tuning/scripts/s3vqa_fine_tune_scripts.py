from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Directory constants (relative to this script: fine_tuning/scripts/...)
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "raw" / "s3vqa"
IMG_ROOT = ROOT_DIR / "images" / "s3vqa_images"
PROC_DIR = ROOT_DIR / "processed" / "s3vqa"
CKPT_DIR = ROOT_DIR / "checkpoints"

MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"

# ---------------------------------------------------------------------------
# Phase 1  — conversion: raw → processed
# ---------------------------------------------------------------------------

def convert_split(split: str) -> None:
    """Convert raw S3VQA jsonl to training‑ready jsonl."""
    src_file = RAW_DIR / f"{split}.jsonl"
    dst_file = PROC_DIR / f"{split}.jsonl"
    if not src_file.exists():
        sys.exit(f"[ERROR] raw file not found: {src_file}")

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    IMG_ROOT.mkdir(parents=True, exist_ok=True)

    written = 0
    with src_file.open("r", encoding="utf-8") as src, dst_file.open("w", encoding="utf-8") as dst:
        for line in src:
            ex = json.loads(line)
            fname = Path(ex["image"]).name
            candidates = [
                ROOT_DIR / ex["image"],
                IMG_ROOT / fname,
                ROOT_DIR / "open_images" / "train" / fname,
            ]
            img_src = next((p for p in candidates if p.exists()), None)
            if img_src is None:
                print(f"[WARN] image not found, skip: {ex['image']}")
                continue

            img_dst = IMG_ROOT / fname
            if not img_dst.exists():
                try:
                    os.symlink(img_src, img_dst)
                except OSError:
                    shutil.copy(img_src, img_dst)

            record = {
                "image_path": str(img_dst.relative_to(ROOT_DIR)),
                "messages": [
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": ex["question"]}]},
                    {"role": "assistant", "content": [{"type": "text", "text": ex["answer"]}]},
                ],
            }
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    print(f"[OK] {split}: {written} samples → {dst_file}")

# ---------------------------------------------------------------------------
# Phase 2  — training: LoRA fine‑tune
# ---------------------------------------------------------------------------

def train_lora(split: str, run_name: str, batch: int, grad_acc: int, epochs: int) -> None:
    """Fine‑tune Llama‑3.2‑Vision with 4‑bit QLoRA."""
    try:
        import torch
        from datasets import load_dataset
        from transformers import (
            AutoConfig, AutoProcessor, MllamaForConditionalGeneration, BitsAndBytesConfig,
            TrainingArguments, Trainer,
        )
        from peft import LoraConfig, get_peft_model
        from PIL import Image
    except ModuleNotFoundError as e:
        sys.exit(f"[ERROR] Missing training dependency: {e.name}.")

    proc_file = PROC_DIR / f"{split}.jsonl"
    if not proc_file.exists():
        sys.exit(f"[ERROR] processed file not found: {proc_file}")

    config = AutoConfig.from_pretrained(MODEL_ID)
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = MllamaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        config=config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_cfg,
    )

    lora_cfg = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_cfg)

    dataset = load_dataset("json", data_files={split: str(proc_file)})[split]

    def preprocess(example):
        img_path = Path(example["image_path"])
        if not img_path.is_absolute():
            img_path = ROOT_DIR / img_path
        img = Image.open(img_path).convert("RGB")
        # Pass the image explicitly as list to satisfy MllamaProcessor signature
        out = processor(example["messages"], images=[img], return_tensors="pt")
        out["labels"] = out["input_ids"].clone()
        return out

    dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

    output_dir = CKPT_DIR / run_name
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=grad_acc,
        num_train_epochs=epochs,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
    trainer.train()
    model.save_pretrained(output_dir)
    print(f"[DONE] LoRA weights saved to {output_dir}")

# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["convert", "train"], required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--run_name", default="s3vqa_lora")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--grad_acc", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    if args.phase == "convert":
        convert_split(args.split)
    else:
        train_lora(args.split, args.run_name, args.batch, args.grad_acc, args.epochs)
