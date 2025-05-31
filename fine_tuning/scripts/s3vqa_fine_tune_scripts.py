from __future__ import annotations
"""s3vqa_fine_tune_scripts.py — 이미지 기반 VQA를 위한 QLoRA 학습 스크립트. 라벨 오류를 방지하기 위해 특수 토큰 마스킹 및 Grad 오류 수정 포함."""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
import gc
import psutil

import torch
from datasets import load_dataset, Image as DatasetsImage
from transformers import (
    AutoConfig, AutoProcessor, MllamaForConditionalGeneration,
    BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model
from PIL import Image, ImageFile

# ---------------------------------------------------------------------------
# 기본 설정
# ---------------------------------------------------------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "raw" / "s3vqa"
IMG_ROOT = ROOT_DIR / "images" / "s3vqa_images"
PROC_DIR = ROOT_DIR / "processed" / "s3vqa"
CKPT_DIR = ROOT_DIR / "checkpoints"
MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---------------------------------------------------------------------------
# Phase 1 — 전처리: JSONL 형식 정리 및 이미지 링크 생성
# ---------------------------------------------------------------------------
def convert_split(split: str) -> None:
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
                    {"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text", "text": ex["question"]},
                    ]},
                    {"role": "assistant", "content": [
                        {"type": "text", "text": ex["answer"]},
                    ]}
                ]
            }
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    print(f"[OK] {split}: {written} samples → {dst_file}")

# ---------------------------------------------------------------------------
# Phase 2 — QLoRA 파인튜닝 루틴
# ---------------------------------------------------------------------------
def train_lora(split: str, run_name: str, batch: int, grad_acc: int, epochs: int) -> None:
    proc_file = PROC_DIR / f"{split}.jsonl"
    if not proc_file.exists():
        sys.exit(f"[ERROR] processed file not found: {proc_file}")

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    vocab_size = len(processor.tokenizer)

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    lora_cfg = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "embed_tokens", "lm_head"
        ],
    )

    dataset_dict = load_dataset("json", data_files={split: str(proc_file)})
    dataset = dataset_dict[split].cast_column("image", DatasetsImage(decode=True))

    from torch.utils.data import Dataset

    class S3VQADataset(Dataset):
        def __init__(self, dataset, processor):
            self.dataset = dataset
            self.processor = processor
            self.valid_indices = [i for i, ex in enumerate(dataset) if "image_path" in ex and os.path.exists(ROOT_DIR / ex["image_path"])]

        def __len__(self):
            return len(self.valid_indices)

        def __getitem__(self, idx):
            max_tries = 5
            tries = 0
            while tries < max_tries:
                ex = self.dataset[self.valid_indices[idx]]
                img_path = ROOT_DIR / ex["image_path"]
                image = Image.open(img_path).convert("RGB")
                prompt = self.processor.tokenizer.apply_chat_template(ex["messages"], tokenize=False)

                encoded = self.processor(
                    text=[prompt],
                    images=[image],
                    return_tensors="pt",
                    padding="max_length",
                    max_length=1024,
                    truncation=True,
                    add_special_tokens=True
                )

                input_ids = encoded["input_ids"]
                labels = input_ids.clone()
                labels[encoded["attention_mask"] == 0] = -100
                labels[labels == processor.image_token_id] = -100
                encoded["labels"] = labels

                if (labels != -100).sum().item() == 0:
                    idx = (idx + 1) % len(self)
                    tries += 1
                    continue
                if labels.max().item() >= vocab_size:
                    print("[ERROR] 잘못된 라벨이 포함됨.")
                    idx = (idx + 1) % len(self)
                    tries += 1
                    continue

                return {k: v.squeeze(0) for k, v in encoded.items()}

            raise IndexError("[ERROR] 유효한 예제를 5회 시도했지만 얻지 못함")

    dataset = S3VQADataset(dataset, processor)

    model = MllamaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_cfg,
    )
    model.resize_token_embeddings(len(processor.tokenizer))
    model = get_peft_model(model, lora_cfg)

    training_args = TrainingArguments(
        output_dir=str(CKPT_DIR / run_name),
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=grad_acc,
        num_train_epochs=epochs,
        learning_rate=2e-4,
        bf16=True,
        remove_unused_columns=False,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        gradient_checkpointing=True
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
    trainer.train()
    model.save_pretrained(CKPT_DIR / run_name)

    del model
    gc.collect()
    torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# CLI 엔트리 포인트
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

