from transformers import AutoTokenizer, AutoProcessor
from transformers.models.mllama.modeling_mllama import MllamaForConditionalGeneration
from peft import PeftModel, PeftConfig
import argparse
import os
import torch
import safetensors.torch
import shutil
from huggingface_hub import snapshot_download
from peft.utils import set_peft_model_state_dict


def download_adapter_from_hf(repo_id: str, local_dir: str):
    print(f"[INFO] Downloading adapter files from: {repo_id}")
    snapshot_path = snapshot_download(repo_id, repo_type="model", local_dir=local_dir, local_dir_use_symlinks=False)
    required_files = ["adapter_config.json", "adapter_model.safetensors"]
    for fname in required_files:
        src = os.path.join(snapshot_path, fname)
        dst = os.path.join(local_dir, fname)
        if os.path.abspath(src) != os.path.abspath(dst):
            if os.path.exists(src):
                shutil.copy2(src, dst)
                print(f"  - Copied: {fname}")
            else:
                print(f"  - [WARN] {fname} not found in snapshot.")
        else:
            print(f"  - [INFO] {fname} already present at destination, skipping copy.")


def get_vocab_size_from_safetensor(safetensor_path):
    weights = safetensors.torch.load_file(safetensor_path)
    for key in weights:
        if "lm_head" in key and "weight" in key:
            return weights[key].shape[0]
    raise ValueError("Could not determine vocab size from safetensor.")


def copy_base_model_files(base_model_id, save_path):
    print(f"[INFO] Copying config/tokenizer files from base model: {base_model_id}")
    snapshot_path = snapshot_download(base_model_id)

    files_to_copy = [
        "config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "special_tokens_map.json",
        "tokenizer.json",
        "generation_config.json",
        "chat_template.json"
    ]

    for fname in files_to_copy:
        src = os.path.join(snapshot_path, fname)
        dst = os.path.join(save_path, fname)
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"  - copied: {fname}")
        else:
            print(f"  - [WARN] {fname} not found in base model snapshot. Skipping.")


def merge_lora_single(base_model: str, adapter_dir: str, save_path: str, hf_adapter_repo: str = None):
    # If Hugging Face repo is given, download adapter files
    if hf_adapter_repo:
        download_adapter_from_hf(hf_adapter_repo, adapter_dir)

    config_path = os.path.join(adapter_dir, "adapter_config.json")
    safetensor_path = next(
        (os.path.join(adapter_dir, f) for f in os.listdir(adapter_dir) if f.endswith(".safetensors")),
        None
    )
    if not os.path.exists(config_path) or safetensor_path is None:
        raise FileNotFoundError(f"Required files not found in {adapter_dir}")

    print(f"[INFO] Loading base model from: {base_model}")
    base = MllamaForConditionalGeneration.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    base_vocab_size = base.get_input_embeddings().weight.shape[0]

    adapter_vocab_size = get_vocab_size_from_safetensor(safetensor_path)
    if adapter_vocab_size != base_vocab_size:
        print(f"[INFO] Adapter vocab size ({adapter_vocab_size}) != base vocab size ({base_vocab_size}), resizing base model vocab")
        base.resize_token_embeddings(adapter_vocab_size)
        with torch.no_grad():
            old_weight = base.lm_head.weight.data
            in_features = base.lm_head.in_features
            base.lm_head = torch.nn.Linear(in_features, adapter_vocab_size, bias=False).to(base.device, dtype=base.dtype)
            base.lm_head.weight.data[:old_weight.shape[0], :] = old_weight

    print("[INFO] Initializing PeftModel")
    peft_config = PeftConfig.from_pretrained(adapter_dir)
    model = PeftModel(base, peft_config)

    print("[INFO] Loading adapter weights from safetensor")
    adapter_state_dict = safetensors.torch.load_file(safetensor_path)
    set_peft_model_state_dict(model, adapter_state_dict)

    print("[INFO] Merging adapter into base model")
    merged_model = model.merge_and_unload()

    print(f"[INFO] Saving merged model to: {save_path}")
    merged_model.save_pretrained(save_path)

    print("[INFO] Saving tokenizer and processor")
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    processor.save_pretrained(save_path)

    print("[INFO] Copying base model config & preprocessor files")
    copy_base_model_files(base_model, save_path)

    print("[✅] Merge completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into a multimodal base model (e.g., LLaMA-Vision-Instruct).")
    parser.add_argument("--base_model", required=True, help="HuggingFace model ID or local path to the base model (e.g. meta-llama/Llama-3.2-11B-Vision-Instruct)")
    parser.add_argument("--adapter_dir", required=True, help="Path to the directory to save or load adapter_config.json and .safetensors")
    parser.add_argument("--save_path", default="merged_multimodal", help="Directory to save the merged model")
    parser.add_argument("--hf_adapter_repo", default=None, help="Hugging Face Hub repo to download adapter files if adapter_dir is empty")

    args = parser.parse_args()
    merge_lora_single(args.base_model, args.adapter_dir, args.save_path, hf_adapter_repo=args.hf_adapter_repo)
