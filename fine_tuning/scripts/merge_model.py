from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from peft.utils import set_peft_model_state_dict
import argparse
import os
import torch
import safetensors.torch


def get_vocab_size_from_safetensor(safetensor_path):
    weights = safetensors.torch.load_file(safetensor_path)
    for key in weights:
        if "lm_head" in key and "weight" in key:
            return weights[key].shape[0]
    raise ValueError("Could not determine vocab size from safetensor.")


def merge_lora_single(base_model: str, adapter_dir: str, save_path: str):
    # Paths
    config_path = os.path.join(adapter_dir, "adapter_config.json")
    safetensor_path = next(
        (os.path.join(adapter_dir, f) for f in os.listdir(adapter_dir) if f.endswith(".safetensors")),
        None
    )
    if not os.path.exists(config_path) or safetensor_path is None:
        raise FileNotFoundError(f"Required files not found in {adapter_dir}")

    print(f"Loading base model from: {base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    base_vocab_size = base.get_input_embeddings().weight.shape[0]

    adapter_vocab_size = get_vocab_size_from_safetensor(safetensor_path)
    if adapter_vocab_size != base_vocab_size:
        print(f"[INFO] Trimming adapter vocab from {adapter_vocab_size} to {base_vocab_size}")

    print(f"Initializing PeftModel with base model and adapter config")
    peft_config = PeftConfig.from_pretrained(adapter_dir)
    model = PeftModel(base, peft_config)

    # Load and trim adapter state dict
    adapter_state_dict = safetensors.torch.load_file(safetensor_path)
    trimmed_state_dict = {}
    for k, v in adapter_state_dict.items():
        if ("lm_head" in k or "lora_B" in k) and v.shape[0] != base_vocab_size:
            trimmed_state_dict[k] = v[:base_vocab_size, ...]
        else:
            trimmed_state_dict[k] = v
    print("Loading trimmed adapter state dict into model")
    set_peft_model_state_dict(model, trimmed_state_dict)

    print("Merging LoRA weights into base model")
    merged_model = model.merge_and_unload()

    print(f"Saving merged model to: {save_path}")
    merged_model.save_pretrained(save_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.save_pretrained(save_path)

    print("Merge completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge a single LoRA adapter into the base model.")
    parser.add_argument("--base_model", required=True, help="HuggingFace model ID or local path to the base model")
    parser.add_argument("--adapter_dir", required=True, help="Directory containing adapter_config.json and .safetensors for the LoRA adapter")
    parser.add_argument("--save_path", default="mergedmodel", help="Output directory to save the merged model")

    args = parser.parse_args()
    merge_lora_single(args.base_model, args.adapter_dir, args.save_path)
    print("Arguments parsed successfully.")
    print(f"Base model: {args.base_model}")
    print(f"Adapter directory: {args.adapter_dir}")
    print(f"Save path: {args.save_path}")