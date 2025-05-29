import torch
from datasets import load_dataset
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
import os
from PIL import Image
from torch.utils.data import Dataset

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WandB not installed. Install with: pip install wandb")

# Initialize wandb for experiment tracking (optional)
use_wandb = False
if WANDB_AVAILABLE:
    try:
        wandb.init(project="llama-vision-finetune", name="crag-mm-finetune-a100")
        use_wandb = True
    except:
        print("WandB not logged in. Run 'wandb login' or continue without logging.")
else:
    print("Continuing without WandB logging.")

# Model configuration
MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"
OUTPUT_DIR = "./llama-vision-finetuned"

# Load model and processor
print("Loading model and processor...")

# Try to use Flash Attention 2 if available
try:
    import flash_attn
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2"  # New syntax for Flash Attention 2
    )
    print("Using Flash Attention 2")
except ImportError:
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa"  # Use PyTorch's native scaled dot-product attention
    )
    print("Flash Attention 2 not available, using PyTorch SDPA")

processor = AutoProcessor.from_pretrained(MODEL_ID,trust_remote_code=True)

try:
    token_128256_str = processor.tokenizer.decode([128256])
    print(f"Token ID 128256 decodes to: '{token_128256_str}'")
except Exception as e:
    print(f"Could not decode token ID 128256: {e}")

# Also check the actual vocab size of the tokenizer object
print(f"Length of tokenizer's vocabulary (len(processor.tokenizer)): {len(processor.tokenizer)}")

# Ensure padding token is set
if processor.tokenizer.pad_token is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

# LoRA configuration - increased parameters for better performance
lora_config = LoraConfig(
    r=64,  # Increased rank for better adaptation
    lora_alpha=128,
    target_modules=[
        "q_proj",
        "k_proj", 
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "embed_tokens",
        "lm_head"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False
)

# Apply LoRA
print("Applying LoRA...")
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Enable gradient checkpointing for memory efficiency
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

# Load dataset
print("Loading dataset...")
dataset = load_dataset("crag-mm-2025/crag-mm-single-turn-public", revision="v0.1.2")

# Check available splits
print(f"Available splits: {list(dataset.keys())}")

# Use validation for training and public_test for evaluation
train_split = 'validation'
eval_split = 'public_test'

print(f"Using {train_split} for training: {len(dataset[train_split])} samples")
print(f"Using {eval_split} for evaluation: {len(dataset[eval_split])} samples")

# Custom Dataset class for better control
class VisionDataset(Dataset):
    def __init__(self, dataset, processor, max_length=2048):
        self.dataset = dataset
        self.processor = processor
        self.max_length = max_length
        
        # Filter out examples without images
        self.valid_indices = []
        print("Filtering dataset for examples with images...")
        for i in range(len(dataset)):
            if dataset[i].get('image') is not None:
                self.valid_indices.append(i)
        
        print(f"Found {len(self.valid_indices)} examples with images out of {len(dataset)} total")
        
        # Option to download images from URLs (commented out for now)
        # self._download_missing_images()
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        # Get the actual dataset index
        dataset_idx = self.valid_indices[idx]
        example = self.dataset[dataset_idx]
        
        # Check if image exists
        if example.get('image') is None:
            # This shouldn't happen after filtering, but just in case
            raise ValueError(f"No image found for example {dataset_idx}")
        
        # Extract the first turn (single-turn dataset)
        query = example['turns']['query'][0]
        answer = example['answers']['ans_full'][0]
        
        # Create conversation format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": query}
                ]
            },
            {
                "role": "assistant",
                "content": answer
            }
        ]
        
        # Process with the processor
        text = self.processor.apply_chat_template(messages, tokenize=False)
        
        # Process image and text
        inputs = self.processor(
            text=text,
            images=example['image'],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        
        # Remove batch dimension
        processed = {k: v.squeeze(0) for k, v in inputs.items()}
        
        # Add labels (same as input_ids for causal LM)
        processed['labels'] = processed['input_ids'].clone()
        
        # Mask padding tokens in labels
        processed['labels'][processed['labels'] == self.processor.tokenizer.pad_token_id] = -100
        processed['labels'][processed['labels'] == 128256] = -100 # Explicitly ignore this problematic token

        
        return processed
    
    def _download_missing_images(self):
        """
        Optional: Download images from URLs if needed
        """
        import requests
        from io import BytesIO
        
        for i in range(len(self.dataset)):
            example = self.dataset[i]
            if example.get('image') is None and example.get('image_url'):
                try:
                    response = requests.get(example['image_url'], timeout=10)
                    if response.status_code == 200:
                        image = Image.open(BytesIO(response.content))
                        # Note: This won't actually modify the dataset
                        # You'd need to handle this differently
                        print(f"Downloaded image from {example['image_url']}")
                except Exception as e:
                    print(f"Failed to download image: {e}")

# Check how many examples have images vs URLs
print("\nAnalyzing dataset image availability:")
train_with_image = sum(1 for ex in dataset[train_split] if ex.get('image') is not None)
eval_with_image = sum(1 for ex in dataset[eval_split] if ex.get('image') is not None)
print(f"Training set: {train_with_image}/{len(dataset[train_split])} examples have images")
print(f"Evaluation set: {eval_with_image}/{len(dataset[eval_split])} examples have images")

# Create datasets
print("Creating datasets...")
train_dataset = VisionDataset(dataset[train_split], processor)
eval_dataset = VisionDataset(dataset[eval_split], processor)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Eval dataset size: {len(eval_dataset)}")

# Debug: Check first example
print("\nChecking first example structure:")
first_example = dataset[train_split][0]
print(f"Keys in example: {first_example.keys()}")
print(f"Has image: {'image' in first_example and first_example['image'] is not None}")
print(f"Has image_url: {'image_url' in first_example and first_example['image_url'] is not None}")
if 'turns' in first_example:
    print(f"Turns keys: {first_example['turns'].keys()}")
    print(f"First query: {first_example['turns']['query'][0][:100]}...")
if 'answers' in first_example:
    print(f"Answers keys: {first_example['answers'].keys()}")
    print(f"First answer: {first_example['answers']['ans_full'][0][:100]}...")

# Debug: Test first batch
print("\nTesting first batch processing...")
try:
    test_example = train_dataset[0]
    print(f"First example keys: {test_example.keys()}")
    print(f"Input shape: {test_example['input_ids'].shape}")
    print(f"Labels shape: {test_example['labels'].shape}")
    if 'pixel_values' in test_example:
        print(f"Pixel values shape: {test_example['pixel_values'].shape}")
except Exception as e:
    print(f"Error processing first example: {e}")

# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=processor.tokenizer,
    model=model,
    padding=True,
    pad_to_multiple_of=8,
    return_tensors="pt"
)

# Training arguments - optimized for A100
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,  # Increased batch size
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,  # Effective batch size = 16
    gradient_checkpointing=True,
    optim="adamw_torch",  # Better optimizer for full precision
    logging_steps=10,
    learning_rate=5e-5,  # Slightly higher learning rate
    weight_decay=0.01,
    fp16=False,
    bf16=True,
    max_grad_norm=1.0,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=False,  # Disabled for vision dataset compatibility
    lr_scheduler_type="cosine",
    report_to="wandb" if use_wandb else "none",
    save_steps=100,
    save_total_limit=3,
    eval_strategy="steps",  # Changed from evaluation_strategy
    eval_steps=100,
    logging_first_step=True,
    push_to_hub=False,  # Set to True if you want to push to HuggingFace Hub
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    ddp_find_unused_parameters=False,
    dataloader_num_workers=2,  # Reduced for stability with image loading
    remove_unused_columns=False,
    label_names=["labels"],  # Explicitly set label names
)

# Custom trainer class to handle vision inputs properly
class VisionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        if labels is not None:
            print(f"\n--- Batch Labels Check (Before Model Forward Pass) ---")
            print(f"Labels dtype: {labels.dtype}, Labels device: {labels.device}")
            print(f"Labels shape: {labels.shape}")
            
            try:
                actual_token_labels = labels[labels != -100]
                if actual_token_labels.numel() > 0:
                    print(f"Labels min (excluding -100): {actual_token_labels.min().item()}")
                    print(f"Labels max (excluding -100): {actual_token_labels.max().item()}")
                else:
                    print("Labels tensor contains only -100 or is empty after filtering -100s.")
                print(f"Labels min (overall, including -100): {labels.min().item()}")
                print(f"Labels max (overall, including -100): {labels.max().item()}")
            except Exception as e:
                print(f"Error getting min/max of labels: {e}")

            model_vocab_size = None
            if hasattr(model.config, 'text_config') and hasattr(model.config.text_config, 'vocab_size'):
                model_vocab_size = model.config.text_config.vocab_size
            elif hasattr(model.config, 'vocab_size'):
                model_vocab_size = model.config.vocab_size
            
            if model_vocab_size is not None:
                print(f"Model's effective vocabulary size (n_classes): {model_vocab_size}")
            else:
                print(f"Warning: Could not reliably determine model's vocab_size from model.config.")
                # Assuming 'processor' is accessible here; if not, you might need to pass it or get vocab size differently
                # For this example, let's assume 'processor' is in the global scope or passed to the trainer.
                # If 'processor' is not defined here, you'll need to adjust how you get processor.tokenizer.vocab_size
                try:
                    print(f"Using processor.tokenizer.vocab_size as a fallback for check: {processor.tokenizer.vocab_size}")
                    model_vocab_size = processor.tokenizer.vocab_size 
                except NameError:
                    print("Error: 'processor' not defined in this scope to get tokenizer.vocab_size as fallback.")
                    # Handle this case, e.g., by not performing the check if processor isn't available
                    # or ensuring processor is available. For now, we'll proceed cautiously.
                    # Setting model_vocab_size to a very large number to avoid false positives if it's None
                    model_vocab_size = float('inf')


            if model_vocab_size != float('inf'): # Only proceed if model_vocab_size is valid
                print(f"Processor's tokenizer vocabulary size: {processor.tokenizer.vocab_size if 'processor' in globals() or 'processor' in locals() else 'processor not found'}")
                
                valid_indices_mask = (labels != -100)
                out_of_bounds_mask = (labels < 0) | (labels >= model_vocab_size)
                problematic_labels_mask = valid_indices_mask & out_of_bounds_mask
                
                if torch.any(problematic_labels_mask):
                    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    print(f"CRITICAL ERROR DETECTED: Invalid labels found in this batch!")
                    print(f"Values of invalid labels: {labels[problematic_labels_mask]}")
                    problematic_indices = problematic_labels_mask.nonzero(as_tuple=False)
                    print(f"Indices of first few invalid labels in this batch (up to 10):")
                    for i in range(min(10, problematic_indices.size(0))):
                        print(f"  - Index: {problematic_indices[i].tolist()}, Value: {labels[tuple(problematic_indices[i])].item()}")
                    input_ids = inputs.get("input_ids")
                    if input_ids is not None and input_ids.shape == labels.shape:
                        print(f"Corresponding input_ids for these invalid labels:")
                        for i in range(min(10, problematic_indices.size(0))):
                             print(f"  - Index: {problematic_indices[i].tolist()}, Input ID: {input_ids[tuple(problematic_indices[i])].item()}")
                    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                else:
                    print("All labels in this batch (not -100) appear to be within the valid range [0, vocab_size-1].")
            else:
                print("Skipping detailed label bound check as model_vocab_size could not be determined.")
            print(f"--- End Batch Labels Check ---")

        # Proceed with model computation
        outputs = model(**inputs) # CUDA error will likely occur here if labels are bad
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

# Initialize trainer
trainer = VisionTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

# Optional: Add custom callbacks
from transformers import TrainerCallback

class LossLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if 'loss' in logs:
                print(f"Step {state.global_step}: Loss = {logs['loss']:.4f}")

trainer.add_callback(LossLoggingCallback())

# Start training
print("Starting training...")
try:
    trainer.train()
except Exception as e:
    print(f"Training error: {e}")
    # Try with smaller batch size if OOM
    if "out of memory" in str(e).lower():
        print("Reducing batch size due to OOM...")
        training_args.per_device_train_batch_size = 2
        training_args.per_device_eval_batch_size = 2
        trainer = VisionTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        trainer.add_callback(LossLoggingCallback())
        trainer.train()
    else:
        raise e

# Save the final model
print("Saving model...")
trainer.save_model(OUTPUT_DIR)

# Save LoRA adapters separately
model.save_pretrained(f"{OUTPUT_DIR}/lora_adapters")
processor.save_pretrained(f"{OUTPUT_DIR}/processor")

# Push to Hub (optional)
# trainer.push_to_hub()

print("Training completed!")

# Merge LoRA weights with base model (optional)
def merge_lora_weights():
    """
    Merge LoRA weights with base model for easier deployment
    """
    from peft import PeftModel
    
    print("Loading base model...")
    base_model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa"  # Use PyTorch SDPA
    )
    
    print("Loading LoRA adapters...")
    model = PeftModel.from_pretrained(
        base_model,
        f"{OUTPUT_DIR}/lora_adapters"
    )
    
    print("Merging weights...")
    model = model.merge_and_unload()
    
    print("Saving merged model...")
    model.save_pretrained(f"{OUTPUT_DIR}/merged_model")
    
    return model

# Example inference code
def inference_example(image_path, query, use_merged_model=False):
    """
    Example function for inference with the fine-tuned model
    """
    if use_merged_model:
        model = AutoModelForVision2Seq.from_pretrained(
            f"{OUTPUT_DIR}/merged_model",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa"
        )
    else:
        from peft import PeftModel
        
        base_model = AutoModelForVision2Seq.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa"
        )
        
        model = PeftModel.from_pretrained(
            base_model,
            f"{OUTPUT_DIR}/lora_adapters"
        )
    
    model.eval()
    
    # Load processor
    processor = AutoProcessor.from_pretrained(f"{OUTPUT_DIR}/processor")
    
    # Prepare input
    image = Image.open(image_path)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": query}
            ]
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=text,
        images=image,
        return_tensors="pt"
    ).to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1
        )
    
    # Decode
    response = processor.decode(outputs[0], skip_special_tokens=True)
    
    # Extract assistant's response
    assistant_marker = "assistant\n\n"
    if assistant_marker in response:
        response = response.split(assistant_marker)[-1].strip()
    
    return response

# Optional: Evaluate on test set
def evaluate_on_test_set():
    """
    Run evaluation on a subset of the test set
    """
    from tqdm import tqdm
    import json
    
    results = []
    test_subset = eval_dataset.dataset.select(range(min(100, len(eval_dataset))))
    
    for idx, example in enumerate(tqdm(test_subset)):
        query = example['turns']['query'][0]
        ground_truth = example['answers']['ans_full'][0]
        
        # Get prediction
        prediction = inference_example(example['image'], query)
        
        results.append({
            'idx': idx,
            'query': query,
            'ground_truth': ground_truth,
            'prediction': prediction
        })
    
    # Save results
    with open(f"{OUTPUT_DIR}/evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return results