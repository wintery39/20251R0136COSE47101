import os
import json # 스키마 확인용으로 남겨둘 수 있으나, 주 데이터 로딩에는 사용 안함
from PIL import Image
import torch
from datasets import load_dataset, concatenate_datasets # concatenate_datasets는 여러 스플릿 병합 시 사용 가능
import functools # functools.partial 사용

from transformers import (
    AutoProcessor,
    MllamaForConditionalGeneration,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
)
from peft import LoraConfig, get_peft_model # prepare_model_for_kbit_training은 양자화 시 필요

# --- 1. Configuration ---
MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"
HF_DATASET_ID = "crag-mm-2025/crag-mm-single_turn-public" # 단일 턴 데이터셋 사용
HF_DATASET_REVISION = "v0.1.1"
# TODO: 사용자가 실제 학습에 사용할 데이터셋 스플릿 이름을 확인하고 설정해야 합니다.
# 예: "train", "public_train". "validation"은 보통 검증용.
# 여러 스플릿을 합쳐서 사용하려면 아래 main 함수 내 로직 수정.
HF_DATASET_SPLIT = "train" # 기본값, 실제 사용 가능한 스플릿으로 변경 필요

OUTPUT_DIR = "./llama3_2_11b_vision_finetuned_crag_mm" # 파인튜닝된 모델 저장 경로



MAX_SEQ_LENGTH = 1024 # <--- [사용자 검토/수정 가능 3]: 데이터의 평균적인 길이와 GPU 메모리 상황을 보고 조절. A100 40GB에서는 1024도 괜찮을 수 있으나, 더 길면 메모리 부족 가능성.
LEARNING_RATE = 1e-4
BATCH_SIZE_PER_DEVICE = 1 # <--- [사용자 검토/수정 가능 4]: A100 40GB에서 11B 모델은 1이 안전. OOM 발생 시 줄일 수는 없으므로, 대신 GRAD_ACCUM_STEPS 늘림. 만약 메모리가 남으면 2도 시도해볼 수 있으나 매우 신중해야 함.
GRADIENT_ACCUMULATION_STEPS = 8 # <--- [사용자 검토/수정 가능 5]: 실질적 배치 크기(BATCH_SIZE * GRAD_ACCUM_STEPS)를 조절. 현재 1*8=8. GPU 메모리 부족 없이 학습 속도를 높이고 싶다면 이 값을 늘려서 실질적 배치 크기를 16, 32 등으로 조절.
NUM_TRAIN_EPOCHS = 1 # <--- [사용자 검토/수정 가능 6]: 1.6GB 데이터셋이면 1 에포크도 꽤 학습량이 될 수 있음. 모델 성능을 보면서 2~3 에포크까지 늘려볼 수 있음.
LOGGING_STEPS = 10
SAVE_STEPS = 100 # <--- [사용자 검토/수정 가능 7]: 데이터셋 크기에 따라 조절. 1.6GB면 샘플 수가 많을 것이므로, 100~500 스텝 정도로 시작. 너무 자주 저장하면 디스크 공간 차지.
BF16_ENABLED = torch.cuda.is_bf16_supported() # A100은 bfloat16 지원하므로 True로 설정됨.

# LoRA 설정
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# --- 2. Preprocessing Function for Hugging Face Dataset ---
def preprocess_dataset_function(examples, processor, max_length):
    all_input_ids = []
    all_attention_masks = []
    all_pixel_values = []
    all_labels = []
    
    skipped_samples_count = 0

    if not examples or not examples.get(list(examples.keys())[0]): # examples가 비었거나, 첫번째 키의 값이 비었으면
        return {"input_ids": torch.tensor([]), "attention_mask": torch.tensor([]), 
                "pixel_values": torch.tensor([]), "labels": torch.tensor([])}
        
    num_samples_in_batch = len(examples[list(examples.keys())[0]])

    for i in range(num_samples_in_batch):
        try:
            current_image = examples['image'][i]
            turns_data = examples['turns'][i]
            answers_data = examples['answers'][i]

            # 이미지 유효성 검사
            if not isinstance(current_image, Image.Image):
                # 데이터셋 설명에 따르면 'image' 필드는 PIL.Image 객체여야 함
                # 만약 None이고 image_url이 있다면, 학습 데이터에서는 직접 사용하기 어려움
                # (평가 서버는 image 필드를 채워준다고 명시됨)
                print(f"Warning: Sample at batch index {i} 'image' is not a PIL Image (type: {type(current_image)}). Skipping.")
                skipped_samples_count += 1
                continue
            
            if current_image.mode != 'RGB':
                current_image = current_image.convert('RGB')

            # 'turns'와 'answers' 구조 및 내용 유효성 검사 (단일 턴 가정)
            if not (isinstance(turns_data, list) and len(turns_data) > 0 and
                    isinstance(answers_data, list) and len(answers_data) > 0 and
                    'query' in turns_data[0] and isinstance(turns_data[0]['query'], str) and
                    'ans_full' in answers_data[0] and isinstance(answers_data[0]['ans_full'], str) and
                    turns_data[0]['query'].strip() and answers_data[0]['ans_full'].strip() ): # 내용이 비어있지 않은지 확인
                print(f"Warning: Sample at batch index {i} has malformed, incomplete, or empty 'turns'/'answers' content. Skipping.")
                skipped_samples_count += 1
                continue
            
            user_query = turns_data[0]['query']
            assistant_answer = answers_data[0]['ans_full']

            messages = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_query}]},
                {"role": "assistant", "content": [{"type": "text", "text": assistant_answer}]}
            ]
            
            prompt_text = processor.apply_chat_template(
                messages,
                add_generation_prompt=False, # 학습 시에는 False
                tokenize=False
            )

            inputs = processor(
                text=prompt_text,
                images=current_image,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_length
            )

            all_input_ids.append(inputs["input_ids"].squeeze(0))
            all_attention_masks.append(inputs["attention_mask"].squeeze(0))
            all_pixel_values.append(inputs["pixel_values"].squeeze(0))
            all_labels.append(inputs["input_ids"].squeeze(0).clone())

        except Exception as e:
            print(f"Error processing sample at batch index {i}: {e}. Skipping this sample.")
            # print(f"Problematic sample 'session_id': {examples.get('session_id', ['N/A'])[i] if 'session_id' in examples and i < len(examples['session_id']) else 'N/A'}")
            skipped_samples_count += 1
            continue
    
    if skipped_samples_count == num_samples_in_batch and num_samples_in_batch > 0:
        return {"input_ids": torch.tensor([]), "attention_mask": torch.tensor([]), 
                "pixel_values": torch.tensor([]), "labels": torch.tensor([])}

    # 스킵되지 않은 샘플이 하나라도 있을 경우에만 stack 수행
    if not all_input_ids: # 모든 샘플이 스킵된 경우 (num_samples_in_batch가 0이었거나, 모두 스킵)
         return {"input_ids": torch.tensor([]), "attention_mask": torch.tensor([]), 
                "pixel_values": torch.tensor([]), "labels": torch.tensor([])}

    try:
        batch_features = {
            "input_ids": torch.stack(all_input_ids),
            "attention_mask": torch.stack(all_attention_masks),
            "pixel_values": torch.stack(all_pixel_values),
            "labels": torch.stack(all_labels),
        }
        return batch_features
    except RuntimeError as e: # 스택 오류 (예: 빈 리스트)
        print(f"Error stacking tensors, likely due to all samples in batch being skipped or other issues: {e}")
        return {"input_ids": torch.tensor([]), "attention_mask": torch.tensor([]), 
                "pixel_values": torch.tensor([]), "labels": torch.tensor([])}


# --- 3. Main Fine-tuning Logic ---
def main():
    print(f"PyTorch version: {torch.__version__}")
    current_time = torch.cuda.Event(enable_timing=True)
    print(f"Current time from Pytorch: {current_time}") # 이 부분은 시간 측정용이므로, 실제 시간 출력에는 적합하지 않습니다.

    if not torch.cuda.is_available():
        print("ERROR: CUDA (GPU) is not available. This script requires a GPU.")
        return
    else:
        print(f"CUDA available. Device: {torch.cuda.get_device_name(0)}")

    # 프로세서 로드
    print(f"Loading processor for {MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    if processor.tokenizer.pad_token is None:
        print("Setting pad_token to eos_token for tokenizer.")
        processor.tokenizer.pad_token = processor.tokenizer.eos_token # 일반적인 처리

    # 모델 로드
    print(f"Loading model {MODEL_ID}...")
    model = MllamaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if BF16_ENABLED else torch.float16,
        device_map="auto",
    )

    # LoRA 설정
    print("Applying LoRA configuration...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Hugging Face 데이터셋 로드
    print(f"Loading Hugging Face dataset: {HF_DATASET_ID}, revision: {HF_DATASET_REVISION}...")
    try:
        raw_datasets = load_dataset(HF_DATASET_ID, revision=HF_DATASET_REVISION)
        print(f"Successfully loaded dataset. Available splits: {list(raw_datasets.keys())}")
    except Exception as e:
        print(f"Failed to load dataset {HF_DATASET_ID} with revision {HF_DATASET_REVISION}. "
              f"Ensure you are logged in (`huggingface-cli login`) and have access. Error: {e}")
        return

    # 사용할 학습 스플릿 결정
    train_split_name_to_use = ""
    if HF_DATASET_SPLIT in raw_datasets:
        train_split_name_to_use = HF_DATASET_SPLIT
    elif "train" in raw_datasets: # 기본 'train' 스플릿 시도
        train_split_name_to_use = "train"
        print(f"Warning: Specified split '{HF_DATASET_SPLIT}' not found. Using 'train' split instead.")
    elif "public_train" in raw_datasets: # CRAG-MM 대회에서 사용될 수 있는 스플릿 이름
        train_split_name_to_use = "public_train"
        print(f"Warning: Specified split '{HF_DATASET_SPLIT}' not found. Using 'public_train' split instead.")
    elif list(raw_datasets.keys()): # 사용 가능한 첫 번째 스플릿 사용 (최후의 수단)
        train_split_name_to_use = list(raw_datasets.keys())[0]
        print(f"Warning: Specified split '{HF_DATASET_SPLIT}' and common train splits not found. Using first available split: '{train_split_name_to_use}'. Please verify this is correct for training.")
    else:
        print(f"Error: No splits found in dataset {HF_DATASET_ID}. Cannot proceed.")
        return
    
    print(f"Using '{train_split_name_to_use}' split for training.")
    raw_train_dataset = raw_datasets[train_split_name_to_use]

    print(f"Raw train dataset features: {raw_train_dataset.features}")
    if len(raw_train_dataset) > 0:
        print(f"First sample of raw train dataset (session_id): {raw_train_dataset[0]['session_id']}")
    else:
        print(f"Warning: The selected train split '{train_split_name_to_use}' is empty.")
        return

    _preprocess_with_args = functools.partial(
        preprocess_dataset_function,
        processor=processor,
        max_length=MAX_SEQ_LENGTH
    )

    print("Preprocessing dataset with .map()...")
    original_columns = raw_train_dataset.column_names
    
    tokenized_train_dataset = raw_train_dataset.map(
        _preprocess_with_args,
        batched=True,
        remove_columns=original_columns,
        num_proc=max(1, os.cpu_count() // 2 if os.cpu_count() else 1),
        load_from_cache_file=True,
        desc=f"Preprocessing {train_split_name_to_use} split",
    )
    
    # .map() 후 빈 배치가 생성되었을 수 있으므로, 실제 데이터가 있는 샘플만 필터링
    def filter_non_empty_samples(example):
        # preprocess_dataset_function이 빈 텐서를 반환할 경우 input_ids의 shape[0]이 0이 됨
        return example['input_ids'].shape[0] > 0 if isinstance(example['input_ids'], torch.Tensor) else len(example['input_ids']) > 0


    final_train_dataset = tokenized_train_dataset.filter(filter_non_empty_samples)


    if len(final_train_dataset) == 0:
        print("No data remaining after preprocessing and filtering. Check dataset and preprocessing logic. Exiting.")
        return

    print(f"Processed dataset ready. Number of samples: {len(final_train_dataset)}")
    # print(f"Features of processed dataset: {final_train_dataset.features}") # .map()에서 remove_columns를 하면 features가 바뀜
    if len(final_train_dataset) > 0:
         print(f"First sample of processed dataset (keys): {final_train_dataset[0].keys()}")


    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE_PER_DEVICE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_strategy="steps",
        logging_steps=LOGGING_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=2, # 이전 체크포인트는 2개까지만 유지
        bf16=BF16_ENABLED,
        fp16=not BF16_ENABLED and torch.cuda.is_available(),
        gradient_checkpointing=True,
        report_to="tensorboard", # 또는 "wandb"
        remove_unused_columns=False, # map에서 이미 처리됨
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=final_train_dataset,
        tokenizer=processor.tokenizer, # 내부 로깅 등에 필요
        data_collator=DefaultDataCollator(),
    )

    print("Starting fine-tuning...")
    trainer.train()

    print(f"Saving final LoRA adapter to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    print(f"Fine-tuning finished successfully! Model saved to {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()