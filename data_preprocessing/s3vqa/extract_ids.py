import json

# 파일 경로 (필요시 수정)
train_json_path = "S3-VQA_train_annotations.json"
dev_json_path = "S3-VQA_dev_annotations.json"

# train 처리
with open(train_json_path, "r") as f:
    train_data = json.load(f)

train_ids = [item["question_id"] for item in train_data]
with open("s3vqa_train_ids.txt", "w") as f:
    for qid in train_ids:
        f.write(f"{qid}\n")

# dev 처리
with open(dev_json_path, "r") as f:
    dev_data = json.load(f)

dev_ids = [item["question_id"] for item in dev_data]
with open("s3vqa_dev_ids.txt", "w") as f:
    for qid in dev_ids:
        f.write(f"{qid}\n")

print(f"✔ Done! {len(train_ids)} train IDs, {len(dev_ids)} dev IDs extracted.")
