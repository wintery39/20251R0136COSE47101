#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S3-VQA questions JSON ➜  <split>/<image_id> 텍스트 파일 생성
"""

import json

# ★ 파일 경로(필요 시 수정) ───────────────────────────────
train_json_path = "S3-VQA_train_questions.json"
dev_json_path   = "S3-VQA_dev_questions.json"
# ───────────────────────────────────────────────────────

# ── train 처리 ─────────────────────────────────────────
with open(train_json_path, "r", encoding="utf-8") as f:
    train_data = json.load(f)

with open("s3vqa_train_ids.txt", "w", encoding="utf-8") as f:
    for item in train_data:
        split   = item.get("split", "train")      # 없으면 train
        img_id  = item["image_id"].strip()        # 16-hex
        f.write(f"{split}/{img_id}\n")

# ── dev/validation 처리 ───────────────────────────────
with open(dev_json_path, "r", encoding="utf-8") as f:
    dev_data = json.load(f)

with open("s3vqa_dev_ids.txt", "w", encoding="utf-8") as f:
    for item in dev_data:
        split   = item.get("split", "validation") # dev 파일이면 보통 validation
        img_id  = item["image_id"].strip()
        f.write(f"{split}/{img_id}\n")

print(f"✔ Done! {len(train_data)} train IDs, {len(dev_data)} dev IDs extracted.")
