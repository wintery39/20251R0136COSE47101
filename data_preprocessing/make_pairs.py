#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S3-VQA  ➜  단일 쌍(이미지·질문·정답) JSONL 변환
"""

import json, argparse, pathlib, sys
from tqdm import tqdm

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main(q_path, a_path, img_dir, out_path):
    questions   = load_json(q_path)
    annotations = {a["question_id"]: a for a in load_json(a_path)}

    img_dir = pathlib.Path(img_dir)
    n_total, n_ok, n_skip = 0, 0, 0

    with open(out_path, "w", encoding="utf-8") as fo:
        for q in tqdm(questions, desc="pairing"):
            n_total += 1
            qid      = q["question_id"]
            img_id   = q["image_id"]
            sent     = q["question"]

            ann = annotations.get(qid)
            if ann is None:
                n_skip += 1
                continue

            img_file = img_dir / f"{img_id}.jpg"
            if not img_file.exists():
                n_skip += 1
                continue

            rec = {
                "image":           str(img_file),
                "interaction_id":  qid,
                "question":        sent,
                "answer":          ann["answer"]["raw"]   # 필요하면 .lower() 등 전처리
            }
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_ok += 1

    print(f"[✓] written {n_ok:,} pairs  (skipped {n_skip:,} / {n_total:,}) → {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions",   required=True, help="S3-VQA_*_questions.json")
    ap.add_argument("--annotations", required=True, help="S3-VQA_*_annotations.json")
    ap.add_argument("--img_dir",     default="open_images/train")
    ap.add_argument("--out",         default="s3vqa_train_pairs.jsonl")
    args = ap.parse_args()

    main(args.questions, args.annotations, args.img_dir, args.out)
