import json
import os
import torch
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import PASSAGE_PATH, QUERY_PATH, RUNS_DIR, ensure_dir, CROSS_ENCODER_MODEL_DIR

# -----------------------------
# Path setting
# -----------------------------
DENSE_RESULT_PATH = RUNS_DIR / "dense_dual_encoder.jsonl"
OUTPUT_PATH = RUNS_DIR / "cross_encoder.jsonl"
TOP_K = 100
MODEL_NAME = "cross_encoder"

# -----------------------------
# 載入模型（加上 local fallback 判斷）
# -----------------------------

# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(CROSS_ENCODER_MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(CROSS_ENCODER_MODEL_DIR).to(device)
model.eval()

# -----------------------------
# 載入 passages 與 queries
# -----------------------------
if not os.path.exists(PASSAGE_PATH):
    raise FileNotFoundError(f"❌ Cannot find passage file at {PASSAGE_PATH}. Did you run passage extraction?")
if not os.path.exists(DENSE_RESULT_PATH):
    raise FileNotFoundError(f"❌ Cannot find dense retrieval result at {DENSE_RESULT_PATH}.")

with open(PASSAGE_PATH, 'r', encoding='utf-8') as f:
    passages = [json.loads(line) for line in f]
    pid_map = {p['pid']: p['text'] for p in passages}

with open(QUERY_PATH, 'r', encoding='utf-8') as f:
    queries = json.load(f)

with open(DENSE_RESULT_PATH, 'r', encoding='utf-8') as f:
    dense_results = [json.loads(line) for line in f]

# -----------------------------
# 整理 top-K 預測結果
# -----------------------------
from collections import defaultdict
topk_by_qid = defaultdict(list)
for r in dense_results:
    topk_by_qid[r['qid']].append((r['pid'], r['score']))

# -----------------------------
# Rerank using cross-encoder
# -----------------------------
results = []
for q in tqdm(queries, desc="Cross encoder reranking"):
    qid = q['qid']
    query = q['query_en']
    pid_scores = topk_by_qid[qid][:TOP_K]

    if not pid_scores:
        continue

    pair_inputs = tokenizer(
        [query] * len(pid_scores),
        [pid_map.get(pid, "") for pid, _ in pid_scores],
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = model(**pair_inputs).logits
        scores = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()

    reranked = sorted(zip([pid for pid, _ in pid_scores], scores), key=lambda x: x[1], reverse=True)

    for rank, (pid, score) in enumerate(reranked, 1):
        results.append({
            "qid": qid,
            "pid": pid,
            "score": float(score),
            "rank": rank,
            "model": MODEL_NAME
        })

# -----------------------------
# 儲存結果
# -----------------------------
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

print(f"[✓] Cross encoder reranked results saved to {OUTPUT_PATH}")
