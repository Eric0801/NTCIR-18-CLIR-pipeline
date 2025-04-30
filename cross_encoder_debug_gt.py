
import json
import os
import torch
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -----------------------------
# 路徑設定
# -----------------------------
PASSAGE_PATH = Path("outputs/structured_passages.jsonl")
QUERY_PATH = Path("data/translated_query_nmt.json")
DENSE_RESULT_PATH = Path("outputs/runs/dense_dual_encoder.jsonl")
GROUND_TRUTH_PATH = Path("data/ground_truths_example.json")
MODEL_DIR = Path("models/cross_encoder")
TOP_K = 100
DEBUG_K = 5
MODEL_NAME = "cross_encoder_debug"

# -----------------------------
# 載入模型
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()

# -----------------------------
# 載入資料
# -----------------------------
with open(PASSAGE_PATH, 'r', encoding='utf-8') as f:
    passages = [json.loads(line) for line in f]
    pid_map = {str(p['pid']): p['text'] for p in passages}

with open(QUERY_PATH, 'r', encoding='utf-8') as f:
    queries = json.load(f)

with open(DENSE_RESULT_PATH, 'r', encoding='utf-8') as f:
    dense_results = [json.loads(line) for line in f]

with open(GROUND_TRUTH_PATH, 'r', encoding='utf-8') as f:
    gt = {str(g["qid"]): str(g["retrieve"]) for g in json.load(f)["ground_truths"]}

# -----------------------------
# Group topK results by qid
# -----------------------------
from collections import defaultdict
topk_by_qid = defaultdict(list)
for r in dense_results:
    topk_by_qid[str(r['qid'])].append((str(r['pid']), r['score']))

# -----------------------------
# Debug each query
# -----------------------------
total = len(gt)
hit = 0

for q in queries:
    qid = str(q['qid'])
    query_en = q.get("query_en", "").strip()
    if not query_en:
        continue

    pid_scores = topk_by_qid.get(qid, [])[:TOP_K]
    pids = [p[0] for p in pid_scores]
    gt_pid = gt.get(qid)

    if gt_pid in pids:
        hit += 1
        hit_flag = "✅"
    else:
        hit_flag = "❌"

    valid_pairs = [(pid, pid_map[pid]) for pid, _ in pid_scores if pid in pid_map]
    if not valid_pairs:
        print(f"❌ QID {qid} has no valid passages.")
        continue

    input_pairs = tokenizer(
        [query_en] * len(valid_pairs),
        [p[1] for p in valid_pairs],
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = model(**input_pairs).logits
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()

    print(f"\n🧪 QID {qid} ({hit_flag}) | GT PID: {gt_pid} | Query: {query_en}")
    top_debug = sorted(zip([p[0] for p in valid_pairs], probs), key=lambda x: x[1], reverse=True)[:DEBUG_K]
    for rank, (pid, score) in enumerate(top_debug, 1):
        ptext = pid_map[pid].replace("\n", " ")[:80]
        gt_mark = " ← 🎯" if pid == gt_pid else ""
        print(f"  #{rank}: PID={pid}, Score={score:.4f} ➜ {ptext}{gt_mark}")

print(f"\n📊 GT Hit Rate from Dense Top-{TOP_K}: {hit}/{total} = {hit/total:.2%}")
