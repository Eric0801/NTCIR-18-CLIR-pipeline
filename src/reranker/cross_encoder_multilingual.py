import json
import torch
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# -----------------------------
# 路徑設定
# -----------------------------
PASSAGE_PATH = Path("/content/NTCIR-18-CLIR-pipeline-team6939/outputs/structured_passages.jsonl")
QUERY_PATH = Path("/content/NTCIR-18-CLIR-pipeline-team6939/data/translated_query.json")
MODEL_DIR = Path("/content/NTCIR-18-CLIR-pipeline-team6939/models/cross_encoder")
OUTPUT_PATH = Path("/content/NTCIR-18-CLIR-pipeline-team6939/outputs/runs/cross_encoder.jsonl")
TOP_K = 500
MODEL_NAME = "cross_encoder"
BATCH_SIZE = 64

# -----------------------------
# 載入模型（強化：驗證路徑與檔案）
# -----------------------------
assert MODEL_DIR.exists(), f"❌ 模型路徑不存在：{MODEL_DIR}"
assert (MODEL_DIR / "config.json").exists(), f"❌ 缺少 config.json"

try:
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR), local_files_only=True).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
except Exception as e:
    raise RuntimeError(f"❌ 無法載入 Cross Encoder 模型：{e}")

# -----------------------------
# 載入資料
# -----------------------------
with open(PASSAGE_PATH, 'r', encoding='utf-8') as f:
    passages = [json.loads(line) for line in f]
    pid_map = {p['pid']: p['text'] for p in passages}

with open(QUERY_PATH, 'r', encoding='utf-8') as f:
    queries = json.load(f)

with open("/content/NTCIR-18-CLIR-pipeline-team6939/outputs/runs/dense_dual_encoder.jsonl", 'r', encoding='utf-8') as f:
    dense_results = [json.loads(line) for line in f]

topk_by_qid = defaultdict(list)
for r in dense_results:
    topk_by_qid[r['qid']].append((r['pid'], r['score']))

# -----------------------------
# Rerank
# -----------------------------
results = []
for q in tqdm(queries, desc="Cross encoder reranking"):
    qid = q['qid']
    query = q['query_en']
    pid_scores = topk_by_qid[qid][:TOP_K]
    if not pid_scores:
        continue

    reranked = []
    for i in range(0, len(pid_scores), BATCH_SIZE):
        batch = pid_scores[i:i + BATCH_SIZE]
        batch_passages = [pid_map[pid] for pid, _ in batch if pid in pid_map]
        batch_pids = [pid for pid, _ in batch if pid in pid_map]
        if not batch_passages:
            continue

        inputs = tokenizer(
            [query] * len(batch_passages),
            batch_passages,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            logits = model(**inputs).logits
            scores = logits[:, 1].tolist() if logits.ndim == 2 and logits.shape[1] == 2 else logits.squeeze().tolist()
            if isinstance(scores, float):
                scores = [scores]

        reranked.extend(zip(batch_pids, scores))

    reranked = sorted(reranked, key=lambda x: x[1], reverse=True)
    for rank, (pid, score) in enumerate(reranked, 1):
        results.append({
            "qid": qid,
            "pid": pid,
            "score": float(score),
            "rank": rank,
            "model": MODEL_NAME
        })

# -----------------------------
# 儲存結果（含檔案確認）
# -----------------------------
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

if OUTPUT_PATH.exists():
    print(f"[✓] Cross encoder reranked results saved to {OUTPUT_PATH}")
else:
    print(f"[❌] 儲存失敗，找不到輸出檔案 {OUTPUT_PATH}")
