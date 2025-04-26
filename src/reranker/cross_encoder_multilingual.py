
import json
import torch
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -----------------------------
# 路徑設定
# -----------------------------
PASSAGE_PATH = Path("clir_pipeline/outputs/structured_passages.jsonl")
QUERY_PATH = Path("clir_pipeline/data/translated_query.json")
MODEL_DIR = Path("models/cross_encoder")
OUTPUT_PATH = Path("clir_pipeline/outputs/runs/cross_encoder.jsonl")
TOP_K = 100
MODEL_NAME = "cross_encoder"

# -----------------------------
# 載入模型
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()

# -----------------------------
# 載入 passages 與 queries
# -----------------------------
with open(PASSAGE_PATH, 'r', encoding='utf-8') as f:
    passages = [json.loads(line) for line in f]
    pid_map = {p['pid']: p['text'] for p in passages}

with open(QUERY_PATH, 'r', encoding='utf-8') as f:
    queries = json.load(f)

# -----------------------------
# 載入 dense encoder 的 top-K 結果
# -----------------------------
with open("clir_pipeline/outputs/runs/dense_dual_encoder.jsonl", 'r', encoding='utf-8') as f:
    dense_results = [json.loads(line) for line in f]

# Group by qid
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

    pair_inputs = tokenizer(
        [query] * len(pid_scores),
        [pid_map[pid] for pid, _ in pid_scores],
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**pair_inputs)
        logits = outputs.logits[:, 1] if outputs.logits.shape[1] == 2 else outputs.logits.squeeze()

    reranked = sorted(zip([pid for pid, _ in pid_scores], logits.cpu().tolist()), key=lambda x: x[1], reverse=True)

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
