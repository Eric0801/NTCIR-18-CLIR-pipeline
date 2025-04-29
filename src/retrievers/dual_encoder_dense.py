
import json
import torch
import faiss
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sentence_transformers import SentenceTransformer

# -----------------------------
# 路徑設定
# -----------------------------
PASSAGE_PATH = Path("./outputs/runs/structured_passages.jsonl")
QUERY_PATH = Path("./data/translated_query.json")
MODEL_DIR = "./models/labse"
OUTPUT_PATH = Path("./outputs/runs/dense_dual_encoder.jsonl")
TOP_K = 100
MODEL_NAME = "dense_dual_encoder"

# -----------------------------
# 載入模型
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer(MODEL_DIR).to(device)

# -----------------------------
# 載入 passages 並編碼
# -----------------------------
with open(PASSAGE_PATH, 'r', encoding='utf-8') as f:
    passages = [json.loads(line) for line in f]
    pid_list = [p['pid'] for p in passages]
    text_list = [p['text'] for p in passages]

print("→ Encoding passages...")
passage_embeddings = model.encode(text_list, batch_size=64, show_progress_bar=True, convert_to_numpy=True, device=device)

# 建立 FAISS index
index = faiss.IndexFlatIP(passage_embeddings.shape[1])
faiss.normalize_L2(passage_embeddings)
index.add(passage_embeddings)

# -----------------------------
# 載入 queries 並檢索
# -----------------------------
with open(QUERY_PATH, 'r', encoding='utf-8') as f:
    queries = json.load(f)

results = []
for q in tqdm(queries, desc="Dense retrieval"):
    qid = q['qid']
    query_text = q['query_en']
    query_vec = model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)[0].reshape(1, -1)
    scores, indices = index.search(query_vec, TOP_K)

    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
        results.append({
            "qid": qid,
            "pid": pid_list[idx],
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

print(f"[✓] Dense encoder results saved to {OUTPUT_PATH}")
