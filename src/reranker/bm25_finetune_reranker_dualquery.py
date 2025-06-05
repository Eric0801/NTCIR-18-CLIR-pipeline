# ✅ Chinese BERT Reranker with BM25 (Dual Query Version)
import json
import jieba
import torch
from rank_bm25 import BM25Okapi
from transformers import BertTokenizer, BertForSequenceClassification
from pathlib import Path
from tqdm import tqdm
from config import ZHBERT_MODEL_DIR, PASSAGE_PATH, QUERY_PATH, RUNS_DIR, ensure_dir

# -----------------------------
# 路徑設定
# -----------------------------
TOP_K = 500
BATCH_SIZE = 64
ensure_dir(RUNS_DIR)

# -----------------------------
# 載入模型（含驗證與例外處理）
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert ZHBERT_MODEL_DIR.exists(), f"❌ 模型路徑不存在：{ZHBERT_MODEL_DIR}"
assert (ZHBERT_MODEL_DIR / "config.json").exists(), "❌ 缺少 config.json"

try:
    tokenizer = BertTokenizer.from_pretrained(str(ZHBERT_MODEL_DIR), local_files_only=True)
    model = BertForSequenceClassification.from_pretrained(str(ZHBERT_MODEL_DIR), local_files_only=True).to(device)
    model.eval()
except Exception as e:
    raise RuntimeError(f"❌ 無法載入 zhBERT 模型：{e}")

# -----------------------------
# 載入資料與建 BM25
# -----------------------------
with open(PASSAGE_PATH, 'r', encoding='utf-8') as f:
    passages = [json.loads(line) for line in f]
    corpus = [p['text'] for p in passages]
    pid_list = [p['pid'] for p in passages]
tokenized_corpus = [list(jieba.cut(text)) for text in corpus]
bm25 = BM25Okapi(tokenized_corpus)

with open(QUERY_PATH, 'r', encoding='utf-8') as f:
    queries = json.load(f)

# -----------------------------
# 開始 reranking
# -----------------------------
for query_version in ["query_zh_nmt", "query"]:
    results = []
    for q in tqdm(queries, desc=f"Running BM25 + Reranker ({query_version})"):
        qid = q['qid']
        query_text = q.get(query_version, "")
        if not query_text:
            continue

        bm25_scores = bm25.get_scores(list(jieba.cut(query_text)))
        topk_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:TOP_K]
        top_passages = [corpus[i] for i in topk_idx]
        top_pids = [pid_list[i] for i in topk_idx]

        reranked = []
        for i in range(0, TOP_K, BATCH_SIZE):
            batch_passages = top_passages[i:i+BATCH_SIZE]
            batch_pids = top_pids[i:i+BATCH_SIZE]
            inputs = tokenizer(
                [query_text] * len(batch_passages),
                batch_passages,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
                return_overflowing_tokens=False
            ).to(device)

            with torch.no_grad():
                logits = model(**inputs).logits
                if logits.ndim == 2 and logits.shape[1] == 2:
                    scores = torch.softmax(logits, dim=1)[:, 1].tolist()
                else:
                    scores = logits.squeeze().tolist()
                if isinstance(scores, float):
                    scores = [scores]
            reranked.extend(zip(batch_pids, scores))

        reranked = sorted(reranked, key=lambda x: x[1], reverse=True)
        for rank, (pid, score) in enumerate(reranked, 1):
            results.append({
                "qid": qid,
                "pid": pid,
                "rank": rank,
                "score": float(score),
                "model": f"bm25_rerank_{query_version}"
            })

    output_file = RUNS_DIR / f"bm25_rerank_{query_version}.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print(f"✅ Saved: {output_file}")