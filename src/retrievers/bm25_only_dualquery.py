import json
import jieba
from rank_bm25 import BM25Okapi
from pathlib import Path
from tqdm import tqdm
import os

# -----------------------------
# 路徑設定
# -----------------------------
PASSAGE_PATH = Path("outputs/structured_passages.jsonl")
QUERY_PATH = Path("data/translated_query.json")
OUTPUT_DIR = Path("outputs/runs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = 100

# -----------------------------
# 載入 passages 並斷詞
# -----------------------------
with open(PASSAGE_PATH, 'r', encoding='utf-8') as f:
    passages = [json.loads(line) for line in f]
    corpus = [p['text'] for p in passages]
    pid_list = [p['pid'] for p in passages]

tokenized_corpus = [list(jieba.cut(text)) for text in corpus]
bm25 = BM25Okapi(tokenized_corpus)

# -----------------------------
# 載入查詢
# -----------------------------
with open(QUERY_PATH, 'r', encoding='utf-8') as f:
    queries = json.load(f)

# -----------------------------
# 開始檢索
# -----------------------------
for query_version in ["query_zh_nmt", "query"]:  # 先 NMT，再原生繁中
    results = []

    for q in tqdm(queries, desc=f"Running BM25 baseline ({query_version})"):
        qid = q['qid']
        query_text = q.get(query_version, "")

        if not query_text:
            continue

        bm25_scores = bm25.get_scores(list(jieba.cut(query_text)))
        topk_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:TOP_K]

        for rank, idx in enumerate(topk_idx, 1):
            results.append({
                "qid": qid,
                "pid": pid_list[idx],
                "rank": rank,
                "score": float(bm25_scores[idx]),
                "model": f"bm25_only_{query_version}"
            })

    # 儲存檔案
    output_file = OUTPUT_DIR / f"bm25_only_{query_version}.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for r in results:
            json.dump(r, f, ensure_ascii=False)
            f.write('\n')

    print(f"✅ Saved: {output_file}")