#這個是dual encoder的程式
import json
import jieba
import torch
from rank_bm25 import BM25Okapi
from transformers import BertTokenizer, BertForSequenceClassification
from pathlib import Path
from tqdm import tqdm
import os

# -----------------------------
# 路徑設定
# -----------------------------
PASSAGE_PATH = Path("outputs/structured_passages.jsonl")
QUERY_PATH = Path("data/translated_query.json")
MODEL_DIR = Path("models/zhbert-finetuned")
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
# 載入 fine-tuned 中文 BERT 模型
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()

# -----------------------------
# 開始 BM25 初選 + reranking
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

        inputs = tokenizer(
            [query_text] * TOP_K,
            top_passages,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits.squeeze()

        scores = logits.tolist() if isinstance(logits, torch.Tensor) else logits
        reranked = sorted(zip(top_pids, scores), key=lambda x: x[1], reverse=True)

        for rank, (pid, score) in enumerate(reranked, 1):
            results.append({
                "qid": qid,
                "pid": pid,
                "rank": rank,
                "score": float(score),
                "model": f"bm25_rerank_{query_version}"
            })

    # 儲存檔案
    output_file = OUTPUT_DIR / f"bm25_rerank_{query_version}.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for r in results:
            json.dump(r, f, ensure_ascii=False)
            f.write('\n')

    print(f"✅ Saved: {output_file}")