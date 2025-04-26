
import json
import jieba
import torch
from rank_bm25 import BM25Okapi
from transformers import BertTokenizer, BertForSequenceClassification
from pathlib import Path
from tqdm import tqdm

# -----------------------------
# 路徑設定
# -----------------------------
PASSAGE_PATH = Path("clir_pipeline/outputs/structured_passages.jsonl")
QUERY_PATH = Path("clir_pipeline/data/translated_query.json")
MODEL_DIR = Path("models/zhbert")
OUTPUT_PATH = Path("clir_pipeline/outputs/runs/bm25_rerank.jsonl")
TOP_K = 100
MODEL_NAME = "bm25_rerank"

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
# 載入查詢（英文欄位）
# -----------------------------
with open(QUERY_PATH, 'r', encoding='utf-8') as f:
    queries = json.load(f)

# -----------------------------
# 載入中文 BERT 模型
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()

# -----------------------------
# 開始 BM25 + rerank
# -----------------------------
results = []
for q in tqdm(queries, desc="BM25 + BERT Reranking"):
    qid = q['qid']
    query_en = q['query_en']
    bm25_scores = bm25.get_scores(list(jieba.cut(query_en)))
    topk_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:TOP_K]
    top_passages = [corpus[i] for i in topk_idx]
    top_pids = [pid_list[i] for i in topk_idx]

    # 對 topK 使用 BERT rerank
    inputs = tokenizer(
        [query_en] * TOP_K, 
        top_passages, 
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[:, 1] if outputs.logits.shape[1] == 2 else outputs.logits.squeeze()

    reranked = sorted(zip(top_pids, logits.cpu().tolist()), key=lambda x: x[1], reverse=True)
    
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

print(f"[✓] BM25 + Reranker results saved to {OUTPUT_PATH}")
