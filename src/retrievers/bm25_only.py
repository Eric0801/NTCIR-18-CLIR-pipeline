
import json
import jieba
from rank_bm25 import BM25Okapi
from pathlib import Path
from tqdm import tqdm

# -----------------------------
# 路徑設定
# -----------------------------
PASSAGE_PATH = Path("clir_pipeline/outputs/structured_passages.jsonl")
QUERY_PATH = Path("clir_pipeline/data/translated_query.json")
OUTPUT_PATH = Path("clir_pipeline/outputs/runs/bm25_only.jsonl")
USER_DICT_PATH = Path("clir_pipeline/data/userdict.txt")
TOP_K = 100
MODEL_NAME = "bm25_only"

# -----------------------------
# 分詞設定：支援自訂詞典
# -----------------------------
if USER_DICT_PATH.exists():
    print(f"🧠 Loading custom dict from: {USER_DICT_PATH}")
    jieba.load_userdict(str(USER_DICT_PATH))

# -----------------------------
# 載入 passages
# -----------------------------
with open(PASSAGE_PATH, 'r', encoding='utf-8') as f:
    passages = [json.loads(line) for line in f]
    corpus = [p['text'] for p in passages]
    pid_list = [p['pid'] for p in passages]

# -----------------------------
# 中文分詞（使用 jieba）
# -----------------------------
tokenized_corpus = [list(jieba.cut(text)) for text in corpus]
bm25 = BM25Okapi(tokenized_corpus)

# -----------------------------
# 載入 queries（使用 GPT 翻譯後的中文）
# -----------------------------
with open(QUERY_PATH, 'r', encoding='utf-8') as f:
    queries = json.load(f)

# -----------------------------
# 開始檢索
# -----------------------------
results = []
for q in tqdm(queries, desc="BM25 Retrieval"):
    qid = q['qid']
    query = q['query_zh_gpt']
    tokenized_query = list(jieba.cut(query))
    scores = bm25.get_scores(tokenized_query)
    topk_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:TOP_K]

    for rank, idx in enumerate(topk_idx, 1):
        results.append({
            "qid": qid,
            "pid": pid_list[idx],
            "score": float(scores[idx]),
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

print(f"[✓] Saved BM25 results to {OUTPUT_PATH}")
