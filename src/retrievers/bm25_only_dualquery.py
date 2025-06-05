import json
import jieba
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from config import (
    PASSAGE_PATH,
    QUERY_PATH,
    RUNS_DIR,
    OUTPUT_PATHS,
    ensure_dir,
    ENVIRONMENT
)

# -----------------------------
# Constants
# -----------------------------
TOP_K = 100
MODEL_NAME = "bm25"

def load_passages():
    """Load and tokenize passages."""
    try:
        with open(PASSAGE_PATH, 'r', encoding='utf-8') as f:
            passages = [json.loads(line) for line in f]
            corpus = [p['text'] for p in passages]
            pid_list = [p['pid'] for p in passages]
        return passages, corpus, pid_list
    except FileNotFoundError:
        raise FileNotFoundError(f"Passage file not found at {PASSAGE_PATH}. Please run passage extraction first.")

def load_queries():
    """Load queries."""
    try:
        with open(QUERY_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Query file not found at {QUERY_PATH}. Please check if the file exists.")

def main():
    """Run BM25 retrieval with both NMT and original queries."""
    print(f"Running BM25 retrieval in {ENVIRONMENT} environment...")
    ensure_dir(RUNS_DIR)

    # Load and prepare data
    passages, corpus, pid_list = load_passages()
    queries = load_queries()

    # Tokenize corpus
    print("Tokenizing corpus...")
    tokenized_corpus = [list(jieba.cut(text)) for text in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    # Process each query version
    for query_version in ["query_zh_nmt", "query"]:  # First NMT, then original Traditional Chinese
        results = []
        output_file = OUTPUT_PATHS['bm25']

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
                    "model": f"{MODEL_NAME}_{query_version}"
                })

        # Save results
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for r in results:
                    json.dump(r, f, ensure_ascii=False)
                    f.write('\n')
            print(f"✅ Saved: {output_file}")
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            raise

if __name__ == "__main__":
    main()