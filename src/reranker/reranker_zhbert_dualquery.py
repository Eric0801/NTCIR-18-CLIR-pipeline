#這個是dual encoder的程式
import json
import jieba
import torch
from rank_bm25 import BM25Okapi
from transformers import BertTokenizer, BertForSequenceClassification
from pathlib import Path
from tqdm import tqdm
from config import (
    PASSAGE_PATH,
    QUERY_PATH,
    ZH_BERT_MODEL_DIR,
    RUNS_DIR,
    ensure_dir,
    is_colab,
    is_kaggle,
    is_local,
    ENVIRONMENT
)

# -----------------------------
# Constants
# -----------------------------
TOP_K = 100

def load_passages():
    """Load and tokenize passages."""
    try:
        with open(PASSAGE_PATH, 'r', encoding='utf-8') as f:
            passages = [json.loads(line) for line in f]
            corpus = [p['text'] for p in passages]
            pid_list = [p['pid'] for p in passages]
        return passages, corpus, pid_list
    except FileNotFoundError:
        if is_colab:
            print(f"Passage file not found at {PASSAGE_PATH}. Please ensure you have mounted Google Drive and the file is in the correct location.")
        elif is_kaggle:
            print(f"Passage file not found at {PASSAGE_PATH}. Please ensure you have added the dataset and the file is in the correct location.")
        else:
            print(f"Passage file not found at {PASSAGE_PATH}. Please run passage extraction first.")
        raise

def load_queries():
    """Load queries."""
    try:
        with open(QUERY_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        if is_colab:
            print(f"Query file not found at {QUERY_PATH}. Please ensure you have mounted Google Drive and the file is in the correct location.")
        elif is_kaggle:
            print(f"Query file not found at {QUERY_PATH}. Please ensure you have added the dataset and the file is in the correct location.")
        else:
            print(f"Query file not found at {QUERY_PATH}. Please check if the file exists.")
        raise

def main():
    """Run BM25 + Reranker retrieval with both NMT and original queries."""
    print(f"Running BM25 + Reranker in {ENVIRONMENT} environment...")
    ensure_dir(RUNS_DIR)

    # Load and prepare data
    passages, corpus, pid_list = load_passages()
    queries = load_queries()

    # Tokenize corpus
    print("Tokenizing corpus...")
    tokenized_corpus = [list(jieba.cut(text)) for text in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    # Load model
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = BertTokenizer.from_pretrained(ZH_BERT_MODEL_DIR)
        model = BertForSequenceClassification.from_pretrained(ZH_BERT_MODEL_DIR).to(device)
        model.eval()
    except Exception as e:
        if is_colab:
            print(f"Error loading model in Colab: {str(e)}")
            print("Please ensure you have mounted Google Drive and the model is in the correct location.")
        elif is_kaggle:
            print(f"Error loading model in Kaggle: {str(e)}")
            print("Please ensure you have added the model dataset and it's in the correct location.")
        else:
            print(f"Error loading model: {str(e)}")
        raise

    # Process each query version
    for query_version in ["query_zh_nmt", "query"]:  # First NMT, then original Traditional Chinese
        results = []
        output_file = RUNS_DIR / f"bm25_rerank_{query_version}.jsonl"

        for q in tqdm(queries, desc=f"Running BM25 + Reranker ({query_version})"):
            qid = q['qid']
            query_text = q.get(query_version, "")

            if not query_text:
                continue

            bm25_scores = bm25.get_scores(list(jieba.cut(query_text)))
            topk_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:TOP_K]

            top_passages = [corpus[i] for i in topk_idx]
            top_pids = [pid_list[i] for i in topk_idx]

            try:
                inputs = tokenizer(
                    [query_text] * TOP_K,
                    top_passages,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(device)

                with torch.no_grad():
                    logits = model(**inputs).logits
                    scores = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()

                reranked = sorted(zip(top_pids, scores), key=lambda x: x[1], reverse=True)

                for rank, (pid, score) in enumerate(reranked, 1):
                    results.append({
                        "qid": qid,
                        "pid": pid,
                        "rank": rank,
                        "score": float(score),
                        "model": f"bm25_rerank_{query_version}"
                    })
            except Exception as e:
                print(f"Error processing query {qid}: {str(e)}")
                if is_colab or is_kaggle:
                    print("If you're running out of memory, try reducing TOP_K or batch size.")
                continue

        # Save results
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for r in results:
                    json.dump(r, f, ensure_ascii=False)
                    f.write('\n')
            print(f"✅ Saved: {output_file}")
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            if is_colab:
                print("Please ensure you have write permissions in your Google Drive.")
            elif is_kaggle:
                print("Please ensure you have write permissions in your Kaggle workspace.")
            raise

if __name__ == "__main__":
    main()