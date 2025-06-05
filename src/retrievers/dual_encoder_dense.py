import json
import torch
import faiss
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sentence_transformers import SentenceTransformer
from config import (
    PASSAGE_PATH,
    QUERY_PATH,
    LABSE_MODEL_DIR,
    DENSE_RESULT_PATH,
    RUNS_DIR,
    is_colab,
    is_kaggle,
    is_local
)

# -----------------------------
# Constants
# -----------------------------
TOP_K = 100
MODEL_NAME = "dense_dual_encoder"

# -----------------------------
# Load model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    model = SentenceTransformer(LABSE_MODEL_DIR).to(device)
except Exception as e:
    if is_colab:
        print("Error loading model in Colab. Please ensure you have mounted Google Drive and the model is in the correct location.")
    elif is_kaggle:
        print("Error loading model in Kaggle. Please ensure you have added the model dataset and it's in the correct location.")
    else:
        print(f"Error loading model: {str(e)}")
    raise

# -----------------------------
# Load and encode passages
# -----------------------------
try:
    with open(PASSAGE_PATH, 'r', encoding='utf-8') as f:
        passages = [json.loads(line) for line in f]
        pid_list = [p['pid'] for p in passages]
        text_list = [p['text'] for p in passages]
except FileNotFoundError:
    if is_colab:
        print(f"Passage file not found at {PASSAGE_PATH}. Please ensure you have mounted Google Drive and the file is in the correct location.")
    elif is_kaggle:
        print(f"Passage file not found at {PASSAGE_PATH}. Please ensure you have added the dataset and the file is in the correct location.")
    else:
        print(f"Passage file not found at {PASSAGE_PATH}. Please run passage extraction first.")
    raise

print("→ Encoding passages...")
try:
    passage_embeddings = model.encode(text_list, batch_size=64, show_progress_bar=True, convert_to_numpy=True, device=device)
except Exception as e:
    print(f"Error encoding passages: {str(e)}")
    if is_colab or is_kaggle:
        print("If you're running out of memory, try reducing the batch size or using a smaller model.")
    raise

# Create FAISS index
index = faiss.IndexFlatIP(passage_embeddings.shape[1])
faiss.normalize_L2(passage_embeddings)
index.add(passage_embeddings)

# -----------------------------
# Load queries and perform retrieval
# -----------------------------
try:
    with open(QUERY_PATH, 'r', encoding='utf-8') as f:
        queries = json.load(f)
except FileNotFoundError:
    if is_colab:
        print(f"Query file not found at {QUERY_PATH}. Please ensure you have mounted Google Drive and the file is in the correct location.")
    elif is_kaggle:
        print(f"Query file not found at {QUERY_PATH}. Please ensure you have added the dataset and the file is in the correct location.")
    else:
        print(f"Query file not found at {QUERY_PATH}. Please check if the file exists.")
    raise

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
# Save results
# -----------------------------
try:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with open(DENSE_RESULT_PATH, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"[✓] Dense encoder results saved to {DENSE_RESULT_PATH}")
except Exception as e:
    print(f"Error saving results: {str(e)}")
    if is_colab:
        print("Please ensure you have write permissions in your Google Drive.")
    elif is_kaggle:
        print("Please ensure you have write permissions in your Kaggle workspace.")
    raise
