import subprocess
import os
import time
import pandas as pd

# 基本資料夾
OUTPUT_DIR = "outputs"
METRIC_DIR = "outputs/metrics"
RUNS_DIR = "outputs/runs"
os.makedirs(METRIC_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)

def timed_run(script_path, name):
    print(f"🚀 Running {name} ...")
    start = time.time()
    subprocess.run(["python", script_path])
    duration = time.time() - start
    print(f"[⏱️] {name} finished in {duration:.2f} seconds.\n")
    return name, duration

def run_all_retrievals():
    timings = []

    # --- Retrieval 模型列表 ---
    retrieval_scripts = [
        ("src/retrievers/bm25_only_dualquery.py", "BM25 Only (Dual Query)"),
        ("src/reranker/reranker_zhbert_dualquery.py", "BM25 + Chinese BERT Reranker (Dual Query)"),
        ("src/retrievers/dual_encoder_dense.py", "Multilingual Dual Encoder (LaBSE)"),
        ("src/reranker/cross_encoder_multilingual.py", "Multilingual Cross Encoder Reranker"),
    ]

    for script_path, name in retrieval_scripts:
        timings.append(timed_run(script_path, name))

    # --- Save runtime summary ---
    df = pd.DataFrame(timings, columns=["Model", "Runtime (seconds)"])
    runtime_csv_path = os.path.join(METRIC_DIR, "runtime_summary.csv")
    df.to_csv(runtime_csv_path, index=False)
    print(f"✅ Runtime summary saved to {runtime_csv_path}")
    print(df)

if __name__ == "__main__":
    run_all_retrievals()