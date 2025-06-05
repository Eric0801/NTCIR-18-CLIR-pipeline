import subprocess
import time
import pandas as pd
from pathlib import Path
from config import (
    OUTPUTS_DIR,
    RUNS_DIR,
    LOGS_DIR,
    ensure_dir,
    is_colab,
    is_kaggle,
    is_local,
    ENVIRONMENT
)

def timed_run(script_path, name):
    """Run a script and measure its execution time."""
    print(f"🚀 Running {name} ...")
    start = time.time()
    try:
        subprocess.run(["python", script_path], check=True)
        duration = time.time() - start
        print(f"[⏱️] {name} finished in {duration:.2f} seconds.\n")
        return name, duration
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {name}: {e}")
        if is_colab:
            print("Please ensure you have mounted Google Drive and all required files are in the correct location.")
        elif is_kaggle:
            print("Please ensure you have added all required datasets and files are in the correct location.")
        else:
            print("Please check if all required files and dependencies are properly set up.")
        return name, -1
    except FileNotFoundError as e:
        print(f"❌ Script not found: {e}")
        return name, -1

def run_all_retrievals():
    """Run all retrieval models and collect timing information."""
    print(f"Running all retrievals in {ENVIRONMENT} environment...")
    
    # Ensure directories exist
    try:
        ensure_dir(OUTPUTS_DIR)
        ensure_dir(RUNS_DIR)
        ensure_dir(LOGS_DIR)
    except Exception as e:
        print(f"❌ Error creating directories: {e}")
        if is_colab:
            print("Please ensure you have write permissions in your Google Drive.")
        elif is_kaggle:
            print("Please ensure you have write permissions in your Kaggle workspace.")
        raise

    timings = []

    # --- Retrieval 模型列表 ---
    retrieval_scripts = [
        ("./src/retrievers/bm25_only_dualquery.py", "BM25 Only (Dual Query)"),
        ("./src/reranker/reranker_zhbert_dualquery.py", "BM25 + Chinese BERT Reranker (Dual Query)"),
        ("./src/retrievers/dual_encoder_dense.py", "Multilingual Dual Encoder (LaBSE)"),
        ("./src/reranker/cross_encoder_multilingual.py", "Multilingual Cross Encoder Reranker"),
    ]

    for script_path, name in retrieval_scripts:
        script_path = Path(script_path)
        if not script_path.exists():
            print(f"⚠️ Warning: Script {script_path} not found, skipping...")
            continue
        timings.append(timed_run(script_path, name))

    # --- Save runtime summary ---
    try:
        df = pd.DataFrame(timings, columns=["Model", "Runtime (seconds)"])
        runtime_csv_path = LOGS_DIR / "runtime_summary.csv"
        df.to_csv(runtime_csv_path, index=False)
        print(f"✅ Runtime summary saved to {runtime_csv_path}")
        print(f"Environment: {ENVIRONMENT}")
        print(df)
    except Exception as e:
        print(f"❌ Error saving runtime summary: {e}")
        if is_colab:
            print("Please ensure you have write permissions in your Google Drive.")
        elif is_kaggle:
            print("Please ensure you have write permissions in your Kaggle workspace.")
        raise

if __name__ == "__main__":
    run_all_retrievals()