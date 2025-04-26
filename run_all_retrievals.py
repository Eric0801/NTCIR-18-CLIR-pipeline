
import subprocess
import time

def run_all_retrievals():
    def timed_run(script_path, name):
        print(f"🚀 Running {name}...")
        start = time.time()
        subprocess.run(["python", script_path])
        duration = time.time() - start
        print(f"[⏱️] {name} took {duration:.2f} seconds.\n")
        return name, duration

    timings = []
    timings.append(timed_run("src/retrievers/bm25_only.py", "BM25 baseline"))
    timings.append(timed_run("src/reranker/reranker_zhbert.py", "BM25 + Chinese BERT reranker"))
    timings.append(timed_run("src/retrievers/dual_encoder_dense.py", "Multilingual Dual Encoder"))
    timings.append(timed_run("src/reranker/cross_encoder_multilingual.py", "Cross Encoder Reranker"))

    print("🧪 Retrieval Runtime Summary:")
    for name, duration in timings:
        print(f"{name:<40}: {duration:.2f} seconds")
