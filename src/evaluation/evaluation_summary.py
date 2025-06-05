import json
import pandas as pd
from pathlib import Path
from src.evaluation.evaluation import compute_mrr, compute_recall_at_k, compute_ndcg_at_k
from config import (
    EVAL_RESULTS_DIR,
    ensure_dir,
    is_colab,
    is_kaggle,
    is_local,
    ENVIRONMENT
)

def load_json(path):
    """Load JSON file with environment-specific error handling."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        if is_colab:
            print(f"File not found at {path}. Please ensure you have mounted Google Drive and the file is in the correct location.")
        elif is_kaggle:
            print(f"File not found at {path}. Please ensure you have added the dataset and the file is in the correct location.")
        else:
            print(f"File not found at {path}. Please check if the file exists.")
        raise
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON file {path}: {str(e)}")
        raise

def evaluate_all_models(ranking_path, ground_truth_path, output_csv_path, ks=[10, 100]):
    """Evaluate all models with environment-specific error handling."""
    print(f"Running evaluation in {ENVIRONMENT} environment...")
    ensure_dir(EVAL_RESULTS_DIR)

    try:
        results = load_json(ranking_path)
        ground_truth = load_json(ground_truth_path)

        rows = []
        for model_name, preds in results.items():
            row = {"Model": model_name}
            mrr = compute_mrr(preds, ground_truth, 10)
            row["MRR"] = round(mrr, 4)

            for k in ks:
                recall = compute_recall_at_k(preds, ground_truth, k)
                ndcg = compute_ndcg_at_k(preds, ground_truth, k)
                row[f"Recall@{k}"] = round(recall, 4)
                row[f"NDCG@{k}"] = round(ndcg, 4)

            rows.append(row)

        df = pd.DataFrame(rows)
        try:
            df.to_csv(output_csv_path, index=False)
            print(f"✅ Evaluation summary saved to: {output_csv_path}")
        except Exception as e:
            print(f"❌ Error saving evaluation results: {e}")
            if is_colab:
                print("Please ensure you have write permissions in your Google Drive.")
            elif is_kaggle:
                print("Please ensure you have write permissions in your Kaggle workspace.")
            raise
        return df
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        raise

if __name__ == "__main__":
    # Example usage:
    evaluate_all_models(
        "retrieval_results.json",
        "ground_truth.json",
        EVAL_RESULTS_DIR / "evaluation_summary.csv",
        ks=[10, 100]
    )