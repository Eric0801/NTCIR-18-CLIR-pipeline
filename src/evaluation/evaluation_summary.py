import json
import pandas as pd
from src.evaluation.evaluation import compute_mrr, compute_recall_at_k, compute_ndcg_at_k

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def evaluate_all_models(ranking_path, ground_truth_path, output_csv_path, ks=[10, 100]):
    results = load_json(ranking_path)
    ground_truth = load_json(ground_truth_path)

    rows = []
    for model_name, preds in results.items():
        row = {"Model": model_name}
        mrr = compute_mrr(preds, ground_truth)
        row["MRR"] = round(mrr, 4)

        for k in ks:
            recall = compute_recall_at_k(preds, ground_truth, k)
            ndcg = compute_ndcg_at_k(preds, ground_truth, k)
            row[f"Recall@{k}"] = round(recall, 4)
            row[f"NDCG@{k}"] = round(ndcg, 4)

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv_path, index=False)
    print(f"✅ Evaluation summary saved to: {output_csv_path}")
    return df

# Example usage:
# evaluate_all_models("retrieval_results.json", "ground_truth.json", "evaluation_summary.csv", ks=[10, 100])