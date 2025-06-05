import json
from pathlib import Path
from config import (
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

def extract_translation_impact(queries_path, predictions_path, ground_truth_path, top_k=5):
    """Analyze translation impact with environment-specific error handling."""
    print(f"Running translation analysis in {ENVIRONMENT} environment...")

    try:
        queries = load_json(queries_path)
        predictions = load_json(predictions_path)

        # Convert ground_truths to dict {qid: [pid]}
        gt_raw = load_json(ground_truth_path)
        gt_dict = {
            str(item["qid"]): [str(item["retrieve"])]
            for item in gt_raw["ground_truths"]
        }

        examples = {
            "correct_translation_hit": [],
            "correct_translation_miss": [],
            "wrong_translation_hit": [],
            "wrong_translation_miss": []
        }

        for q in queries:
            qid = str(q["qid"])
            zh = q.get("query_zh_nmt", "").strip()
            en = q.get("query_en", "").strip()
            gt = set(gt_dict.get(qid, []))

            for model_name, model_preds in predictions.items():
                pred_topk = model_preds.get(qid, [])[:top_k]
                hit = any(pid.split('_')[0] in gt for pid in pred_topk)

                is_translation_good = (len(zh) > 0 and zh != en)

                if is_translation_good and hit:
                    examples["correct_translation_hit"].append((qid, en, zh, pred_topk, list(gt)))
                elif is_translation_good and not hit:
                    examples["correct_translation_miss"].append((qid, en, zh, pred_topk, list(gt)))
                elif not is_translation_good and hit:
                    examples["wrong_translation_hit"].append((qid, en, zh, pred_topk, list(gt)))
                else:
                    examples["wrong_translation_miss"].append((qid, en, zh, pred_topk, list(gt)))

        return examples
    except Exception as e:
        print(f"❌ Error during translation analysis: {e}")
        raise

if __name__ == "__main__":
    # Example usage:
    examples = extract_translation_impact(
        "data/translated_query.json",
        "retrieval_results.json",
        "ground_truth.json",
        top_k=5
    )
    print(f"Analysis complete. Found examples:")
    for category, items in examples.items():
        print(f"- {category}: {len(items)} examples")
