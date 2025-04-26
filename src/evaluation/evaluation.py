
import numpy as np
import time
from functools import wraps
from sklearn.metrics import ndcg_score

def timed_block(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = fn(*args, **kwargs)
        end = time.time()
        print(f" {fn.__name__} took {end - start:.2f}s")
        return result
    return wrapper

def compute_mrr(results, ground_truth):
    reciprocal_ranks = []
    for qid, ranked_pids in results.items():
        gt = ground_truth.get(str(qid), [])
        for rank, pid in enumerate(ranked_pids, start=1):
            if pid in gt:
                reciprocal_ranks.append(1 / rank)
                break
        else:
            reciprocal_ranks.append(0)
    return np.mean(reciprocal_ranks)

def compute_recall_at_k(results, ground_truth, k=100):
    recall_scores = []
    for qid, ranked_pids in results.items():
        gt = set(ground_truth.get(str(qid), []))
        retrieved = set(ranked_pids[:k])
        if not gt:
            recall_scores.append(0)
            continue
        recall_scores.append(len(gt & retrieved) / len(gt))
    return np.mean(recall_scores)

def compute_ndcg_at_k(results, ground_truth, k=10):
    y_true = []
    y_score = []
    all_pids = list({pid for ranked in results.values() for pid in ranked})
    pid_index = {pid: i for i, pid in enumerate(all_pids)}

    for qid, ranked_pids in results.items():
        gt = set(ground_truth.get(str(qid), []))
        y_true_row = [1 if pid in gt else 0 for pid in all_pids]
        y_score_row = [0] * len(all_pids)
        for rank, pid in enumerate(ranked_pids[:k]):
            if pid in pid_index:
                y_score_row[pid_index[pid]] = k - rank  # Higher score for higher rank
        y_true.append(y_true_row)
        y_score.append(y_score_row)

    return ndcg_score(np.array(y_true), np.array(y_score))
