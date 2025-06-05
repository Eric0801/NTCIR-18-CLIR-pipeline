import numpy as np
import time
from functools import wraps
from sklearn.metrics import ndcg_score

# 計時器，裝飾器
def timed_block(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = fn(*args, **kwargs)
        end = time.time()
        print(f" {fn.__name__} took {end - start:.2f}s")
        return result
    return wrapper

# Convert pid to pure number
def extract_pid(pid):
    return pid.split('_')[0]  #  extract the character before _ only, e.g. "474_p0_b19" -> "474"

def normalize_ground_truth(raw_gt_list):
    """
    將 list[dict(qid, retrieve)] 的 ground truth 轉成 dict[qid] = [pid]
    """
    gt_dict = {}
    for item in raw_gt_list:
        qid = str(item["qid"])
        pid = str(item["retrieve"])
        if qid not in gt_dict:
            gt_dict[qid] = []
        gt_dict[qid].append(pid)
    return gt_dict

# calculate MRR
def compute_mrr(results, ground_truth, k= 10): #cutoff = 10
    reciprocal_ranks = []
    for qid, ranked_pids in results.items():
        gt = set(map(str, ground_truth.get(str(qid), [])))  # ground truth 轉成字串集合
        for rank, pid in enumerate(ranked_pids, start=1):
            pid_id = extract_pid(pid)
            if pid_id in gt:
                reciprocal_ranks.append(1 / rank)
                break
        else:
            reciprocal_ranks.append(0)
    return np.mean(reciprocal_ranks)

# calculate Recall@k
def compute_recall_at_k(results, ground_truth, k=100):
    recall_scores = []
    for qid, ranked_pids in results.items():
        gt = set(map(str, ground_truth.get(str(qid), [])))
        retrieved = set(extract_pid(pid) for pid in ranked_pids[:k])
        if not gt:
            recall_scores.append(0)
            continue
        recall_scores.append(len(gt & retrieved) / len(gt))
    return np.mean(recall_scores)

# calculate NDCG@k
def compute_ndcg_at_k(results, ground_truth, k=10):
    y_true = []
    y_score = []
    all_pids = list({extract_pid(pid) for ranked in results.values() for pid in ranked})
    pid_index = {pid: i for i, pid in enumerate(all_pids)}

    for qid, ranked_pids in results.items():
        gt = set(map(str, ground_truth.get(str(qid), [])))
        y_true_row = [1 if pid in gt else 0 for pid in all_pids]
        y_score_row = [0] * len(all_pids)

        for rank, pid in enumerate(ranked_pids[:k]):
            pid_id = extract_pid(pid)
            if pid_id in pid_index:
                y_score_row[pid_index[pid_id]] = k - rank  # 分數：越前面排名越高

        y_true.append(y_true_row)
        y_score.append(y_score_row)

    return ndcg_score(np.array(y_true), np.array(y_score))