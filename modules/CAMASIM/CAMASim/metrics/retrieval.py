"""
CAMASim/metrics/retrieval.py
----------------------------
Retrieval accuracy metrics for the crossbar retrieval simulator.

Usage
-----
    from CAMASim.metrics.retrieval import ideal_topk, recall_at_k

    gt   = ideal_topk(docs, queries, k=10)
    pred = <results from CAMSearch.search(...)>
    r    = recall_at_k(pred, gt)
    print(f"Recall@10 = {r:.3f}")
"""

import numpy as np


def ideal_topk(docs, queries, k):
    """
    Compute the ground-truth top-k doc indices for each query using exact
    (noiseless) inner products.

    Args
    ----
    docs    : (num_docs,    dim) float
    queries : (num_queries, dim) float
    k       : int  — number of results per query

    Returns
    -------
    gt : (num_queries, k) int array  — indices of top-k docs, best first
    """
    scores = queries @ docs.T                              # (num_queries, num_docs)
    return np.argsort(scores, axis=1)[:, ::-1][:, :k]


def recall_at_k(pred_topk, gt_topk):
    """
    Mean Recall@k across all queries.

    Recall@k for one query = |predicted ∩ ground-truth| / k

    Args
    ----
    pred_topk : list of length num_queries, each element a list/array of
                predicted doc indices (length up to k)
    gt_topk   : (num_queries, k) int array from ideal_topk(), or equivalent
                list-of-lists

    Returns
    -------
    float  — mean Recall@k in [0, 1]
    """
    recalls = []
    for pred, gt in zip(pred_topk, gt_topk):
        pred_set = set(int(x) for x in pred)
        gt_set   = set(int(x) for x in gt)
        hit      = len(pred_set & gt_set)
        recalls.append(hit / len(gt_set) if gt_set else 0.0)
    return float(np.mean(recalls))


def ndcg_at_k(pred_topk, gt_topk):
    """
    Mean nDCG@k across all queries.

    nDCG rewards results that appear earlier in the ranked list more than
    results that appear later.  Relevance is binary (1 if in gt, 0 if not).

    Args
    ----
    pred_topk : list of length num_queries, each element a list/array of
                predicted doc indices
    gt_topk   : (num_queries, k) int array from ideal_topk()

    Returns
    -------
    float  — mean nDCG@k in [0, 1]
    """
    def dcg(ranked, relevant_set):
        score = 0.0
        for rank, idx in enumerate(ranked, start=1):
            if int(idx) in relevant_set:
                score += 1.0 / np.log2(rank + 1)
        return score

    ndcgs = []
    for pred, gt in zip(pred_topk, gt_topk):
        gt_set   = set(int(x) for x in gt)
        ideal    = dcg(list(gt_set)[:len(pred)], gt_set)   # ideal DCG
        actual   = dcg(pred, gt_set)
        ndcgs.append(actual / ideal if ideal > 0 else 0.0)
    return float(np.mean(ndcgs))