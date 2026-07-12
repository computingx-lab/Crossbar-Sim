from collections import Counter

import numpy as np


def getKeyEqual(dct, value):
    return [key for key in dct if (dct[key] == value)]

def getKeyLarge(dct, value):
    return [key for key in dct if (dct[key] >= value)]

def sortDic(dct):
    return dict(sorted(dct.items(), key=lambda item: item[1], reverse=True))

def exact_merge(matchInd, rowCams, colCams):
    """Find indices that have matches in all CAM arrays (exact match)."""
    count = Counter(matchInd)
    results = getKeyEqual(count, colCams)
    return results

def knn_merge(matchInd, rowCams, colCams, topk):
    """Sort matches by frequency and return the top-k most frequent indices."""
    count = Counter(matchInd)
    if rowCams == 1:
        sortedCount = sortDic(count)
        results = list(sortedCount.keys())[:topk]
    else:
        raise NotImplementedError
    return results

def threshold_merge(matchInd, rowCams, colCams):
    """Find indices with matches in a specified number of CAM arrays."""
    count = Counter(matchInd)
    if colCams == 1:
        results = getKeyEqual(count, colCams)
        return results
    else:
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# Top-k score merge  (Phase 1 — crossbar retrieval)
# ─────────────────────────────────────────────────────────────────────────────

def topk_merge(cand_indices, cand_scores, k, sigma_compare=0.0, rng=None):
    """
    Pool local top-k candidates from all row-arrays and re-rank by score to
    produce the global top-k.

    Why this is needed: a real embedding library will not fit in a single
    crossbar.  Docs are split across arrays by row.  Each array returns its
    own local top-k.  We pool all local candidates and pick the globally
    highest-scored k.

    Phase 1 assumption: dim fits in one array's columns (no column-splitting).
    Column-splitting (partial sums summed before sensing) is a TODO.

    Args
    ----
    cand_indices : list or array of global doc indices pooled from all
                   row-arrays for ONE query
    cand_scores  : list or array of the matching scores (same order/length)
    k            : int  — how many to return

    Returns
    -------
    global_topk : 1-D int array of the k best doc indices, best-first
    """
    # TODO: column-splitting not yet modelled — assumes dim fits in one array's columns
    # TODO: partial sums from column-split arrays would need to be summed before sensing


    cand_indices = np.asarray(cand_indices)
    cand_scores  = np.asarray(cand_scores,  dtype=float)

    kk    = min(k, len(cand_indices))
    # Compare-side noise at the MERGE: the global re-rank is ordered by a noisy
    # key (score + fresh comparator noise), so near-ties among the pooled
    # candidates can flip here too. sigma_compare = 0 -> exact (unchanged).
    _rng = rng if rng is not None else np.random
    if sigma_compare > 0:
        key = cand_scores + _rng.normal(0.0, sigma_compare, size=cand_scores.shape)
    else:
        key = cand_scores
    order = np.argsort(key)[::-1][:kk]   # largest (noisy) score first
    return cand_indices[order]