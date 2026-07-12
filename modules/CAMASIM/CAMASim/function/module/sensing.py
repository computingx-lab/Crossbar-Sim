import random

import numpy as np


def get_array_best_results(distance_matrix):
    """
    For each query, find the best match index and distance from the distance matrix.
    """
    k = 1
    indices = np.argpartition(distance_matrix, k - 1)[:, :k]
    distances = np.array([distance_matrix[i, x] for i, x in enumerate(indices)])
    return indices, distances


def get_array_best_results_sensing(distance_matrix, sensing_limit):
    """
    For each query, find the best match index and distance from the distance matrix.
    """
    k = 1
    indices = np.argpartition(distance_matrix, k - 1)[:, :k]
    for rowIndex in range(distance_matrix.shape[0]):
        allRowIndices = np.where(
            (
                distance_matrix[rowIndex]
                <= distance_matrix[rowIndex, indices[rowIndex]] + sensing_limit
            )
            & (
                distance_matrix[rowIndex]
                >= distance_matrix[rowIndex, indices[rowIndex]] - sensing_limit
            )
        )[0]
        indices[rowIndex] = random.choice(allRowIndices)
    distances = np.array([distance_matrix[i, x] for i, x in enumerate(indices)])
    return indices, distances


def get_array_exact_results(distance_matrix):
    """
    For each query, find indices with a distance of 0 (exact match) and their distances.
    """
    indices = []
    distances = []
    for i in range(distance_matrix.shape[0]):
        indice = np.where(distance_matrix[i] == 0)[0]
        indices.append(indice)
        distances.append(
            np.array([distance_matrix[i, x] for p, x in enumerate(indice)])
        )
    return indices, distances


def get_array_threshold_results(distance_matrix, threshold):
    """
    For each query, find indices with distances <= threshold and their distances.
    """
    indices = []
    distances = []
    for i in range(distance_matrix.shape[0]):
        indice = np.where(distance_matrix[i] <= threshold)[0]
        indices.append(indice)
        distances.append(
            np.array([distance_matrix[i, x] for p, x in enumerate(indice)])
        )
    return indices, distances


# ─────────────────────────────────────────────────────────────────────────────
# Top-k sensing  (Phase 1 — crossbar retrieval)
# ─────────────────────────────────────────────────────────────────────────────

def get_array_topk_results(score_matrix, k, sigma_compare=0.0, rng=None):
    """
    For each query, return the indices of the k docs with the LARGEST scores.

    Unlike the existing sensing functions (which look for the *smallest*
    distance), retrieval uses inner products where higher = more similar.

    Args
    ----
    score_matrix : (num_queries, num_docs) float  — noisy inner-product scores
    k            : int  — number of top results to return per query

    Returns
    -------
    indices   : list of length num_queries, each entry a 1-D int array of
                the top-k doc indices, sorted best-first (largest score first)
    distances : list of length num_queries, each entry the matching scores
                (kept so merge can re-rank across arrays)
    """
    indices   = []
    distances = []
    kk = min(k, score_matrix.shape[1])
    _rng = rng if rng is not None else np.random

    for i in range(score_matrix.shape[0]):
        row  = score_matrix[i]
        # Comparator (compare-side) noise: the selection is ordered by a noisy
        # key (row + fresh comparator noise), so near-ties can flip -- but the
        # scores we REPORT are the true sensed scores of whatever got selected.
        # sigma_compare = 0 -> exact selection (unchanged behaviour).
        if sigma_compare > 0:
            key = row + _rng.normal(0.0, sigma_compare, size=row.shape)
        else:
            key = row
        # argpartition gives the kk largest indices (unordered) cheaply
        part  = np.argpartition(key, -kk)[-kk:]
        # sort those kk candidates so the best (largest) comes first
        order = part[np.argsort(key[part])[::-1]]
        indices.append(order)
        distances.append(row[order])   # TRUE scores of the selected docs

    return indices, distances