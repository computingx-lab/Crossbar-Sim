'''
The file define hamming, l1, l2, innerproduct distance.
Refer to repository: https://github.com/JKnighten/k-nearest-neighbors
'''

from math import sqrt

import numpy as np


######################
# Euclidean Distance #
######################
def euclidean(vector_a, vector_b):
    dims = vector_a.shape[0]
    distance = 0
    for i in range(dims):
        temp = vector_a[i] - vector_b[i]
        distance += (temp*temp)
    return sqrt(distance)


def euclidean_pairwise(vectors_a, vectors_b):
    num_vectors_a = vectors_a.shape[0]
    num_vectors_b = vectors_b.shape[0]
    num_dims = vectors_a.shape[1]
    distances = np.zeros([num_vectors_b, num_vectors_a])
    for i in range(num_vectors_b):
        for j in range(num_vectors_a):
            distance = 0.0
            for k in range(num_dims):
                temp = vectors_a[j, k] - vectors_b[i, k]
                distance += (temp*temp)
            distances[i, j] = sqrt(distance)
    return distances

######################
# Manhattan Distance #
######################
def manhattan(vector_a, vector_b):
    dims = vector_a.shape[0]
    distance = 0
    for i in range(dims):
        temp = abs(vector_a[i] - vector_b[i])
        distance += temp
    return distance

def manhattan_pairwise(vectors_a, vectors_b):
    num_vectors_a = vectors_a.shape[0]
    num_vectors_b = vectors_b.shape[0]
    num_dims = vectors_a.shape[1]
    distances = np.zeros([num_vectors_b, num_vectors_a])
    for i in range(num_vectors_b):
        for j in range(num_vectors_a):
            for k in range(num_dims):
                 distances[i, j] += abs(vectors_a[j, k] - vectors_b[i, k])
    return distances

####################
# Hamming Distance #
####################
def hamming(vector_a, vector_b):
    dims = vector_a.shape[0]
    distance = 0
    for i in range(dims):
        if vector_a[i] != vector_b[i]:
            distance += 1.0
    return distance


def hamming_pairwise(vectors_a, vectors_b):
    num_vectors_a = vectors_a.shape[0]
    num_vectors_b = vectors_b.shape[0]
    num_dims = vectors_a.shape[1]
    distances = np.zeros([num_vectors_b, num_vectors_a])
    for i in range(num_vectors_b):
        for j in range(num_vectors_a):
            for k in range(num_dims):
                if vectors_a[j, k] != vectors_b[i, k]:
                    distances[i, j] += 1.0
    return distances

##########################
# Inner Product Distance #
##########################
def innerproduct(vector_a, vector_b):
    dims = vector_a.shape[0]
    distance = 0
    for i in range(dims):
        distance += (vector_a[i]*vector_b[i])
    return distance

def innerproduct_pairwise(vectors_a, vectors_b):
    num_vectors_a = vectors_a.shape[0]
    num_vectors_b = vectors_b.shape[0]
    num_dims = vectors_a.shape[1]
    distances = np.zeros([num_vectors_b, num_vectors_a])
    for i in range(num_vectors_b):
        for j in range(num_vectors_a):
            for k in range(num_dims):
                 distances[i, j] += vectors_a[j, k] * vectors_b[i, k]
    return distances

######################
# Range query #
######################
def rangequery(vector_a, vector_b):
    dims = vector_a.shape[0]
    distance = 0
    for i in range(dims):
        temp = 0 if (vector_a[i] >= vector_b[i, 0]) & (vector_a[i] <= vector_b[i, 1]) else 1
        distance += temp
    return (distance)


def rangequery_pairwise(vectors_a, vectors_b):
    num_vectors_a = vectors_a.shape[0]
    num_vectors_b = vectors_b.shape[0]
    num_dims = vectors_a.shape[1]
    distances = np.zeros([num_vectors_b, num_vectors_a])
    for i in range(num_vectors_b):
        for j in range(num_vectors_a):
            distance = 0.0
            for k in range(num_dims):
                temp = 0 if (vectors_b[i, k] >= vectors_a[j, k, 0]) & (vectors_b[i, k] <= vectors_a[j, k, 1]) else 1
                distance += temp
            distances[i, j] = distance
    return distances


# ─────────────────────────────────────────────────────────────────────────────
# Crossbar Inner Product  (Phase 1 — noisy retrieval)
# ─────────────────────────────────────────────────────────────────────────────

def quantize_adc(x, bits):
    """Uniform mid-tread ADC quantization over the observed range of x."""
    lo, hi = x.min(), x.max()
    if hi == lo:
        return x
    levels = 2 ** bits - 1
    step   = (hi - lo) / levels
    return lo + np.round((x - lo) / step) * step

# TODO: IR drop along bitlines/wordlines not modelled (Phase 1 assumption)
# TODO: column-splitting for embeddings wider than one array not modelled (Phase 1 assumption)

def crossbar_innerproduct_pairwise(docs, queries, hw):
    """
    Computes inner products the way an analog crossbar does, with hardware noise.

    Physics modelled
    ----------------
    1. Docs are stored as conductances.  Device variation adds Gaussian noise
       once at programming time (multiplicative, so it scales with the value).
    2. Queries are applied as input voltages.  The column current is the
       dot product  scores = queries @ docs.T  (vectorised, no triple loop).
    3. An ADC quantises the column output to a fixed number of bits.
    4. Optional per-query read noise is added after the ADC.

    Args
    ----
    docs    : (num_docs,    dim)  float  — stored embeddings / conductances
    queries : (num_queries, dim)  float  — query vectors / input voltages
    hw      : dict with keys
                sigma_conductance  float  relative std of device variation (0 = off)
                adc_bits           int    ADC resolution; None = ideal (off)
                sigma_read         float  std of additive read noise (0 = off)

    Returns
    -------
    scores : (num_queries, num_docs)  — NOISY inner products, higher = more similar
    """
    # 1. Program docs as conductances with device variation (applied once)
    sigma_g = hw.get("sigma_conductance", 0.0)
    if sigma_g > 0:
        noise     = np.random.normal(0.0, sigma_g, size=docs.shape)
        docs_noisy = docs * (1.0 + noise)   # multiplicative: scales with value
    else:
        docs_noisy = docs

    # 2. Analog matrix-vector multiply — one matmul, no loops
    scores = queries @ docs_noisy.T          # (num_queries, num_docs)

    # 3. ADC quantisation of the readout
    adc_bits = hw.get("adc_bits", None)
    if adc_bits is not None:
        scores = quantize_adc(scores, adc_bits)

    # 4. Optional per-query read noise
    sigma_r = hw.get("sigma_read", 0.0)
    if sigma_r > 0:
        scores = scores + np.random.normal(0.0, sigma_r, size=scores.shape)

    return scores