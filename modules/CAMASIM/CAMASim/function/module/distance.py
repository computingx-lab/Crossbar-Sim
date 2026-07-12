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

def make_level_profile(bits, g_min=-1.0, g_max=1.0, sigma_low=0.01, sigma_high=0.03,
                       level_conductances=None, level_sigmas=None):
    """Build (conductances, sigmas) for a 2**bits-level cell.

    Level-based model: 2**bits discrete conductance levels, each with its OWN
    target conductance and its OWN write-noise sigma. Everything is a parameter
    (never hard-coded), so measured device values can replace these later.

    Defaults (used only when explicit lists are not supplied):
      * conductances : spread evenly across [g_min, g_max]
      * sigmas       : grow linearly from sigma_low (level 0) to sigma_high
                       (top level), since higher levels usually drift more
    """
    n = 2 ** int(bits)
    g = np.asarray(level_conductances, dtype=float) if level_conductances is not None \
        else np.linspace(g_min, g_max, n)
    s = np.asarray(level_sigmas, dtype=float) if level_sigmas is not None \
        else np.linspace(sigma_low, sigma_high, n)
    if len(g) != n or len(s) != n:
        raise ValueError(f"expected {n} levels (bits={bits}); got "
                         f"{len(g)} conductances / {len(s)} sigmas")
    return g, s


def quantize_to_levels(x, level_conductances):
    """Snap each value in x to the nearest level; return (indices, snapped).

    Works for arbitrary (even non-uniform) level spacing. This is the
    quantization half of the level-based write model.
    """
    g = np.asarray(level_conductances, dtype=float)
    idx = np.abs(np.asarray(x, dtype=float)[..., None] - g).argmin(axis=-1)
    return idx, g[idx]


def apply_level_based_noise(docs, hw):
    """Level-based WRITE noise (Point 1): quantize each stored value to the
    nearest level, then add that level's OWN sigma. Captures both the
    quantization error and the level-specific device variation in one move.

    hw keys (all optional, all parameters):
      level_bits            int   bits per cell -> 2**bits levels (default 2)
      level_g_min/g_max     float conductance range to spread levels over
                                  (default: the data's own min/max)
      level_sigma_low/high  float end sigmas for the default growing ramp
      level_conductances    list  explicit per-level conductances (overrides)
      level_sigmas          list  explicit per-level sigmas (overrides)
    """
    docs = np.asarray(docs, dtype=float)
    bits = int(hw.get("level_bits", 2))
    g, sig = make_level_profile(
        bits,
        g_min=hw.get("level_g_min", float(docs.min())),
        g_max=hw.get("level_g_max", float(docs.max())),
        sigma_low=hw.get("level_sigma_low", 0.01),
        sigma_high=hw.get("level_sigma_high", 0.03),
        level_conductances=hw.get("level_conductances"),
        level_sigmas=hw.get("level_sigmas"),
    )
    idx, snapped = quantize_to_levels(docs, g)
    per_value_sigma = sig[idx]                       # each cell gets its level's sigma
    noise = np.random.normal(0.0, 1.0, size=docs.shape) * per_value_sigma
    return snapped + noise


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
    # 1. Program docs as conductances with WRITE-side device variation.
    #    Point 1 in the noise model: static -- written once and frozen, so every
    #    query in this call sees the same fixed error. Two switchable models via
    #    hw["noise_type"]: "gaussian" (one global sigma) or "level_based"
    #    (2**bits discrete levels, each with its own conductance and sigma).
    noise_type = hw.get("noise_type", "gaussian")
    if noise_type == "level_based":
        docs_noisy = apply_level_based_noise(docs, hw)
    elif noise_type == "gaussian":
        sigma_g = hw.get("sigma_conductance", 0.0)
        if sigma_g > 0:
            noise     = np.random.normal(0.0, sigma_g, size=docs.shape)
            docs_noisy = docs * (1.0 + noise)   # multiplicative: scales with value
        else:
            docs_noisy = docs
    else:
        raise ValueError(f"unknown noise_type '{noise_type}' "
                         "(choose 'gaussian' or 'level_based')")

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