"""
run_retrieval_sweep.py
----------------------
Phase 1 headline experiment: Recall@k and nDCG@k vs conductance noise,
for several ADC bit-widths.  PerfEval = 0 throughout.

Run from the repo root:
    python run_retrieval_sweep.py
"""

import sys
sys.path.insert(0, "/home/charlie_power/CAMASim-Hybrid/modules/CAMASIM")

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from CAMASim.function.module.distance import crossbar_innerproduct_pairwise
from CAMASim.function.module.sensing  import get_array_topk_results
from CAMASim.function.module.merge    import topk_merge
from CAMASim.metrics.retrieval        import ideal_topk, recall_at_k, ndcg_at_k

# ─────────────────────────────────────────────────────────────────────────────
# Experiment parameters
# ─────────────────────────────────────────────────────────────────────────────
NUM_DOCS    = 512
NUM_QUERIES = 64
DIM         = 128
K           = 10

SIGMAS   = [0.0, 0.02, 0.05, 0.1, 0.2]
ADC_BITS = [None, 8, 6, 4]
LABELS   = {None: "Ideal ADC", 8: "8-bit ADC", 6: "6-bit ADC", 4: "4-bit ADC"}
COLORS   = {None: "#2196F3", 8: "#4CAF50", 6: "#FF9800", 4: "#F44336"}
MARKERS  = {None: "o", 8: "s", 6: "^", 4: "D"}

np.random.seed(42)

docs    = np.random.randn(NUM_DOCS,    DIM).astype(np.float32)
docs   /= np.linalg.norm(docs,    axis=1, keepdims=True)
queries = np.random.randn(NUM_QUERIES, DIM).astype(np.float32)
queries /= np.linalg.norm(queries, axis=1, keepdims=True)

gt = ideal_topk(docs, queries, k=K)
print(f"Experiment: {NUM_DOCS} docs, {NUM_QUERIES} queries, dim={DIM}, K={K}")

# ── Sanity check ──────────────────────────────────────────────────────────────
hw_ideal = {"sigma_conductance": 0.0, "adc_bits": None, "sigma_read": 0.0}
scores_ideal = crossbar_innerproduct_pairwise(docs, queries, hw_ideal)
pred_idx, _ = get_array_topk_results(scores_ideal, k=K)
r_sanity = recall_at_k(pred_idx, gt)
assert r_sanity == 1.0, f"Sanity check FAILED: Recall@{K} = {r_sanity:.3f}"
print(f"Sanity check PASSED: Recall@{K} = {r_sanity:.3f} with no noise.\n")

# ── Sweep ─────────────────────────────────────────────────────────────────────
recall_results = {}
ndcg_results   = {}

for adc in ADC_BITS:
    label   = LABELS[adc]
    recalls = []
    ndcgs   = []

    for sigma in SIGMAS:
        hw = {"sigma_conductance": sigma, "adc_bits": adc, "sigma_read": 0.0}
        scores = crossbar_innerproduct_pairwise(docs, queries, hw)
        pred_idx, _ = get_array_topk_results(scores, k=K)
        r = recall_at_k(pred_idx, gt)
        n = ndcg_at_k(pred_idx, gt)
        recalls.append(r)
        ndcgs.append(n)
        print(f"  {label:<18}  sigma={sigma:.2f}  Recall@{K}={r:.3f}  nDCG@{K}={n:.3f}")

    recall_results[adc] = recalls
    ndcg_results[adc]   = ndcgs
    print()

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    f"Crossbar Retrieval: Hardware Noise vs Retrieval Quality\n"
    f"({NUM_DOCS} docs, dim={DIM}, K={K})",
    fontsize=13, fontweight='bold'
)

for adc in ADC_BITS:
    label  = LABELS[adc]
    color  = COLORS[adc]
    marker = MARKERS[adc]
    ls     = "--" if adc is None else "-"

    ax1.plot(SIGMAS, recall_results[adc],
             marker=marker, color=color, linestyle=ls,
             linewidth=2, markersize=7, label=label)

    ax2.plot(SIGMAS, ndcg_results[adc],
             marker=marker, color=color, linestyle=ls,
             linewidth=2, markersize=7, label=label)

for ax, ylabel, title in [
    (ax1, f"Recall@{K}",  f"Recall@{K} vs Conductance Noise"),
    (ax2, f"nDCG@{K}",    f"nDCG@{K} vs Conductance Noise"),
]:
    ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=1, alpha=0.6, label='Perfect score')
    ax.set_xlabel("Conductance noise σ (relative std)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(SIGMAS)
    ax.set_ylim(0.5, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("recall_vs_noise.png", dpi=200, bbox_inches='tight')
print("Figure saved to recall_vs_noise.png")