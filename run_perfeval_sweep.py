"""
run_perfeval_sweep.py — PerfEval Part 2 payoff figure.

Sweeps the DATABASE SIZE at fixed hardware and plots how energy, latency and
area grow, with the MVM cost (Part 1) and the top-k selection cost (Part 2)
drawn separately. This is the picture that makes the novelty visible: the
top-k cost is a real, size-dependent number that CAMASim reports as zero.

Reads the real characterised hardware from array_cost.json (produced by
characterize.py). Run characterize first if it is missing.

Usage:
    python run_perfeval_sweep.py                      # uses ./array_cost.json
    python run_perfeval_sweep.py path/to/array_cost.json
"""

import os
import sys

sys.path.insert(0, "/mnt/c/Users/Charlie Power/CAMASim-Hybrid/modules/CAMASIM")
# also work regardless of where it is launched from
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules", "CAMASIM"))

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")   # WSL: no display
import matplotlib.pyplot as plt

from CAMASim.performance.perfeval import evaluate_workload

# ── workload knobs (the hardware is fixed; we vary the data size) ─────────────
DIM        = 384        # embedding dimension (e.g. all-MiniLM-L6-v2)
N_QUERIES  = 200
K          = 10
N_DOCS     = [1_000, 3_000, 10_000, 30_000, 100_000,
              300_000, 1_000_000, 3_000_000, 10_000_000]

cost_path = sys.argv[1] if len(sys.argv) > 1 else "array_cost.json"
if not os.path.exists(cost_path):
    sys.exit(f"ERROR: {cost_path} not found. Run characterize.py first:\n"
             f"  PYTHONPATH=modules/CAMASIM python -m CAMASim.performance.characterize hardware_example.json")

with open(cost_path) as f:
    array_cost = json.load(f)
if "comparator_cost" not in array_cost:
    sys.exit(f"ERROR: {cost_path} has no comparator_cost block. Rebuild NeuroSim "
             f"(make) and re-run characterize.py so --comparator-cost is captured.")

rows = array_cost["hardware"]["array"]["rows"]
cols = array_cost["hardware"]["array"]["cols"]
dev  = array_cost["hardware"].get("device", "?")
print(f"Hardware: {rows}x{cols} {dev} | dim={DIM}, queries={N_QUERIES}, k={K}\n")

# ── run the sweep ─────────────────────────────────────────────────────────────
mvm  = {"energy": [], "latency": [], "area": []}
topk = {"energy": [], "latency": [], "area": []}
tot  = {"energy": [], "latency": [], "area": []}

hdr = f"{'n_docs':>11} {'E_tot(J)':>11} {'topk_E%':>8} {'L_tot(s)':>11} {'topk_L%':>8} {'area(m2)':>11}"
print(hdr)
for n in N_DOCS:
    r = evaluate_workload(n_docs=n, dim=DIM, n_queries=N_QUERIES, k=K, array_cost=array_cost)
    mvm["energy"].append(r.mvm_energy_j);   topk["energy"].append(r.topk_energy_j);   tot["energy"].append(r.total_energy_j)
    mvm["latency"].append(r.mvm_latency_s); topk["latency"].append(r.topk_latency_s); tot["latency"].append(r.total_latency_s)
    mvm["area"].append(r.mvm_area_m2);      topk["area"].append(r.topk_area_m2);      tot["area"].append(r.total_area_m2)
    print(f"{n:>11} {r.total_energy_j:>11.3e} {r.topk_energy_share*100:>7.1f}% "
          f"{r.total_latency_s:>11.3e} {r.topk_latency_share*100:>7.1f}% {r.total_area_m2:>11.3e}")

# ── plot: 3 panels (energy / latency / area) vs database size ─────────────────
x = np.array(N_DOCS, dtype=float)
panels = [("Energy (J)", "energy"), ("Latency (s)", "latency"), ("Area (m$^2$)", "area")]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, (ylabel, key) in zip(axes, panels):
    ax.loglog(x, tot[key],  "o-",  color="black", label="Total",  linewidth=2)
    ax.loglog(x, mvm[key],  "s--", color="#1f77b4", label="MVM (Part 1)")
    ax.loglog(x, topk[key], "^--", color="#d62728", label="Top-k (Part 2)")
    ax.set_xlabel("Number of documents")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
axes[0].set_title("Energy: top-k a fixed fraction (both scale ~n_docs)")
axes[1].set_title("Latency: top-k dominates (serial vs parallel MVM)")
axes[2].set_title("Area: arrays grow, comparators fixed")
fig.suptitle(f"PerfEval cost vs database size  ({rows}x{cols} {dev}, dim={DIM}, k={K})",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])

out = "perfeval_cost_vs_datasize.png"
fig.savefig(out, dpi=130)
print(f"\nSaved {out}")
