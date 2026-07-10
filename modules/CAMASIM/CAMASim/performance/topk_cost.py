"""
topk_cost.py — Part 2 of PerfEval: the top-k selection cost.

This is the piece that sets our simulator apart from CAMASim. In a crossbar
CAM, the comparison happens inside the array and finishes there, so there is
no separate top-k cost. In retrieval (MIPS) we instead get a score per
document and then have to *select* the best k. That selection is real work on
real hardware, and — crucially — it grows with the size of the database.

Where it lives
--------------
The selection happens at the MERGE stage (merge.py::topk_merge), where
per-array candidates are pooled and the best k are picked. This module costs
that step; the wiring that calls it lives in Part 2c.

The cost model
--------------
A comparator-based top-k over N candidates costs on the order of

    comparisons  ~  k * N          ("linear" model, the draft's default)

Intuition: to pull out the best, then the next best, ... k times, each pass
scans the remaining candidates. Small N -> negligible; large N -> this shows
up, which is exactly the size-dependent cost CAMASim never reports.

An alternative "tree" model (comparator tree / heap selection) is also
provided for comparison:

    comparisons  ~  N + k * ceil(log2 N)

The model is swappable so switching is a one-line change if the advisor
prefers a different accounting.

Getting the per-comparison cost
-------------------------------
Same "characterize once" idea as Part 1: NeuroSim's Comparator block gives the
area / latency / energy of ONE comparison; everything here scales that by the
comparison count. (Sourcing those three numbers from NeuroSim is Part 2b — a
small additive --comparator-cost mode in main.cpp. Until then this module
accepts them as plain arguments so it is fully testable on its own.)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from typing import Optional


# --- comparison-count models --------------------------------------------------

def comparisons_linear(n_candidates: int, k: int) -> int:
    """Draft default: ~k*N comparisons (k selection passes over N candidates)."""
    kk = min(k, n_candidates)
    return kk * n_candidates


def comparisons_tree(n_candidates: int, k: int) -> int:
    """Comparator-tree / heap selection: ~N + k*ceil(log2 N) comparisons."""
    if n_candidates <= 1:
        return 0
    kk = min(k, n_candidates)
    return n_candidates + kk * math.ceil(math.log2(n_candidates))


_MODELS = {
    "linear": comparisons_linear,
    "tree": comparisons_tree,
}


def topk_comparisons(n_candidates: int, k: int, model: str = "linear") -> int:
    """Number of comparisons to select top-k from N candidates, per the model."""
    if model not in _MODELS:
        raise ValueError(f"unknown model '{model}' (choose from {list(_MODELS)})")
    if n_candidates <= 0 or k <= 0:
        raise ValueError("n_candidates and k must be > 0")
    return _MODELS[model](n_candidates, k)


@dataclass
class TopkCostResult:
    """Breakdown of one top-k selection's hardware cost."""

    n_candidates: int
    k: int
    model: str
    num_comparators: int        # comparator units placed (parallelism knob)
    comparisons: int            # total comparisons the selection performs
    serial_steps: int           # comparison "waves" given the comparators placed

    area_m2: float
    latency_s: float
    energy_j: float

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent)


def topk_cost(
    n_candidates: int,
    k: int,
    area_per_comparator_m2: float,
    latency_per_comparison_s: float,
    energy_per_comparison_j: float,
    num_comparators: int = 1,
    model: str = "linear",
) -> TopkCostResult:
    """Cost of selecting top-k from N candidates.

    Parameters
    ----------
    n_candidates :
        N — how many scored candidates the selection must compare. For a whole
        database search this scales with the number of documents, which is why
        the cost grows with data size.
    k :
        How many results to return.
    area_per_comparator_m2, latency_per_comparison_s, energy_per_comparison_j :
        Per-comparison numbers from NeuroSim's Comparator block (Part 2b).
    num_comparators :
        How many comparator units are placed. 1 (default) = fully serial
        selection. More comparators run comparisons in parallel: latency drops,
        area rises (energy is unchanged) — the same area<->latency trade-off as
        the arrays in Part 1.
    model :
        Comparison-count model, "linear" (~k*N, default) or "tree".

    Returns
    -------
    TopkCostResult
    """
    if num_comparators <= 0:
        raise ValueError("num_comparators must be > 0")

    comparisons = topk_comparisons(n_candidates, k, model=model)

    # Energy: each comparison spends its own energy.
    energy_j = energy_per_comparison_j * comparisons

    # Area: proportional to comparator units placed.
    area_m2 = area_per_comparator_m2 * num_comparators

    # Latency: comparisons run in parallel across the placed comparators;
    # only the "waves" that wait their turn add up.
    serial_steps = math.ceil(comparisons / num_comparators)
    latency_s = serial_steps * latency_per_comparison_s

    return TopkCostResult(
        n_candidates=n_candidates,
        k=k,
        model=model,
        num_comparators=num_comparators,
        comparisons=comparisons,
        serial_steps=serial_steps,
        area_m2=area_m2,
        latency_s=latency_s,
        energy_j=energy_j,
    )


# --- helper: how many candidates does a retrieval merge actually compare? -----

def retrieval_n_candidates(n_docs: int, k: int, num_doc_tiles: int,
                           local_preselect: bool = True) -> int:
    """Figure out N for a crossbar retrieval top-k.

    Two regimes:

    * local_preselect=True (default): each of the ``num_doc_tiles`` arrays
      first returns its own local top-k, and the merge selects the global
      top-k from that pool. Candidates compared at the merge = num_doc_tiles*k.
      (The per-array local selection over all docs is the larger cost; callers
      that want the full-database comparison count should pass local_preselect
      =False.)

    * local_preselect=False: the selection compares every document's score,
      so N = n_docs. This is the size-dependent worst case the draft
      highlights ("a lot more values to compare" as data grows).
    """
    if local_preselect:
        return max(num_doc_tiles * k, k)
    return n_docs


def load_comparator_cost(path: str) -> dict:
    """Load a comparator-cost block (from array_cost.json or its own file)."""
    with open(path, "r") as f:
        data = json.load(f)
    # accept either a bare comparator dict or an array_cost.json with a nested block
    return data.get("comparator_cost", data)


if __name__ == "__main__":
    # Illustrative: top-k cost growing with database size (placeholder unit costs).
    for n in (1_000, 10_000, 100_000):
        r = topk_cost(
            n_candidates=n, k=10,
            area_per_comparator_m2=1e-11,
            latency_per_comparison_s=1e-10,
            energy_per_comparison_j=1e-15,
        )
        print(f"N={n:>7}  comparisons={r.comparisons:>9}  "
              f"energy={r.energy_j:.3e} J  latency={r.latency_s:.3e} s")
