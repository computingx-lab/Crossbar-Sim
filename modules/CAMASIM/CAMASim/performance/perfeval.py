"""
perfeval.py — PerfEval integrator: MVM cost (Part 1) + top-k cost (Part 2).

Given a workload (how many documents, what embedding dimension, how many
queries, and k) plus a characterised piece of hardware (array_cost.json from
characterize.py), this returns the full cost of running the search, split into
its two parts:

    * MVM cost   — from mapper.py: tiling the data onto arrays and paying for
                   every matrix-vector multiply.
    * top-k cost — from topk_cost.py: the comparator-based selection that picks
                   the best k. This is the size-dependent cost CAMASim does not
                   report, and the reason our tool is different.

The headline top-k count is  k * n_docs : selecting the top-k out of the whole
database compares (on the order of) every document's score. A smaller merge
pass — re-ranking the pooled num_doc_tiles * k local winners — is added on top
when include_merge_pass=True.

Everything is deterministic in the workload shape; noise/accuracy (Phase 1)
lives on a separate path and does not affect these numbers.

Two behaviours worth knowing (both fall out of the model, both are real):
    * ENERGY: MVM energy (~arrays~n_docs) and top-k energy (~k*n_docs) both
      scale linearly with data size, so top-k energy GROWS with the database
      but its SHARE of total energy stays roughly constant — a fixed, non-
      trivial tax CAMASim reports as zero.
    * LATENCY: MVM arrays run in parallel (latency flat in n_docs) while the
      top-k selection runs serially on the comparator(s) (latency ~ k*n_docs),
      so top-k latency grows with data size and dominates at scale. Placing
      more comparators (num_comparators) trades area to bring it back down.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional

try:
    from CAMASim.performance.mapper import map_and_cost, MappingResult
    from CAMASim.performance.topk_cost import topk_cost, TopkCostResult
except ModuleNotFoundError:  # allow running this file directly
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..")))
    from CAMASim.performance.mapper import map_and_cost, MappingResult
    from CAMASim.performance.topk_cost import topk_cost, TopkCostResult


@dataclass
class PerfEvalReport:
    """Full PPA of one workload on one hardware config, MVM + top-k."""

    n_docs: int
    dim: int
    n_queries: int
    k: int

    mvm_area_m2: float
    mvm_latency_s: float
    mvm_energy_j: float

    topk_area_m2: float
    topk_latency_s: float
    topk_energy_j: float
    topk_comparisons_total: int
    topk_model: str

    total_area_m2: float
    total_latency_s: float
    total_energy_j: float

    topk_energy_share: float
    topk_latency_share: float
    topk_area_share: float

    mapping: dict = field(default_factory=dict)
    topk_local: dict = field(default_factory=dict)
    topk_merge: dict = field(default_factory=dict)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent)


def _require(d: dict, key: str, ctx: str):
    if key not in d:
        raise KeyError(f"array_cost is missing '{key}' ({ctx})")
    return d[key]


def evaluate_workload(
    n_docs: int,
    dim: int,
    n_queries: int,
    k: int,
    array_cost: dict,
    array_budget: Optional[int] = None,
    num_comparators: int = 1,
    model: str = "linear",
    include_merge_pass: bool = True,
) -> PerfEvalReport:
    """Combine Part 1 (MVM) and Part 2 (top-k) into one cost report."""
    # --- MVM cost (Part 1) ---
    hw = _require(array_cost, "hardware", "run characterize.py first")
    array = _require(hw, "array", "hardware.array")
    array_rows = int(array["rows"])
    array_cols = int(array["cols"])

    mapping = map_and_cost(
        n_docs=n_docs, dim=dim, n_queries=n_queries,
        array_rows=array_rows, array_cols=array_cols,
        area_per_array_m2=float(_require(array_cost, "area_per_array_m2", "MVM area")),
        latency_per_mvm_s=float(_require(array_cost, "latency_per_mvm_s", "MVM latency")),
        energy_per_mvm_j=float(_require(array_cost, "energy_per_mvm_j", "MVM energy")),
        array_budget=array_budget,
    )

    # --- top-k cost (Part 2) ---
    if "comparator_cost" not in array_cost:
        raise KeyError(
            "array_cost has no 'comparator_cost' block. Re-run characterize.py "
            "with a NeuroSim binary that includes the --comparator-cost mode "
            "(rebuild: cd modules/NeuroSim/NeuroSIM && make)."
        )
    cc = array_cost["comparator_cost"]
    a_cmp = float(_require(cc, "area_per_comparator_m2", "comparator area"))
    l_cmp = float(_require(cc, "latency_per_comparison_s", "comparator latency"))
    e_cmp = float(_require(cc, "energy_per_comparison_j", "comparator energy"))

    # Headline selection: pick top-k out of the whole database, per query.
    local = topk_cost(
        n_candidates=n_docs, k=k,
        area_per_comparator_m2=a_cmp,
        latency_per_comparison_s=l_cmp,
        energy_per_comparison_j=e_cmp,
        num_comparators=num_comparators, model=model,
    )

    # Optional smaller merge pass: re-rank pooled local winners (tiles*k).
    merge = None
    if include_merge_pass:
        pooled = max(mapping.num_doc_tiles * k, k + 1)
        merge = topk_cost(
            n_candidates=pooled, k=k,
            area_per_comparator_m2=a_cmp,
            latency_per_comparison_s=l_cmp,
            energy_per_comparison_j=e_cmp,
            num_comparators=num_comparators, model=model,
        )

    per_query_topk_energy = local.energy_j + (merge.energy_j if merge else 0.0)
    per_query_topk_latency = local.latency_s + (merge.latency_s if merge else 0.0)
    per_query_comparisons = local.comparisons + (merge.comparisons if merge else 0)

    topk_energy = per_query_topk_energy * n_queries
    topk_latency = per_query_topk_latency * n_queries          # queries serialized
    topk_area = local.area_m2                                  # comparator bank, reused
    topk_comparisons_total = per_query_comparisons * n_queries

    total_energy = mapping.energy_j + topk_energy
    total_latency = mapping.latency_s + topk_latency
    total_area = mapping.area_m2 + topk_area

    def share(part, whole):
        return (part / whole) if whole > 0 else 0.0

    return PerfEvalReport(
        n_docs=n_docs, dim=dim, n_queries=n_queries, k=k,
        mvm_area_m2=mapping.area_m2,
        mvm_latency_s=mapping.latency_s,
        mvm_energy_j=mapping.energy_j,
        topk_area_m2=topk_area,
        topk_latency_s=topk_latency,
        topk_energy_j=topk_energy,
        topk_comparisons_total=topk_comparisons_total,
        topk_model=model,
        total_area_m2=total_area,
        total_latency_s=total_latency,
        total_energy_j=total_energy,
        topk_energy_share=share(topk_energy, total_energy),
        topk_latency_share=share(topk_latency, total_latency),
        topk_area_share=share(topk_area, total_area),
        mapping=asdict(mapping),
        topk_local=asdict(local),
        topk_merge=asdict(merge) if merge else {},
    )


def evaluate_from_file(n_docs, dim, n_queries, k, array_cost_path, **kwargs):
    """Load array_cost.json and evaluate a workload in one call."""
    with open(array_cost_path, "r") as f:
        array_cost = json.load(f)
    return evaluate_workload(n_docs, dim, n_queries, k, array_cost, **kwargs)


if __name__ == "__main__":
    demo_cost = {
        "area_per_array_m2": 3.6e-8,
        "latency_per_mvm_s": 2.19e-8,
        "energy_per_mvm_j": 4.77e-10,
        "hardware": {"array": {"rows": 512, "cols": 512}},
        "comparator_cost": {
            "area_per_comparator_m2": 1.32e-11,
            "latency_per_comparison_s": 2.68e-10,
            "energy_per_comparison_j": 5.52e-14,
        },
    }
    print(f"{'n_docs':>10} {'E total':>12} {'E top-k%':>9} {'L top-k%':>9}")
    for n in (1_000, 10_000, 100_000, 1_000_000):
        r = evaluate_workload(n_docs=n, dim=384, n_queries=200, k=10, array_cost=demo_cost)
        print(f"{n:>10} {r.total_energy_j:>12.3e} "
              f"{r.topk_energy_share*100:>8.1f}% {r.topk_latency_share*100:>8.1f}%")
