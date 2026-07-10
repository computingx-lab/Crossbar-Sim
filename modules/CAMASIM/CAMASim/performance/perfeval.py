"""
perfeval.py — PerfEval integrator: MVM cost (Part 1) + top-k cost (Part 2).

Given a workload (how many documents, what embedding dimension, how many
queries, and k) plus a characterised piece of hardware (array_cost.json from
characterize.py), this returns the full cost of running the search, split into
its parts:

    * MVM cost          — from mapper.py: tiling the data onto arrays and paying
                          for every matrix-vector multiply.
    * partial-sum merge — Part 1 recombination: when the embedding dimension is
                          wider than the array it is split across arrays, and the
                          partial dot products must be ADDED back together. This
                          is ZERO unless dim > array rows (num_dim_tiles > 1).
    * top-k cost        — from topk_cost.py: the comparator-based selection that
                          picks the best k. The size-dependent cost CAMASim does
                          not report.

Everything is deterministic in the workload shape; noise/accuracy (Phase 1)
lives on a separate path and does not affect these numbers.
"""

from __future__ import annotations

import json
import math
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
    """Full PPA of one workload on one hardware config: MVM + psum-merge + top-k."""

    n_docs: int
    dim: int
    n_queries: int
    k: int

    # MVM (Part 1)
    mvm_area_m2: float
    mvm_latency_s: float
    mvm_energy_j: float

    # partial-sum merge (Part 1 recombination; zero unless dim > array rows)
    psum_area_m2: float
    psum_latency_s: float
    psum_energy_j: float
    psum_additions_total: int
    num_dim_tiles: int

    # top-k (Part 2)
    topk_area_m2: float
    topk_latency_s: float
    topk_energy_j: float
    topk_comparisons_total: int
    topk_model: str

    # totals
    total_area_m2: float
    total_latency_s: float
    total_energy_j: float

    # shares of total (novelty / bottleneck at a glance)
    topk_energy_share: float
    topk_latency_share: float
    psum_energy_share: float
    psum_latency_share: float

    # detailed sub-breakdowns
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
    num_adders: int = 1,
    model: str = "linear",
    include_merge_pass: bool = True,
) -> PerfEvalReport:
    """Combine MVM, partial-sum merge, and top-k into one cost report."""
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

    # --- partial-sum merge (Part 1 recombination) ---
    # Only exists when one dot product is split across arrays (dim > rows).
    # Each of the n_docs scores needs (num_dim_tiles - 1) additions to combine
    # its partial pieces, for every query. Cost model mirrors the top-k bank:
    # additions run across a bank of `num_adders` adders (serial waves add up).
    n_dim_tiles = mapping.num_dim_tiles
    psum_additions_total = 0
    psum_energy = psum_latency = psum_area = 0.0
    if n_dim_tiles > 1:
        if "adder_cost" not in array_cost:
            raise KeyError(
                "dim > array rows requires an 'adder_cost' block in array_cost, "
                "but it is missing. Re-run characterize.py with a NeuroSim binary "
                "that includes the --adder-cost mode (rebuild: make)."
            )
        ac = array_cost["adder_cost"]
        a_add = float(_require(ac, "area_per_adder_m2", "adder area"))
        l_add = float(_require(ac, "latency_per_add_s", "adder latency"))
        e_add = float(_require(ac, "energy_per_add_j", "adder energy"))

        adds_per_query = n_docs * (n_dim_tiles - 1)
        psum_additions_total = adds_per_query * n_queries
        psum_energy = e_add * psum_additions_total
        psum_area = a_add * num_adders
        serial_steps = math.ceil(psum_additions_total / num_adders)
        psum_latency = serial_steps * l_add

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

    local = topk_cost(
        n_candidates=n_docs, k=k,
        area_per_comparator_m2=a_cmp,
        latency_per_comparison_s=l_cmp,
        energy_per_comparison_j=e_cmp,
        num_comparators=num_comparators, model=model,
    )
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
    topk_latency = per_query_topk_latency * n_queries
    topk_area = local.area_m2
    topk_comparisons_total = per_query_comparisons * n_queries

    # --- totals ---
    total_energy = mapping.energy_j + psum_energy + topk_energy
    total_latency = mapping.latency_s + psum_latency + topk_latency
    total_area = mapping.area_m2 + psum_area + topk_area

    def share(part, whole):
        return (part / whole) if whole > 0 else 0.0

    return PerfEvalReport(
        n_docs=n_docs, dim=dim, n_queries=n_queries, k=k,
        mvm_area_m2=mapping.area_m2,
        mvm_latency_s=mapping.latency_s,
        mvm_energy_j=mapping.energy_j,
        psum_area_m2=psum_area,
        psum_latency_s=psum_latency,
        psum_energy_j=psum_energy,
        psum_additions_total=psum_additions_total,
        num_dim_tiles=n_dim_tiles,
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
        psum_energy_share=share(psum_energy, total_energy),
        psum_latency_share=share(psum_latency, total_latency),
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
        "area_per_array_m2": 5.5e-8,
        "latency_per_mvm_s": 3.06e-7,
        "energy_per_mvm_j": 2.9e-9,
        "hardware": {"array": {"rows": 512, "cols": 512}},
        "comparator_cost": {
            "area_per_comparator_m2": 1.32e-11,
            "latency_per_comparison_s": 2.68e-10,
            "energy_per_comparison_j": 5.52e-14,
        },
        "adder_cost": {
            "area_per_adder_m2": 5.0e-11,
            "latency_per_add_s": 3.0e-10,
            "energy_per_add_j": 1.0e-13,
        },
    }
    print(f"{'dim':>6} {'dim_tiles':>9} {'psum_E%':>8} {'topk_E%':>8}")
    for d in (384, 512, 1024, 4096):
        r = evaluate_workload(n_docs=100_000, dim=d, n_queries=200, k=10, array_cost=demo_cost)
        print(f"{d:>6} {r.num_dim_tiles:>9} {r.psum_energy_share*100:>7.1f}% {r.topk_energy_share*100:>7.1f}%")
