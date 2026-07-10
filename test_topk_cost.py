"""
test_topk_cost.py — sanity checks for PerfEval Part 2a (topk_cost.py).

Pure Python, no NeuroSim needed: unit per-comparison costs (area=1,
latency=1, energy=1) make the aggregated numbers easy to verify by hand.

Run:  python test_topk_cost.py
"""

import sys

sys.path.insert(0, "modules/CAMASIM")
try:
    from CAMASim.performance.topk_cost import (
        topk_cost, topk_comparisons, retrieval_n_candidates,
        comparisons_linear, comparisons_tree,
    )
except ModuleNotFoundError:
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, "modules", "CAMASIM"))
    from CAMASim.performance.topk_cost import (
        topk_cost, topk_comparisons, retrieval_n_candidates,
        comparisons_linear, comparisons_tree,
    )


UNIT = dict(area_per_comparator_m2=1.0,
            latency_per_comparison_s=1.0,
            energy_per_comparison_j=1.0)


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        raise AssertionError(name)


def test_linear_model_counts():
    print("test_linear_model_counts")
    check("k*N: 10*1000", comparisons_linear(1000, 10) == 10_000)
    check("k capped at N", comparisons_linear(5, 10) == 25)  # min(10,5)*5
    check("via topk_comparisons", topk_comparisons(1000, 10, "linear") == 10_000)


def test_cost_grows_with_N():
    print("test_cost_grows_with_N")
    small = topk_cost(1_000, 10, **UNIT)
    big = topk_cost(10_000, 10, **UNIT)
    check("10x the docs -> 10x comparisons", big.comparisons == 10 * small.comparisons)
    check("energy grows with N", big.energy_j == 10 * small.energy_j)
    check("this is the size-dependent cost", big.energy_j > small.energy_j)


def test_cost_grows_with_k():
    print("test_cost_grows_with_k")
    k5 = topk_cost(10_000, 5, **UNIT)
    k20 = topk_cost(10_000, 20, **UNIT)
    check("4x k -> 4x comparisons", k20.comparisons == 4 * k5.comparisons)
    check("energy grows with k", k20.energy_j > k5.energy_j)


def test_energy_equals_comparisons():
    print("test_energy_equals_comparisons")
    r = topk_cost(2_048, 10, **UNIT)
    check("unit energy => energy == comparisons", r.energy_j == r.comparisons)
    check("serial by default => latency == comparisons", r.latency_s == r.comparisons)
    check("1 comparator by default => area == 1", r.area_m2 == 1.0)


def test_comparator_parallelism_tradeoff():
    print("test_comparator_parallelism_tradeoff")
    serial = topk_cost(1_000, 10, num_comparators=1, **UNIT)
    parallel = topk_cost(1_000, 10, num_comparators=100, **UNIT)
    check("more comparators -> fewer serial steps",
          parallel.serial_steps < serial.serial_steps)
    check("more comparators -> lower latency", parallel.latency_s < serial.latency_s)
    check("more comparators -> more area", parallel.area_m2 > serial.area_m2)
    check("energy unchanged by parallelism", parallel.energy_j == serial.energy_j)


def test_tree_model_cheaper_at_scale():
    print("test_tree_model_cheaper_at_scale")
    n = 100_000
    lin = topk_comparisons(n, 10, "linear")
    tree = comparisons_tree(n, 10)
    check("tree << linear at large N", tree < lin)
    check("tree model selectable via topk_cost",
          topk_cost(n, 10, model="tree", **UNIT).comparisons == tree)


def test_retrieval_n_candidates():
    print("test_retrieval_n_candidates")
    # full-database selection: N = n_docs
    check("no preselect -> N == n_docs",
          retrieval_n_candidates(50_000, 10, num_doc_tiles=98, local_preselect=False) == 50_000)
    # merge-pool selection: N = num_doc_tiles * k
    check("preselect -> N == tiles*k",
          retrieval_n_candidates(50_000, 10, num_doc_tiles=98, local_preselect=True) == 980)


def test_bad_inputs_raise():
    print("test_bad_inputs_raise")
    for bad in [dict(n_candidates=0, k=10), dict(n_candidates=100, k=0)]:
        try:
            topk_cost(**bad, **UNIT)
            check("should have raised", False)
        except ValueError:
            check("raised ValueError on bad input", True)
    try:
        topk_comparisons(100, 10, model="nope")
        check("bad model should raise", False)
    except ValueError:
        check("raised ValueError on bad model", True)


if __name__ == "__main__":
    tests = [
        test_linear_model_counts,
        test_cost_grows_with_N,
        test_cost_grows_with_k,
        test_energy_equals_comparisons,
        test_comparator_parallelism_tradeoff,
        test_tree_model_cheaper_at_scale,
        test_retrieval_n_candidates,
        test_bad_inputs_raise,
    ]
    for t in tests:
        t()
    print("\nAll top-k cost sanity checks passed.")
