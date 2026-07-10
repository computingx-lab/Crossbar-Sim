"""
test_perfeval.py — sanity checks for the PerfEval integrator (perfeval.py).

Pure Python. A small synthetic array_cost with easy numbers checks the
arithmetic; a realistic one checks the share behaviour (which only means
something with a realistic comparator:MVM ratio).

Run:  python test_perfeval.py
"""

import sys

sys.path.insert(0, "modules/CAMASIM")
try:
    from CAMASim.performance.perfeval import evaluate_workload
except ModuleNotFoundError:
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, "modules", "CAMASIM"))
    from CAMASim.performance.perfeval import evaluate_workload


# Synthetic hardware for exact-arithmetic tests (unit-ish numbers).
COST = {
    "area_per_array_m2": 10.0,
    "latency_per_mvm_s": 2.0,
    "energy_per_mvm_j": 5.0,
    "hardware": {"array": {"rows": 512, "cols": 512}},
    "comparator_cost": {
        "area_per_comparator_m2": 1.0,
        "latency_per_comparison_s": 1.0,
        "energy_per_comparison_j": 1.0,
    },
}

# Realistic hardware (real characterised numbers) for share-behaviour tests.
REAL = {
    "area_per_array_m2": 3.60059e-08,
    "latency_per_mvm_s": 2.18966e-08,
    "energy_per_mvm_j": 4.77266e-10,
    "hardware": {"array": {"rows": 512, "cols": 512}},
    "comparator_cost": {
        "area_per_comparator_m2": 1.31834e-11,
        "latency_per_comparison_s": 2.68021e-10,
        "energy_per_comparison_j": 5.51672e-14,
    },
}


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        raise AssertionError(name)


def test_total_is_sum_of_parts():
    print("test_total_is_sum_of_parts")
    r = evaluate_workload(n_docs=5000, dim=384, n_queries=10, k=10, array_cost=COST)
    check("energy total = mvm + topk",
          abs(r.total_energy_j - (r.mvm_energy_j + r.topk_energy_j)) < 1e-9)
    check("latency total = mvm + topk",
          abs(r.total_latency_s - (r.mvm_latency_s + r.topk_latency_s)) < 1e-9)
    check("area total = mvm + topk",
          abs(r.total_area_m2 - (r.mvm_area_m2 + r.topk_area_m2)) < 1e-9)


def test_headline_comparisons_scale_with_docs():
    print("test_headline_comparisons_scale_with_docs")
    r = evaluate_workload(n_docs=5000, dim=384, n_queries=10, k=10,
                          array_cost=COST, include_merge_pass=False)
    check("comparisons = k*n_docs*n_queries",
          r.topk_comparisons_total == 10 * 5000 * 10)
    check("unit energy => topk energy == comparisons",
          abs(r.topk_energy_j - r.topk_comparisons_total) < 1e-6)


def test_topk_energy_grows_but_share_is_stable():
    print("test_topk_energy_grows_but_share_is_stable")
    small = evaluate_workload(n_docs=1_000, dim=384, n_queries=50, k=10, array_cost=REAL)
    big = evaluate_workload(n_docs=1_000_000, dim=384, n_queries=50, k=10, array_cost=REAL)
    check("top-k energy grows with database size", big.topk_energy_j > 100 * small.topk_energy_j)
    check("top-k energy share stays roughly constant",
          abs(big.topk_energy_share - small.topk_energy_share) < 0.1)


def test_topk_latency_dominates_and_grows():
    print("test_topk_latency_dominates_and_grows")
    small = evaluate_workload(n_docs=1_000, dim=384, n_queries=50, k=10, array_cost=REAL)
    big = evaluate_workload(n_docs=1_000_000, dim=384, n_queries=50, k=10, array_cost=REAL)
    check("top-k latency share grows with database size",
          big.topk_latency_share > small.topk_latency_share)
    check("top-k dominates latency at scale (serial vs parallel)",
          big.topk_latency_share > 0.95)


def test_more_comparators_cut_topk_latency():
    print("test_more_comparators_cut_topk_latency")
    one = evaluate_workload(n_docs=100_000, dim=384, n_queries=20, k=10,
                            array_cost=REAL, num_comparators=1)
    many = evaluate_workload(n_docs=100_000, dim=384, n_queries=20, k=10,
                             array_cost=REAL, num_comparators=256)
    check("more comparators -> lower top-k latency", many.topk_latency_s < one.topk_latency_s)
    check("more comparators -> unchanged top-k energy",
          abs(many.topk_energy_j - one.topk_energy_j) < 1e-15)


def test_merge_pass_adds_cost():
    print("test_merge_pass_adds_cost")
    with_merge = evaluate_workload(n_docs=50_000, dim=384, n_queries=5, k=10,
                                   array_cost=COST, include_merge_pass=True)
    without = evaluate_workload(n_docs=50_000, dim=384, n_queries=5, k=10,
                                array_cost=COST, include_merge_pass=False)
    check("merge pass increases comparisons",
          with_merge.topk_comparisons_total > without.topk_comparisons_total)
    check("but only slightly (merge pool << n_docs)",
          with_merge.topk_comparisons_total < 1.5 * without.topk_comparisons_total)


def test_dimension_tiling_flows_through():
    print("test_dimension_tiling_flows_through")
    narrow = evaluate_workload(n_docs=5000, dim=384, n_queries=10, k=10, array_cost=COST)
    wide = evaluate_workload(n_docs=5000, dim=1024, n_queries=10, k=10, array_cost=COST)
    check("wider embedding -> more MVM energy", wide.mvm_energy_j > narrow.mvm_energy_j)
    check("wider embedding -> same top-k energy (N=n_docs unchanged)",
          abs(wide.topk_energy_j - narrow.topk_energy_j) < 1e-9)


def test_missing_comparator_cost_raises():
    print("test_missing_comparator_cost_raises")
    bad = {k: v for k, v in COST.items() if k != "comparator_cost"}
    try:
        evaluate_workload(n_docs=1000, dim=384, n_queries=1, k=10, array_cost=bad)
        check("should have raised", False)
    except KeyError as e:
        check("raised KeyError mentioning comparator_cost", "comparator_cost" in str(e))


def test_realistic_numbers():
    print("test_realistic_numbers")
    r = evaluate_workload(n_docs=100_000, dim=384, n_queries=200, k=10, array_cost=REAL)
    check("all costs positive", r.total_energy_j > 0 and r.total_latency_s > 0 and r.total_area_m2 > 0)
    check("shares in [0,1]", 0.0 <= r.topk_energy_share <= 1.0)
    print(f"       (100k docs: total energy={r.total_energy_j:.3e} J, "
          f"top-k energy share={r.topk_energy_share*100:.1f}%, "
          f"top-k latency share={r.topk_latency_share*100:.1f}%)")


if __name__ == "__main__":
    tests = [
        test_total_is_sum_of_parts,
        test_headline_comparisons_scale_with_docs,
        test_topk_energy_grows_but_share_is_stable,
        test_topk_latency_dominates_and_grows,
        test_more_comparators_cut_topk_latency,
        test_merge_pass_adds_cost,
        test_dimension_tiling_flows_through,
        test_missing_comparator_cost_raises,
        test_realistic_numbers,
    ]
    for t in tests:
        t()
    print("\nAll PerfEval integrator sanity checks passed.")
