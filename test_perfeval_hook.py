"""
test_perfeval_hook.py — checks the in-loop top-k cost hook in CAMSearch (#1).

Exercises the sensing-stage and merge-stage accounting directly (no full data
pipeline needed) and confirms it's completely off by default.

Run:  python test_perfeval_hook.py
"""

import sys

sys.path.insert(0, "modules/CAMASIM")
try:
    import numpy as np
    from CAMASim.function.search import CAMSearch
except ModuleNotFoundError:
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, "modules", "CAMASIM"))
    import numpy as np
    from CAMASim.function.search import CAMSearch


UNIT_CC = {"area_per_comparator_m2": 1.0,
           "latency_per_comparison_s": 1.0,
           "energy_per_comparison_j": 1.0}


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        raise AssertionError(name)


def make_search(enable, num_comparators=1):
    qc = {"searchScheme": "topk", "parameter": 10, "distance": "crossbar_ip"}
    if enable:
        qc["perfeval"] = {"comparator_cost": UNIT_CC, "num_comparators": num_comparators}
    ac = {"sensing": "topk", "row": 512, "cell": "6T"}
    return CAMSearch(qc, ac)


def test_disabled_by_default():
    print("test_disabled_by_default")
    cs = make_search(enable=False)
    check("no perfeval -> report is None", cs.get_perfeval_report() is None)
    # sensing still runs fine with accounting off
    cs.array_sensing(np.random.rand(3, 40))
    check("sensing works with accounting off", True)


def test_sensing_stage_counts():
    print("test_sensing_stage_counts")
    cs = make_search(enable=True)
    cs._reset_perfeval()
    # one array: 4 queries x 100 candidates, k=10 -> k*N*n_q = 10*100*4
    cs.array_sensing(np.random.rand(4, 100))
    check("per-array local top-k = k*n_docs*n_queries", cs._pe_sensing_cmp == 10 * 100 * 4)
    # a second array accumulates
    cs.array_sensing(np.random.rand(4, 100))
    check("second array accumulates", cs._pe_sensing_cmp == 2 * 10 * 100 * 4)


def test_merge_stage_counts():
    print("test_merge_stage_counts")
    cs = make_search(enable=True)
    cs._reset_perfeval()
    pooled_idx = list(range(50))
    pooled_scores = list(np.random.rand(50))
    cs.merge_indices(pooled_idx, pooled_scores)
    check("cross-array merge = k*n_pool", cs._pe_merge_cmp == 10 * 50)


def test_report_aggregates():
    print("test_report_aggregates")
    cs = make_search(enable=True)
    cs._reset_perfeval()
    cs.array_sensing(np.random.rand(2, 100))          # 10*100*2 = 2000
    cs.merge_indices(list(range(30)), list(np.random.rand(30)))  # 10*30 = 300
    r = cs.get_perfeval_report()
    check("sensing counted", r["sensing_comparisons"] == 2000)
    check("merge counted", r["merge_comparisons"] == 300)
    check("total = sensing + merge", r["topk_comparisons_total"] == 2300)
    check("unit energy => energy == total comparisons", r["topk_energy_j"] == 2300)
    check("1 comparator => latency == total", r["topk_latency_s"] == 2300)
    check("area == num_comparators", r["topk_area_m2"] == 1.0)


def test_more_comparators_cut_latency():
    print("test_more_comparators_cut_latency")
    cs = make_search(enable=True, num_comparators=100)
    cs._reset_perfeval()
    cs.array_sensing(np.random.rand(1, 1000))         # 10*1000 = 10000
    r = cs.get_perfeval_report()
    check("energy unchanged by comparators", r["topk_energy_j"] == 10000)
    check("latency = ceil(total/num_comp)", r["topk_latency_s"] == 100)   # 10000/100
    check("area = num_comparators", r["topk_area_m2"] == 100.0)


if __name__ == "__main__":
    for t in [test_disabled_by_default, test_sensing_stage_counts,
              test_merge_stage_counts, test_report_aggregates,
              test_more_comparators_cut_latency]:
        t()
    print("\nAll PerfEval merge-stage hook checks passed.")
