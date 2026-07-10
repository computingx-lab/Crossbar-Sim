"""
test_mapper.py — sanity checks for PerfEval Part 1 (mapper.py).

Pure Python, no NeuroSim needed: we feed simple per-array costs (area=1,
latency=1, energy=1) so the aggregated numbers are easy to check by hand.

Run:  python test_mapper.py
"""

import sys

# Resolve the CAMASim package (same doubled-path layout the sweep scripts use).
sys.path.insert(0, "modules/CAMASIM")
try:
    from CAMASim.performance.mapper import map_and_cost
except ModuleNotFoundError:
    # allow running from an arbitrary cwd
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, "modules", "CAMASIM"))
    from CAMASim.performance.mapper import map_and_cost


UNIT = dict(area_per_array_m2=1.0, latency_per_mvm_s=1.0, energy_per_mvm_j=1.0)


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        raise AssertionError(name)


def test_fits_in_one_array():
    print("test_fits_in_one_array")
    r = map_and_cost(n_docs=300, dim=128, n_queries=10,
                     array_rows=512, array_cols=512, **UNIT)
    check("dim fits -> 1 dim tile", r.num_dim_tiles == 1)
    check("docs fit -> 1 doc tile", r.num_doc_tiles == 1)
    check("1 array total", r.num_arrays == 1)
    check("area = 1 array", r.area_m2 == 1.0)
    check("energy = 10 MVMs", r.energy_j == 10.0)
    check("latency = 10 queries x 1 pass", r.latency_s == 10.0)


def test_exact_multiple_tiling():
    print("test_exact_multiple_tiling")
    # dim 1024 / 512 = 2 dim tiles; docs 1024 / 512 = 2 doc tiles -> 4 arrays
    r = map_and_cost(n_docs=1024, dim=1024, n_queries=5,
                     array_rows=512, array_cols=512, **UNIT)
    check("2 dim tiles", r.num_dim_tiles == 2)
    check("2 doc tiles", r.num_doc_tiles == 2)
    check("4 arrays", r.num_arrays == 4)
    check("total MVMs = 5 queries x 4 arrays", r.total_mvms == 20)
    check("energy = 20", r.energy_j == 20.0)
    check("area = 4 (all placed)", r.area_m2 == 4.0)


def test_ceil_boundary():
    print("test_ceil_boundary")
    # one past a boundary should round UP to an extra tile
    r = map_and_cost(n_docs=513, dim=513, n_queries=1,
                     array_rows=512, array_cols=512, **UNIT)
    check("513 docs -> 2 doc tiles", r.num_doc_tiles == 2)
    check("513 dim  -> 2 dim tiles", r.num_dim_tiles == 2)
    check("4 arrays", r.num_arrays == 4)


def test_energy_scales_linearly():
    print("test_energy_scales_linearly")
    small = map_and_cost(n_docs=5000, dim=384, n_queries=100,
                         array_rows=512, array_cols=512, **UNIT)
    big = map_and_cost(n_docs=10000, dim=384, n_queries=100,
                       array_rows=512, array_cols=512, **UNIT)
    # doubling docs (across a clean tile boundary) doubles doc tiles -> doubles energy
    check("5000 docs -> 10 doc tiles", small.num_doc_tiles == 10)
    check("10000 docs -> 20 doc tiles", big.num_doc_tiles == 20)
    check("energy doubles", big.energy_j == 2 * small.energy_j)


def test_full_parallel_latency_is_flat():
    print("test_full_parallel_latency_is_flat")
    # With unlimited arrays, latency depends only on query count, not data size.
    a = map_and_cost(n_docs=1000, dim=384, n_queries=50,
                     array_rows=512, array_cols=512, **UNIT)
    b = map_and_cost(n_docs=50000, dim=384, n_queries=50,
                     array_rows=512, array_cols=512, **UNIT)
    check("more data uses more arrays", b.num_arrays > a.num_arrays)
    check("but latency stays flat (all parallel)", a.latency_s == b.latency_s == 50.0)
    check("area grows with data", b.area_m2 > a.area_m2)


def test_array_budget_tradeoff():
    print("test_array_budget_tradeoff")
    # 4 arrays needed; cap at 2 physical -> 2 serial waves.
    full = map_and_cost(n_docs=1024, dim=1024, n_queries=10,
                        array_rows=512, array_cols=512, **UNIT)
    capped = map_and_cost(n_docs=1024, dim=1024, n_queries=10,
                          array_rows=512, array_cols=512, array_budget=2, **UNIT)
    check("full: 1 serial pass", full.serial_passes == 1)
    check("capped: 2 serial passes", capped.serial_passes == 2)
    check("capped latency is higher", capped.latency_s == 2 * full.latency_s)
    check("capped area is lower", capped.area_m2 < full.area_m2)
    check("energy unchanged by parallelism", capped.energy_j == full.energy_j)


def test_bad_inputs_raise():
    print("test_bad_inputs_raise")
    for kwargs in [
        dict(n_docs=0, dim=128, n_queries=1, array_rows=512, array_cols=512),
        dict(n_docs=100, dim=128, n_queries=1, array_rows=0, array_cols=512),
    ]:
        try:
            map_and_cost(**kwargs, **UNIT)
            check("should have raised", False)
        except ValueError:
            check("raised ValueError on bad input", True)


if __name__ == "__main__":
    tests = [
        test_fits_in_one_array,
        test_exact_multiple_tiling,
        test_ceil_boundary,
        test_energy_scales_linearly,
        test_full_parallel_latency_is_flat,
        test_array_budget_tradeoff,
        test_bad_inputs_raise,
    ]
    for t in tests:
        t()
    print("\nAll mapper sanity checks passed.")
