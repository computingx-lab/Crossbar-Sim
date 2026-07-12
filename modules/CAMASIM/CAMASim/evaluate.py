"""
evaluate.py — one unified driver over the functional line and PerfEval.

You hand it a single config (hardware + dataset + noise) and it runs BOTH
lines on that same configuration and returns the four numbers together:

    input:  one config
    run functional line  -> Recall@k, nDCG@k   (accuracy under the noise model)
    run PerfEval line     -> area, latency, energy
    return: { Recall@k, nDCG@k, area, latency, energy }

Design (see the discussion in datasets.py): the CORE works on embeddings passed
in as arrays, which keeps it fast, deterministic and testable. If you don't pass
embeddings and the config names a dataset, the thin loader in datasets.py embeds
it for you.

Functional path: if the documents fit in one array (n_docs <= array cols) the
retrieval is a single-array selection. If they don't, the docs are tiled across
arrays and the retrieval stacks the noise the way real hardware does:

    each array: write noise (frozen) + fresh read noise -> noisy local top-k
                (with compare-side comparator error)
        -> pool all local winners
        -> global merge top-k (compare-side comparator error again)

So read noise is redrawn per array, and comparator error (sigma_compare) enters
at BOTH the local selections and the merge -- the three-injection-point picture.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from typing import Optional

try:
    import numpy as np
    from CAMASim.function.module.distance import crossbar_innerproduct_pairwise
    from CAMASim.function.module.sensing import get_array_topk_results
    from CAMASim.function.module.merge import topk_merge
    from CAMASim.metrics.retrieval import ideal_topk, recall_at_k, ndcg_at_k
    from CAMASim.performance.perfeval import evaluate_workload, PerfEvalReport
except ModuleNotFoundError:  # allow running this file directly
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))
    import numpy as np
    from CAMASim.function.module.distance import crossbar_innerproduct_pairwise
    from CAMASim.function.module.sensing import get_array_topk_results
    from CAMASim.function.module.merge import topk_merge
    from CAMASim.metrics.retrieval import ideal_topk, recall_at_k, ndcg_at_k
    from CAMASim.performance.perfeval import evaluate_workload, PerfEvalReport


@dataclass
class UnifiedResult:
    """The four headline numbers from one config, plus context and breakdown."""

    recall_at_k: float
    ndcg_at_k: float
    area_m2: float
    latency_s: float
    energy_j: float

    n_docs: int
    dim: int
    n_queries: int
    k: int
    noise_type: str
    num_doc_arrays: int          # how many arrays the docs were tiled across

    perfeval: dict = field(default_factory=dict)

    def four_numbers(self) -> dict:
        return {
            "recall_at_k": self.recall_at_k,
            "ndcg_at_k": self.ndcg_at_k,
            "area_m2": self.area_m2,
            "latency_s": self.latency_s,
            "energy_j": self.energy_j,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent)


def _load_array_cost(hardware):
    if isinstance(hardware, str):
        with open(hardware, "r") as f:
            return json.load(f)
    if isinstance(hardware, dict):
        return hardware
    raise ValueError("config['hardware'] must be an array_cost dict or a path to array_cost.json")


def _run_functional(docs, queries, k, noise, array_cols):
    """Noisy retrieval -> predicted top-k indices per query.

    One array if docs fit (n_docs <= array_cols); otherwise tile docs across
    arrays with per-array fresh read noise, noisy local top-k, and a noisy merge.
    Returns (pred_list, num_tiles).
    """
    n_docs = docs.shape[0]
    n_queries = queries.shape[0]
    sigma_compare = float(noise.get("sigma_compare", 0.0))
    num_tiles = int(math.ceil(n_docs / array_cols)) if array_cols and array_cols > 0 else 1

    # --- single array: one selection (comparator error at that one point) ---
    if num_tiles <= 1:
        scores = crossbar_innerproduct_pairwise(docs, queries, noise)
        pred, _ = get_array_topk_results(scores, k, sigma_compare=sigma_compare)
        return pred, 1

    # --- multi array: per-array local top-k, then merge ---
    cand_idx = [[] for _ in range(n_queries)]
    cand_score = [[] for _ in range(n_queries)]
    for t in range(num_tiles):
        s = t * array_cols
        e = min(n_docs, (t + 1) * array_cols)
        # fresh read noise per array; write noise frozen within this one call
        tile_scores = crossbar_innerproduct_pairwise(docs[s:e], queries, noise)
        local_idx, local_score = get_array_topk_results(tile_scores, k, sigma_compare=sigma_compare)
        for qi in range(n_queries):
            cand_idx[qi].extend((np.asarray(local_idx[qi]) + s).tolist())
            cand_score[qi].extend(np.asarray(local_score[qi]).tolist())

    pred = [topk_merge(cand_idx[qi], cand_score[qi], k, sigma_compare=sigma_compare)
            for qi in range(n_queries)]
    return pred, num_tiles


def evaluate(config, docs=None, queries=None, qrels=None) -> UnifiedResult:
    """Run the functional line and PerfEval on ONE config; return both results.

    config keys:
        hardware : array_cost dict (from characterize.py) OR path to array_cost.json
        dataset  : dataset name/kwargs (only used if docs/queries not passed)
        noise    : dict for the noise model (noise_type, sigmas, adc_bits,
                   sigma_read, sigma_compare, level_* ...). Empty = noiseless.
        k        : top-k (default 10)
        # optional PerfEval knobs: array_budget, num_comparators, num_adders, topk_model
    docs, queries : optional pre-embedded arrays. If given, the dataset is not
                    loaded (fast path used by tests/sweeps).
    """
    # 1. Resolve embeddings.
    if docs is None or queries is None:
        if "dataset" not in config:
            raise ValueError("pass docs=/queries= embeddings, or set config['dataset'].")
        from CAMASim.datasets import resolve_dataset   # lazy: heavy stack only if needed
        docs, queries, qrels = resolve_dataset(config["dataset"])

    docs = np.asarray(docs, dtype=float)
    queries = np.asarray(queries, dtype=float)
    n_docs, dim = docs.shape
    n_queries = int(queries.shape[0])
    k = int(config.get("k", 10))

    array_cost = _load_array_cost(config["hardware"])
    array_cols = int(array_cost.get("hardware", {}).get("array", {}).get("cols", n_docs))

    # 2. Functional line -> Recall@k, nDCG@k under the chosen noise model.
    noise = dict(config.get("noise", {}))
    gt = ideal_topk(docs, queries, k)                       # exact noiseless ground truth
    pred, num_doc_arrays = _run_functional(docs, queries, k, noise, array_cols)
    recall = recall_at_k(pred, gt)
    ndcg = ndcg_at_k(pred, gt)

    # 3. PerfEval line -> area, latency, energy (same config, same shape).
    report: PerfEvalReport = evaluate_workload(
        n_docs=n_docs, dim=dim, n_queries=n_queries, k=k, array_cost=array_cost,
        array_budget=config.get("array_budget"),
        num_comparators=int(config.get("num_comparators", 1)),
        num_adders=int(config.get("num_adders", 1)),
        model=config.get("topk_model", "linear"),
    )

    # 4. Return the four numbers together (+ context + full PerfEval breakdown).
    return UnifiedResult(
        recall_at_k=recall,
        ndcg_at_k=ndcg,
        area_m2=report.total_area_m2,
        latency_s=report.total_latency_s,
        energy_j=report.total_energy_j,
        n_docs=n_docs, dim=dim, n_queries=n_queries, k=k,
        noise_type=noise.get("noise_type", "gaussian"),
        num_doc_arrays=num_doc_arrays,
        perfeval=asdict(report),
    )


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    d = rng.normal(size=(2000, 128)).astype("float32")
    q = rng.normal(size=(50, 128)).astype("float32")
    cost = {
        "area_per_array_m2": 5.5e-8, "latency_per_mvm_s": 3.06e-7, "energy_per_mvm_j": 2.9e-9,
        "hardware": {"array": {"rows": 512, "cols": 512}},
        "comparator_cost": {"area_per_comparator_m2": 1.32e-11,
                            "latency_per_comparison_s": 2.68e-10, "energy_per_comparison_j": 5.52e-14},
    }
    print("single array (2000 docs, 512 cols -> tiled):")
    for name, noise in [("noiseless", {}),
                        ("read noise", {"sigma_read": 0.05}),
                        ("+ comparator", {"sigma_read": 0.05, "sigma_compare": 0.05})]:
        r = evaluate({"hardware": cost, "noise": noise, "k": 10}, docs=d, queries=q)
        print(f"  {name:>14}: Recall@10={r.recall_at_k:.3f}  arrays={r.num_doc_arrays}")
