"""
mapper.py — Part 1 of PerfEval: MVM cost via data-to-array mapping.

This module answers a single question:

    "Given a fixed piece of hardware (one array, whose area / latency / energy
     we already measured with NeuroSim), and a dataset of arbitrary size, how
     much does it cost to run all the searches?"

The hardware is characterised ONCE (see characterize.py, which writes
array_cost.json). Everything here is pure Python arithmetic on top of those
three per-array numbers — no circuit tool is invoked.

Mapping model (agreed design)
-----------------------------
A crossbar computes one matrix-vector multiply (MVM). We store documents on
the array and stream query vectors through it:

    * the embedding DIMENSION maps onto the array ROWS
      (the query enters along the rows; the crossbar sums along this axis —
       this is the dot-product / contraction axis)
    * the DOCUMENTS map onto the array COLUMNS
      (each column holds one stored document and outputs one score)

If the data does not fit on a single array we cut it into tiles:

    num_dim_tiles = ceil(dim      / array_rows)   # one dot product split across
                                                  # arrays -> partial sums to merge
    num_doc_tiles = ceil(n_docs   / array_cols)   # more documents -> more arrays,
                                                  # independent (parallel) columns

    num_arrays    = num_dim_tiles * num_doc_tiles

The number of arrays is NOT a user choice — it falls out of the mapping
(dataset size / array size). Data size is the dial the user turns; area,
latency and energy all follow from it.

Cost aggregation
----------------
    energy  = energy_per_mvm * total_MVMs
              (every MVM spends its own energy; just multiply)

    area    = area_per_array * physical_arrays
              (more arrays placed on the chip -> more area)

    latency = only the arrays that "wait their turn" add up.
              By default every tile is its own physical array and they all
              fire in parallel, so one query costs a single MVM latency.
              Queries are streamed one at a time (serial), so total latency
              scales with the query count.

Area <-> latency trade-off
--------------------------
With full parallelism (the default) area grows with data size while latency
stays flat — you paid area to keep latency down. If instead you cap how many
arrays are physically placed (``array_budget``), the tiles are reused in
serial waves: latency climbs but area is bounded. That push-and-pull between
area and latency is exactly what Part 1 is meant to expose. ``array_budget``
defaults to None (unlimited / full parallel) so the user never has to set it.

NOTE (Part 2, not modelled here):
    * The partial-sum merge across dim-tiles, and the top-k selection cost,
      are Part 2 (comparator / adder peripheral cost). They are reported as
      structural counts here (num_dim_tiles) but not yet costed in latency.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class MappingResult:
    """Full breakdown of a single dataset-on-hardware mapping."""

    # --- inputs echoed back (for transparency / plotting) ---
    n_docs: int
    dim: int
    n_queries: int
    array_rows: int
    array_cols: int

    # --- tiling ---
    num_dim_tiles: int          # cuts along embedding dimension (rows)
    num_doc_tiles: int          # cuts along documents (columns)
    num_arrays: int             # tiles needed = dim_tiles * doc_tiles
    physical_arrays: int        # arrays actually placed (<= num_arrays if budgeted)
    serial_passes: int          # how many sequential waves of MVMs per query

    # --- work ---
    mvms_per_query: int         # = num_arrays (one MVM per array per query)
    total_mvms: int             # = n_queries * num_arrays

    # --- aggregated cost ---
    area_m2: float
    latency_s: float
    energy_j: float

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent)


def map_and_cost(
    n_docs: int,
    dim: int,
    n_queries: int,
    array_rows: int,
    array_cols: int,
    area_per_array_m2: float,
    latency_per_mvm_s: float,
    energy_per_mvm_j: float,
    array_budget: Optional[int] = None,
) -> MappingResult:
    """Map a dataset onto fixed-size arrays and aggregate the MVM cost.

    Parameters
    ----------
    n_docs, dim :
        Dataset shape — number of documents and embedding dimension.
    n_queries :
        How many query vectors are searched.
    array_rows, array_cols :
        Fixed hardware array size (dimension maps to rows, docs to columns).
    area_per_array_m2, latency_per_mvm_s, energy_per_mvm_j :
        The three per-array numbers measured once by NeuroSim
        (see characterize.py / array_cost.json).
    array_budget :
        Optional cap on how many arrays are physically placed. None (default)
        means unlimited — every tile is its own array and all run in parallel.
        Setting a smaller number reuses arrays in serial waves (area down,
        latency up). The user normally leaves this as None.

    Returns
    -------
    MappingResult
        Tile counts, work counts, and aggregated area / latency / energy.
    """
    if min(n_docs, dim, n_queries, array_rows, array_cols) <= 0:
        raise ValueError("n_docs, dim, n_queries, array_rows, array_cols must all be > 0")
    if array_budget is not None and array_budget <= 0:
        raise ValueError("array_budget must be > 0 (or None for unlimited)")

    # --- tiling: how many pieces does the data cut into? ---
    num_dim_tiles = math.ceil(dim / array_rows)
    num_doc_tiles = math.ceil(n_docs / array_cols)
    num_arrays = num_dim_tiles * num_doc_tiles

    # --- how many of those arrays do we physically place? ---
    if array_budget is None:
        physical_arrays = num_arrays          # full parallelism
    else:
        physical_arrays = min(array_budget, num_arrays)

    # arrays that must "wait their turn" = serial waves per query
    serial_passes = math.ceil(num_arrays / physical_arrays)

    # --- work counts ---
    mvms_per_query = num_arrays               # one MVM per array, per query
    total_mvms = n_queries * mvms_per_query

    # --- aggregate cost ---
    # Energy: every MVM spends its own energy -> just multiply.
    energy_j = energy_per_mvm_j * total_mvms

    # Area: proportional to arrays actually placed on the chip.
    area_m2 = area_per_array_m2 * physical_arrays

    # Latency: within one query, all physically-placed arrays overlap, so each
    # wave costs one MVM latency; serial_passes waves add up. Queries stream
    # one at a time, so multiply by n_queries.
    # (Partial-sum merge across dim-tiles and top-k cost are Part 2 — not here.)
    latency_s = n_queries * serial_passes * latency_per_mvm_s

    return MappingResult(
        n_docs=n_docs,
        dim=dim,
        n_queries=n_queries,
        array_rows=array_rows,
        array_cols=array_cols,
        num_dim_tiles=num_dim_tiles,
        num_doc_tiles=num_doc_tiles,
        num_arrays=num_arrays,
        physical_arrays=physical_arrays,
        serial_passes=serial_passes,
        mvms_per_query=mvms_per_query,
        total_mvms=total_mvms,
        area_m2=area_m2,
        latency_s=latency_s,
        energy_j=energy_j,
    )


def load_array_cost(path: str) -> dict:
    """Load an array_cost.json produced by characterize.py.

    Returns the dict; callers pull area_per_array_m2 / latency_per_mvm_s /
    energy_per_mvm_j out of it.
    """
    with open(path, "r") as f:
        return json.load(f)


def map_from_cost_file(
    n_docs: int,
    dim: int,
    n_queries: int,
    array_cost_path: str,
    array_budget: Optional[int] = None,
) -> MappingResult:
    """Convenience wrapper: read array_cost.json and map a dataset in one call.

    The array size (rows/cols) and the three per-array costs all come from the
    JSON that characterize.py wrote, so the caller only supplies dataset shape.
    """
    cost = load_array_cost(array_cost_path)
    hw = cost.get("hardware", {})
    array = hw.get("array", {})
    array_rows = int(array.get("rows"))
    array_cols = int(array.get("cols"))
    return map_and_cost(
        n_docs=n_docs,
        dim=dim,
        n_queries=n_queries,
        array_rows=array_rows,
        array_cols=array_cols,
        area_per_array_m2=float(cost["area_per_array_m2"]),
        latency_per_mvm_s=float(cost["latency_per_mvm_s"]),
        energy_per_mvm_j=float(cost["energy_per_mvm_j"]),
        array_budget=array_budget,
    )


if __name__ == "__main__":
    # Tiny illustrative run (uses placeholder per-array numbers).
    demo = map_and_cost(
        n_docs=5000, dim=384, n_queries=200,
        array_rows=512, array_cols=512,
        area_per_array_m2=3.6e-9,
        latency_per_mvm_s=4.58e-6,
        energy_per_mvm_j=3.77e-9,
    )
    print(demo.to_json())
