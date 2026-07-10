"""
characterize.py — Part 1 of PerfEval: measure ONE array with NeuroSim.

This is the "run once per hardware" step. The user describes their fixed
hardware in a small JSON file (array size, device, cell / adc / input bits);
this script hands that description to NeuroSim via the ``--cam-subarray``
bridge, reads back the three per-array numbers (area, latency, energy), and
writes them to ``array_cost.json``.

The user never authors the area/latency/energy numbers themselves — they only
describe the hardware. NeuroSim produces the costs; this script just caches
them so the sweeps in mapper.py never have to re-run the circuit simulator.

Usage
-----
    python -m CAMASim.performance.characterize hardware_example.json
    python -m CAMASim.performance.characterize hardware_example.json -o array_cost.json

Hardware-config JSON (input) looks like::

    {
      "array":      { "rows": 512, "cols": 512 },
      "device":     "FeFET",
      "cell_bits":  2,
      "adc_bits":   8,
      "input_bits": 8
    }

array_cost.json (output) looks like::

    {
      "area_per_array_m2": 3.6e-9,
      "latency_per_mvm_s": 4.58e-6,
      "energy_per_mvm_j":  3.77e-9,
      "source": "NeuroSim --cam-subarray",
      "hardware": { ...echo of the input config... }
    }

Device and ADC precision
------------------------
Device type is passed to NeuroSim via the NEUROSIM_CELLTYPE environment
variable (read by the Param constructor, so it must be set before the process
starts). ADC precision is passed as the optional trailing --cam-subarray
argument, which sets levelOutput = 2^adc_bits. So changing 'device' or
'adc_bits' in the hardware config now changes what NeuroSim simulates.
(How strongly device type shifts the numbers is bounded by NeuroSim's built-in
device models; customise device resistances in Param.cpp for finer control.)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np


# Cell-type codes NeuroSim uses internally (kept for provenance / future use).
CELL_TYPE_MAP = {
    "SRAM": 1,
    "RRAM": 2,
    "FeFET": 3,
    "nvCap": 4,
}


def find_repo_root(start: str) -> str:
    """Walk up from ``start`` until we hit the CAMASim-Hybrid repo root."""
    cur = os.path.abspath(start)
    while cur:
        if os.path.basename(cur) == "CAMASim-Hybrid":
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    # Fallback: assume this file is at modules/CAMASIM/CAMASim/performance/
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def find_neurosim_binary(repo_root: str) -> str:
    return os.path.join(repo_root, "modules", "NeuroSim", "NeuroSIM", "main")


def load_hardware_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = json.load(f)
    # minimal validation with friendly errors
    if "array" not in cfg or "rows" not in cfg["array"] or "cols" not in cfg["array"]:
        raise ValueError('hardware config needs an "array" block with "rows" and "cols"')
    return cfg


def _parse_metric_lines(stdout: str, prefix: str) -> dict:
    """Pull AREA/LATENCY/ENERGY out of PREFIX|METRIC|value lines."""
    out = {}
    for line in stdout.splitlines():
        if not line.startswith(prefix + "|"):
            continue
        parts = line.strip().split("|")
        if len(parts) != 3:
            continue
        _, metric, value_str = parts
        try:
            out[metric] = float(value_str)
        except ValueError:
            pass
    return out


def run_comparator_cost(binary: str, neurosim_dir: str, num_bit: int,
                        verbose: bool = True, env: dict | None = None) -> dict | None:
    """Run the --comparator-cost mode; return per-comparison area/latency/energy.

    Returns None (with a warning) if the mode fails, so a missing/old binary
    doesn't block the main array characterisation.
    """
    cmd = [binary, "--comparator-cost", str(num_bit)]
    if verbose:
        print(f"[characterize] running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, cwd=neurosim_dir, capture_output=True,
                                text=True, check=True, env=env)
    except subprocess.CalledProcessError as e:
        msg = (e.stderr or e.stdout or "").strip()
        print(f"[characterize] WARNING: --comparator-cost failed ({e.returncode}): {msg}")
        print("[characterize] (rebuild NeuroSim so the binary includes the new mode)")
        return None

    m = _parse_metric_lines(result.stdout, "CMP_METRIC")
    if not {"AREA", "LATENCY", "ENERGY"} <= set(m):
        print("[characterize] WARNING: could not parse CMP_METRIC lines; "
              "comparator cost omitted.")
        return None
    return {
        "area_per_comparator_m2": m["AREA"],
        "latency_per_comparison_s": m["LATENCY"],
        "energy_per_comparison_j": m["ENERGY"],
        "num_bit": num_bit,
        "source": "NeuroSim --comparator-cost",
    }


def run_adder_cost(binary: str, neurosim_dir: str, num_bit: int,
                   verbose: bool = True, env: dict | None = None) -> dict | None:
    """Run the --adder-cost mode; return per-addition area/latency/energy.

    Used to cost the partial-sum recombination when the embedding dimension is
    split across arrays (dim > array rows). Returns None (with a warning) if the
    mode fails, so an old binary doesn't block the rest of characterisation.
    """
    cmd = [binary, "--adder-cost", str(num_bit)]
    if verbose:
        print(f"[characterize] running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, cwd=neurosim_dir, capture_output=True,
                                text=True, check=True, env=env)
    except subprocess.CalledProcessError as e:
        msg = (e.stderr or e.stdout or "").strip()
        print(f"[characterize] WARNING: --adder-cost failed ({e.returncode}): {msg}")
        print("[characterize] (rebuild NeuroSim so the binary includes the new mode)")
        return None
    m = _parse_metric_lines(result.stdout, "ADD_METRIC")
    if not {"AREA", "LATENCY", "ENERGY"} <= set(m):
        print("[characterize] WARNING: could not parse ADD_METRIC lines; "
              "adder cost omitted.")
        return None
    return {
        "area_per_adder_m2": m["AREA"],
        "latency_per_add_s": m["LATENCY"],
        "energy_per_add_j": m["ENERGY"],
        "num_bit": num_bit,
        "source": "NeuroSim --adder-cost",
    }


def _write_representative_csvs(rows: int, cols: int, input_bits: int, tmpdir: str):
    """Generate representative weight/input CSVs to exercise one array.

    The per-array area/latency/energy depend on array size and technology, not
    on the specific stored values, so representative (dense random binary) data
    is used to characterise a typical single-MVM cost.
    """
    rng = np.random.default_rng(0)
    weight = rng.integers(0, 2, size=(rows, cols))
    inputs = rng.integers(0, 2, size=(input_bits, rows))

    weight_file = os.path.join(tmpdir, "cam_weight.csv")
    input_file = os.path.join(tmpdir, "cam_input.csv")
    np.savetxt(weight_file, weight, delimiter=",", fmt="%d")
    np.savetxt(input_file, inputs, delimiter=",", fmt="%d")
    return weight_file, input_file


def characterize(config_path: str, output_path: str | None = None, verbose: bool = True) -> dict:
    """Run NeuroSim once for the given hardware config; return the cost dict."""
    cfg = load_hardware_config(config_path)
    repo_root = find_repo_root(os.path.dirname(os.path.abspath(config_path)) or ".")
    binary = find_neurosim_binary(repo_root)

    if not os.path.exists(binary):
        raise FileNotFoundError(
            f"NeuroSim binary not found at {binary}.\n"
            "Build it first:  cd modules/NeuroSim/NeuroSIM && make clean && make"
        )

    rows = int(cfg["array"]["rows"])
    cols = int(cfg["array"]["cols"])
    cell_bits = int(cfg.get("cell_bits", 1))
    input_bits = int(cfg.get("input_bits", 8))
    device = str(cfg.get("device", "RRAM"))
    adc_bits = int(cfg.get("adc_bits", input_bits))
    memcelltype = CELL_TYPE_MAP.get(device, 2)

    if verbose:
        print(f"[characterize] hardware: {rows}x{cols} {device}, "
              f"cell_bits={cell_bits}, input_bits={input_bits}")
        if device not in CELL_TYPE_MAP:
            print(f"[characterize] WARNING: unknown device '{device}' "
                  f"(known: {list(CELL_TYPE_MAP)}). Recorded for provenance only.")
        else:
            print(f"[characterize] device -> NEUROSIM_CELLTYPE={memcelltype}, "
                  f"adc_bits={adc_bits} (levelOutput={2 ** adc_bits})")

    neurosim_dir = os.path.dirname(binary)

    # Device type is read by NeuroSim's Param constructor from the environment,
    # so it must be set BEFORE the process starts (not via a param override in
    # main()). ADC precision is passed as an explicit --cam-subarray argument.
    env = os.environ.copy()
    env["NEUROSIM_CELLTYPE"] = str(memcelltype)
    env["NEUROSIM_ROWS"] = str(rows)
    env["NEUROSIM_COLS"] = str(cols)

    with tempfile.TemporaryDirectory() as tmpdir:
        weight_file, input_file = _write_representative_csvs(rows, cols, input_bits, tmpdir)

        # --cam-subarray <weightfile> <inputfile> <numRows> <numCols> <numBitInput> <numBitSynapse> [adcBits]
        cmd = [
            binary, "--cam-subarray", weight_file, input_file,
            str(rows), str(cols), str(input_bits), str(cell_bits), str(adc_bits),
        ]
        if verbose:
            print(f"[characterize] running: {' '.join(cmd)}  (NEUROSIM_CELLTYPE={memcelltype})")

        try:
            result = subprocess.run(
                cmd, cwd=neurosim_dir, capture_output=True, text=True, check=True, env=env
            )
        except subprocess.CalledProcessError as e:
            msg = (e.stderr or e.stdout or "").strip()
            raise RuntimeError(f"NeuroSim exited with code {e.returncode}: {msg}") from e

    m = _parse_metric_lines(result.stdout, "SYS_METRIC")
    if not {"AREA", "LATENCY", "ENERGY"} <= set(m):
        raise RuntimeError(
            "Could not parse SYS_METRIC AREA/LATENCY/ENERGY from NeuroSim output.\n"
            "--- NeuroSim stdout ---\n" + result.stdout
        )
    area, latency, energy = m["AREA"], m["LATENCY"], m["ENERGY"]

    # Part 2: also characterise ONE comparison (top-k cost). Scores are
    # ADC-quantised, so the comparator precision follows adc_bits (defined above).
    comparator_cost = run_comparator_cost(binary, neurosim_dir, adc_bits,
                                          verbose=verbose, env=env)
    # Part 1 extra: cost of ONE addition, for recombining dim-split partial sums.
    # The partial sums are ADC outputs, so the adder width follows adc_bits.
    adder_cost = run_adder_cost(binary, neurosim_dir, adc_bits,
                                verbose=verbose, env=env)

    cost = {
        "area_per_array_m2": area,
        "latency_per_mvm_s": latency,
        "energy_per_mvm_j": energy,
        "source": "NeuroSim --cam-subarray",
        "hardware": cfg,
    }
    if comparator_cost is not None:
        cost["comparator_cost"] = comparator_cost
    if adder_cost is not None:
        cost["adder_cost"] = adder_cost

    if output_path is None:
        output_path = os.path.join(os.path.dirname(os.path.abspath(config_path)), "array_cost.json")
    with open(output_path, "w") as f:
        json.dump(cost, f, indent=2)

    if verbose:
        print(f"[characterize] per-array cost:")
        print(f"    area    = {area:.4e} m^2")
        print(f"    latency = {latency:.4e} s / MVM")
        print(f"    energy  = {energy:.4e} J / MVM")
        if comparator_cost is not None:
            print(f"[characterize] per-comparison cost ({adc_bits}-bit):")
            print(f"    area    = {comparator_cost['area_per_comparator_m2']:.4e} m^2")
            print(f"    latency = {comparator_cost['latency_per_comparison_s']:.4e} s / compare")
            print(f"    energy  = {comparator_cost['energy_per_comparison_j']:.4e} J / compare")
        if adder_cost is not None:
            print(f"[characterize] per-addition cost ({adc_bits}-bit):")
            print(f"    area    = {adder_cost['area_per_adder_m2']:.4e} m^2")
            print(f"    latency = {adder_cost['latency_per_add_s']:.4e} s / add")
            print(f"    energy  = {adder_cost['energy_per_add_j']:.4e} J / add")
        print(f"[characterize] wrote {output_path}")

    return cost


def main(argv=None):
    p = argparse.ArgumentParser(description="Characterise one array with NeuroSim (PerfEval Part 1).")
    p.add_argument("config", help="hardware-config JSON (array size, device, bits)")
    p.add_argument("-o", "--output", default=None,
                   help="output path for array_cost.json (default: next to config)")
    p.add_argument("-q", "--quiet", action="store_true", help="suppress progress output")
    args = p.parse_args(argv)

    try:
        characterize(args.config, args.output, verbose=not args.quiet)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
