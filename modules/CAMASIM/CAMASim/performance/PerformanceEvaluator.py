import os
import subprocess
import numpy as np
from CAMASim.performance.energy import EnergyEval
from CAMASim.performance.latency import LatencyEval

CELL_TYPE_MAP = {
    "SRAM":  1,
    "RRAM":  2,
    "FeFET": 3,
    "nvCap": 4,
}

class PerformanceEvaluator:
    def __init__(self, arch_config, array_config, cell_config):
        self.arch_config = arch_config
        self.array_config = array_config
        self.cell_config = cell_config

    def initialize(self, cam_arch):
        self.cam_arch = cam_arch
        self.peripheral_cost = {
            "decoder":  {"latency": 1.0, "energy": 1.0},
            "encoder":  {"latency": 1.0, "energy": 1.0},
            "adder":    {"latency": 1.0, "energy": 1.0},
            "register": {"latency": 1.0, "energy": 1.0},
            "mux":      {"latency": 1.0, "energy": 1.0},
        }
        self.extract_arch_arrays()
        self.extract_arch_peripherals()
        self.latency_eval = LatencyEval(
            self.arch_config, self.array_config, self.array_cost,
            self.peripheral_cost, self.num_array, self.num_mat, self.num_bank,
            self.array_peripherals, self.mat_peripherals, self.bank_peripherals,
        )
        self.energy_eval = EnergyEval(
            self.arch_config, self.array_config, self.array_cost,
            self.peripheral_cost, self.num_array, self.num_mat, self.num_bank,
            self.array_peripherals, self.mat_peripherals, self.bank_peripherals,
        )

    def write(self, data):
        latency = self.latency_eval.calculate_write_latency()
        energy  = self.energy_eval.calculate_write_energy()
        return latency, energy

    def query(self, data):
        print("\n[CAMASimBridge] Incoming query data:")
        print(f"  Type : {type(data)}")
        data_shape = None
        if hasattr(data, "shape"):
            print(f"  Shape: {data.shape}")
            data_shape = data.shape
        else:
            print(f"  Value: {data}")
        latency, energy = self._invoke_neurosim_backend(data=data, data_shape=data_shape)
        print(f"[CAMASimBridge] NeuroSim returned → latency={latency:.6e} s, energy={energy:.6e} J")
        return latency, energy

    def _invoke_neurosim_backend(self, data=None, data_shape=None):
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = current_file_dir
        while repo_root:
            if os.path.basename(repo_root) == "CAMASim-Hybrid":
                break
            parent = os.path.dirname(repo_root)
            if parent == repo_root:
                break
            repo_root = parent

        neurosim_dir = os.path.join(repo_root, "modules", "NeuroSim", "NeuroSIM")
        executable   = os.path.join(neurosim_dir, "main")

        if not os.path.exists(executable):
            print(f"[Bridge] ERROR: binary not found at {executable}.")
            return 0.0, 0.0

        cfg_rows = int(self.array_config.get("row", 128))
        cfg_cols = int(self.array_config.get("col", 128))

        if data_shape and len(data_shape) >= 2:
            workload_rows = max(cfg_rows, int(data_shape[0]))
            workload_cols = max(cfg_cols, int(data_shape[1]))
        else:
            workload_rows = cfg_rows
            workload_cols = cfg_cols

        workload_rows = max(32, workload_rows)
        workload_cols = max(32, workload_cols)

        synapse_bit   = int(self.cell_config.get("precision", self.cell_config.get("bit", 1)))
        num_bit_input = int(self.array_config.get("inputBit", 8))
        cam_cell_str  = str(self.cell_config.get("type", "RRAM"))
        memcell_type  = CELL_TYPE_MAP.get(cam_cell_str, 2)

        print(f"[Bridge] CAMASim config → rows={workload_rows}, cols={workload_cols}, "
              f"cell={cam_cell_str}({memcell_type}), synapseBit={synapse_bit}, inputBit={num_bit_input}")

        weight_file = os.path.join(neurosim_dir, "cam_weight.csv")
        input_file  = os.path.join(neurosim_dir, "cam_input.csv")

        try:
            if data is not None and hasattr(data, "shape") and data.size > 0:
                weight_data = (np.asarray(data) != 0).astype(int)
                w = np.zeros((workload_rows, workload_cols), dtype=int)
                r = min(workload_rows, weight_data.shape[0])
                c = min(workload_cols, weight_data.shape[1]) if weight_data.ndim > 1 else 1
                if weight_data.ndim > 1:
                    w[:r, :c] = weight_data[:r, :c]
                else:
                    w[:r, 0] = weight_data[:r]
                weight_data = w
            else:
                weight_data = np.random.randint(0, 2, (workload_rows, workload_cols))

            np.savetxt(weight_file, weight_data, delimiter=",", fmt="%d")

            if data is not None and hasattr(data, "shape"):
                base_input = (np.asarray(data).flatten() != 0).astype(int)
                input_data = np.tile(base_input, (num_bit_input, 1))
                ic = min(workload_rows, input_data.shape[1])
                padded = np.zeros((num_bit_input, workload_rows), dtype=int)
                padded[:, :ic] = input_data[:, :ic]
                input_data = padded
            else:
                input_data = np.random.randint(0, 2, (num_bit_input, workload_rows))

            np.savetxt(input_file, input_data, delimiter=",", fmt="%d")

        except Exception as e:
            print(f"[Bridge] WARNING: could not write weight/input files: {e}")
            return 0.0, 0.0

        argv_str = (
            f"--cam-subarray {weight_file} {input_file} "
            f"{workload_rows} {workload_cols} {num_bit_input} {synapse_bit}"
        )

        env = os.environ.copy()
        env["NEUROSIM_ROWS"]     = str(workload_rows)
        env["NEUROSIM_COLS"]     = str(workload_cols)
        env["NEUROSIM_CELLTYPE"] = str(memcell_type)

        cmd = ["sh", "-c", f"cd '{neurosim_dir}' && '{executable}' {argv_str}"]

        print("─── [CAMASim → NeuroSim] Launching SubArray-level evaluation ───")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
        except subprocess.CalledProcessError as e:
            err_msg = e.stderr.strip() if e.stderr else e.output.strip()
            print(f"[Bridge] NeuroSim exited with code {e.returncode}: {err_msg}")
            return 0.0, 0.0
        except Exception as e:
            print(f"[Bridge] System error: {e}")
            return 0.0, 0.0

        latency = energy = area = 0.0
        for line in result.stdout.splitlines():
            if not line.startswith("SYS_METRIC|"):
                continue
            parts = line.strip().split("|")
            if len(parts) != 3:
                continue
            _, metric, value_str = parts
            try:
                value = float(value_str)
            except ValueError:
                continue
            if metric == "LATENCY":
                latency = value
            elif metric == "ENERGY":
                energy = value
            elif metric == "AREA":
                area = value

        print(f"[Bridge] Parsed → latency={latency:.4e} s, energy={energy:.4e} J, area={area:.4e} m²")
        return latency, energy

    def extract_arch_arrays(self):
        self.num_array = self.cam_arch["array"]["size"]
        self.num_mat   = self.cam_arch["mat"]["size"]
        self.num_bank  = self.cam_arch["bank"]["size"]
        self.array_cost = {
            "write":        {"latency": 1.0, "energy": 1.0},
            "query":        {"latency": 1.0, "energy": 1.0},
            "search":       {"latency": 1.0, "energy": 1.0},
            "interconnect": {"latency": 1.0, "energy": 1.0},
            "peripheral":   {"latency": 1.0, "energy": 1.0},
            "cell":         {"latency": 1.0, "energy": 1.0},
        }
        print("[Bridge] EvaCAM bypassed. Circuit metrics from NeuroSim backend.\n")

    def extract_arch_peripherals(self):
        self.array_peripherals = self.cam_arch["array"]["peripherals"]
        self.mat_peripherals   = self.cam_arch["mat"]["peripherals"]
        self.bank_peripherals  = self.cam_arch["bank"]["peripherals"]
