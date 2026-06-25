import os
from CAMASim.performance.energy import EnergyEval
from CAMASim.performance.latency import LatencyEval

class PerformanceEvaluator:
    def __init__(self, arch_config, array_config, cell_config):
        """
        Initialize the PerformanceEvaluator with architecture and array configurations.
        """
        self.arch_config = arch_config
        self.array_config = array_config
        self.cell_config = cell_config

    def initialize(self, cam_arch):
        """
        Initialize the PerformanceEvaluator with estimated CAM architecture and components.
        """
        self.cam_arch = cam_arch
        
        self.peripheral_cost = {
            "decoder": {"latency": 1.0, "energy": 1.0},
            "encoder": {"latency": 1.0, "energy": 1.0},
            "adder": {"latency": 1.0, "energy": 1.0},
            "register": {"latency": 1.0, "energy": 1.0},
            "mux": {"latency": 1.0, "energy": 1.0}
        }
        self.extract_arch_arrays()
        self.extract_arch_peripherals()
        
        self.latency_eval = LatencyEval(
            self.arch_config,
            self.array_config,
            self.array_cost,
            self.peripheral_cost,
            self.num_array,
            self.num_mat,
            self.num_bank,
            self.array_peripherals,
            self.mat_peripherals,
            self.bank_peripherals,
        )
        self.energy_eval = EnergyEval(
            self.arch_config,
            self.array_config,
            self.array_cost,
            self.peripheral_cost,
            self.num_array,
            self.num_mat,
            self.num_bank,
            self.array_peripherals,
            self.mat_peripherals,
            self.bank_peripherals,
        )

    def write(self, data):
        """
        Calculate the latency and energy consumption for a write operation.
        """
        latency = self.latency_eval.calculate_write_latency()
        energy = self.energy_eval.calculate_write_energy()
        return latency, energy

    def query(self, data):
        """
        Intercepts runtime data matrix sizes and triggers the NeuroSim backend dynamically.
        """
        print("\n Camasim matrix")
        print(f"Data Type: {type(data)}")
        
        # Capture the shape dynamically from the incoming matrix
        data_shape = None
        if hasattr(data, 'shape'):
            print(f"Data Shape: {data.shape}")
            data_shape = data.shape
        else:
            print(f"Raw Data Value: {data}")

        # Pass the dynamic shape into Neurosim
        latency, energy = self._invoke_neurosim_backend(data_shape=data_shape)
        return 0.0, 0.0

    def _invoke_neurosim_backend(self, data_shape=None):
    """
    Streamlined bridge that evaluates hardware performance natively.
    """
    import os
    import subprocess

    # Locate directories 
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = current_file_dir
    while repo_root and os.path.basename(repo_root) != "CAMASim-NueroSim-Hybrid":
        parent = os.path.dirname(repo_root)
        if parent == repo_root: break
        repo_root = parent

    # Point directly to where C++ is compiled
    executable = os.path.join(repo_root, "main")
    neurosim_dir = os.path.join(repo_root, "modules", "NeuroSim", "NeuroSIM")
    dynamic_csv_path = os.path.join(neurosim_dir, "Net.csv")

    # Extract dynamic dimensions and enforce hardware floor limits (128x128)
    workload_rows = max(128, data_shape[0]) if (data_shape and len(data_shape) >= 2) else 128
    workload_cols = max(128, data_shape[1]) if (data_shape and len(data_shape) >= 2) else 128
    
    # Write out the CSV \
    try:
        with open(dynamic_csv_path, 'w', newline='\n') as f:
            f.write("1\n") 
            f.write(f"1,{workload_rows},{workload_cols},1,1,1,1,1\n")
        print(f"[Bridge] Workload Padded to Hardware Minimum: {workload_rows}x{workload_cols}")
    except Exception as csv_err:
        print(f"[Warning] Failed to generate dynamic CSV: {csv_err}")
        return 0.0, 0.0

    # Ensure the binary exists before spawning sub-processes
    if not os.path.exists(executable):
        print(f"[Error] Target binary missing at: {executable}. Run your g++ command first!")
        return 0.0, 0.0

    #  pass the dynamic row/col sizes into the standard NeuroSim positional arguments
    args_str = f"Net.csv 8 8 {workload_rows} {workload_cols} 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8"
    
    # Direct Execution under Native WSL Environment
    cmd = ["sh", "-c", f"cd '{neurosim_dir}' && '{executable}' {args_str}"]

    # Execute binary 
    try:
        print(f"--- [CAMASim-NeuroSim PIM Hardware Evaluation] ---")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("\n--- [Bridge] Intercepted NeuroSim Output ---")
        print(result.stdout.strip())
        print("-------------------------------------------\n")
    except subprocess.CalledProcessError as e:
        print(f"\n[NeuroSim Backend Failed]: Exit Code {e.returncode}\nError: {e.stderr.strip() if e.stderr else e.output.strip()}")
    except Exception as e:
        print(f"\n[Bridge System Error]: {e}")
        
    return 0.0, 0.0
    def extract_arch_arrays(self):
        """
        Decoupled from EvaCAM. Extracts structural constraints safely.
        """
        self.num_array = self.cam_arch["array"]["size"]
        self.num_mat = self.cam_arch["mat"]["size"]
        self.num_bank = self.cam_arch["bank"]["size"]

        # Full mock nested structure to satisfy all internal legacy math equations
        self.array_cost = {
            "write": {"latency": 1.0, "energy": 1.0},
            "query": {"latency": 1.0, "energy": 1.0},
            "search": {"latency": 1.0, "energy": 1.0},
            "interconnect": {"latency": 1.0, "energy": 1.0},
            "peripheral": {"latency": 1.0, "energy": 1.0},
            "cell": {"latency": 1.0, "energy": 1.0}
        }
        print('EvaCAM cost engine successfully bypassed. Routing metrics to NeuroSim backend...\n')

    def extract_arch_peripherals(self):
        """
        Extract the peripherals for arrays, mats, and banks based on architecture configuration.
        """
        self.array_peripherals = self.cam_arch["array"]["peripherals"]
        self.mat_peripherals = self.cam_arch["mat"]["peripherals"]
        self.bank_peripherals = self.cam_arch["bank"]["peripherals"]