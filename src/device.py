import torch
import yaml
import os

def get_device():
    """Parses the MNAT config and securely allocates the hardware tensor processor."""
    config_path = "config.yaml"
    
    if not os.path.exists(config_path):
        print("[WARN] config.yaml not found. Defaulting to CUDA if available.")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    # Safely extract from the new 'system' hierarchy
    requested_device = cfg.get("system", {}).get("device", "cuda")
    
    if requested_device == "cuda" and torch.cuda.is_available():
        print("[HARDWARE] CUDA constraint satisfied. Accelerating tensors on GPU.")
        return torch.device("cuda")
    else:
        print("[HARDWARE] WARNING: CUDA unavailable or CPU explicitly requested.")
        print("[HARDWARE] Running MNAT on a CPU will take weeks. Check your drivers.")
        return torch.device("cpu")

# Global device constant imported by all neural scripts
DEVICE = get_device()