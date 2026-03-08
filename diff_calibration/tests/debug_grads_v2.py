import torch
import sys
import os
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from diff_calibration.src.diff_wrapper import DiffOSOG
from diff_calibration.src.parameter_manager import ParameterManager
from osog.config import SynthConfig

def debug_gradients_detailed():
    print("=== Debugging Gradients: Focus & Noise ===")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Setup Config
    config = SynthConfig()
    config.sensor.resolution = (128, 128)
    config.sensor.blur_sigma = 1.0 # Set valid blur
    config.optics.focus_z = 50.0 # Initial Focus
    config.sensor.bg_noise_std = 0.5 # Initial Noise
    
    # Ensure features are enabled
    config.sensor.dof_enable = True
    config.sensor.aperture = 0.5 # MUST be > 0 for focus_z
    
    # 2. Setup Model & Manager
    model = DiffOSOG(config, device=device)
    param_manager = ParameterManager(model, rules_path="diff_calibration/optimization_rules.json")
    
    # Fix: Set aperture to non-zero so focus_z has an effect!
    print(f"--- Setting Aperture to 0.2 (Fix for Pinhole Camera) ---")
    model.cfg.optics.aperture = 0.2

    # 3. Register Parameters
    print(f"--- Registering Parameters ---")
    params_to_test = ['optics.focus_z', 'sensor.bg_noise_std']
    param_manager.init_parameters(params_to_test)
    
    # 3. Check Gradients
    print("\n--- Testing Gradients ---")
    
    # A. Update Config
    param_manager.update_model_config()
    
    print(f"Config Focus: {model.cfg.optics.focus_z} (Tensor? {torch.is_tensor(model.cfg.optics.focus_z)})")
    print(f"Config Noise: {model.cfg.sensor.bg_noise_std} (Tensor? {torch.is_tensor(model.cfg.sensor.bg_noise_std)})")
    
    # B. Forward
    output = model(seed=42)
    output = output / 255.0
    
    # C. Loss & Backward
    # Use sum() as loss to maximize gradient flow
    loss = output.sum()
    loss.backward()
    
    # D. Report
    for name, p in param_manager.latent_params.items():
        grad = p.grad
        val = p.item()
        # name is safe_name (underscores), convert to real name (dots)
        # But wait, name in loop is key of latent_params, which is safe_name
        # e.g. optics_focus_z
        # We need to map it back or just manually check
        
        # Simple hack: find the key in get_physical_values() that matches
        phys_vals = param_manager.get_physical_values()
        # Try replacing _ with . but be careful about existing underscores
        # Better: iterate phys_vals and match safe_name
        
        real_name = None
        for k in phys_vals.keys():
            if k.replace('.', '_') == name:
                real_name = k
                break
                
        phys_val = phys_vals.get(real_name, -999.0)
        
        print(f"\nParameter: {real_name} ({name})")
        print(f"  Latent Value: {val:.4f}")
        print(f"  Physical Value: {phys_val:.4f}")
        
        if grad is None:
            print("  Gradient: NONE (Disconnected!)")
        elif grad == 0:
            print("  Gradient: ZERO (Connected but flat?)")
        else:
            print(f"  Gradient: {grad:.6f} (Connected)")

    # 4. Sensitivity Analysis (Perturbation)
    print("\n--- Sensitivity Analysis (Finite Differences) ---")
    
    base_loss = model(seed=42).float().mean().item()
    
    for name in params_to_test:
        # Perturb physical value manually
        orig_val = param_manager._get_config_value(model.cfg, name, None)
        
        # Small perturbation
        delta = 1.0 if 'focus' in name else 0.1
        new_val = orig_val + delta
        
        # Set manually (bypassing manager to test DiffOSOG directly)
        if name == 'optics.focus_z':
            model.cfg.optics.focus_z = new_val
        elif name == 'sensor.bg_noise_std':
            model.cfg.sensor.bg_noise_std = new_val
            
        new_loss = model(seed=42).float().mean().item()
        
        diff = new_loss - base_loss
        print(f"{name}:")
        print(f"  Base Loss: {base_loss:.4f}")
        print(f"  Perturbed Loss ({orig_val} -> {new_val}): {new_loss:.4f}")
        print(f"  Difference: {diff:.6f}")
        
        if abs(diff) < 1e-6:
            print("  WARNING: Output did not change! Parameter has no effect.")
        else:
            print("  OK: Output changed.")

if __name__ == "__main__":
    debug_gradients_detailed()
