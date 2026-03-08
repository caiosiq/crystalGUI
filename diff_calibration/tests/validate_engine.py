import torch
import os
import sys
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from diff_calibration.src.diff_wrapper import DiffOSOG
from diff_calibration.src.calibration_engine import CalibrationEngine
from osog.config import SynthConfig

def validate_engine_full_cycle():
    print("=== Starting Engine Validation: Full Cycle ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = "diff_calibration/validation_output/engine_test"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Ground Truth Generation
    print("--- Generating Ground Truth ---")
    gt_config = SynthConfig()
    gt_config.sensor.resolution = (256, 256)
    
    # GT Values
    GT_BLUR = 2.0
    GT_NOISE = 0.5 # High noise to test Texture stage
    GT_FOCUS = -2.0
    
    gt_config.sensor.blur_sigma = GT_BLUR
    gt_config.sensor.bg_noise_std = GT_NOISE
    gt_config.optics.focus_z = GT_FOCUS
    
    # CRITICAL FIX: Set Aperture > 0 so focus_z actually matters!
    # A pinhole camera (aperture=0) has infinite DoF, making focus_z irrelevant.
    gt_config.optics.aperture = 0.2 
    
    # Physics (Fixed for this test)
    gt_config.physics.rods.n_rods_rng_lo_hi = (5, 10, 10)
    
    gt_model = DiffOSOG(gt_config, device=device)
    
    # Use a fixed seed for GT
    with torch.no_grad():
        target_img_raw = gt_model(seed=42)
        target_img = target_img_raw / 255.0
        save_image(target_img, f"{out_dir}/target_gt.png")
        print(f"Saved GT image to {out_dir}/target_gt.png")

    # 2. Initialization (Bad Parameters)
    print("--- Initializing Calibration Model ---")
    init_config = SynthConfig()
    init_config.sensor.resolution = (256, 256)
    init_config.physics.rods.n_rods_rng_lo_hi = (5, 10, 10) # Match physics count
    
    # Start far away
    INIT_BLUR = 0.0 # Start at 0 to test gradient robustness
    INIT_NOISE = 0.0
    INIT_FOCUS = 2.0
    
    init_config.sensor.blur_sigma = INIT_BLUR
    init_config.sensor.bg_noise_std = INIT_NOISE
    init_config.optics.focus_z = INIT_FOCUS
    init_config.optics.aperture = 0.2 # Match GT Aperture
    # init_config.sensor.dof_enable = True # (Implicitly enabled by non-zero aperture)
    
    model = DiffOSOG(init_config, device=device) # No batching for simpler debug
    
    # Save Initial State
    with torch.no_grad():
        init_img_raw = model(seed=42)
        save_image(init_img_raw / 255.0, f"{out_dir}/initial_state.png")
        print(f"Saved Initial image to {out_dir}/initial_state.png")

    # 3. Setup Engine
    print("--- Setting up Calibration Engine ---")
    # We want to optimize: Blur (Geometry), Focus (Geometry), Noise (Texture)
    params_to_optimize = [
        'sensor.blur_sigma',
        'optics.focus_z',
        'sensor.bg_noise_std'
    ]
    
    engine = CalibrationEngine(
        model=model,
        device=device,
        base_lr=0.1, # Robust LR
        max_steps=100 # Short run for testing (scheduler splits this budget)
    )
    
    # 4. Run Calibration
    print("--- Running Calibration ---")
    
    logs = []
    def callback(step, total, info):
        logs.append(info)
        # Optional: print specific metric
        # print(f"Callback: {info['stage']} | Loss: {info['loss']:.4f}")
        
    final_params = engine.calibrate(target_img, params_to_optimize, progress_callback=callback)
    
    # 5. Analysis
    print("\n=== Validation Results ===")
    print(f"Parameter | GT | Initial | Final | Error")
    print("-" * 50)
    
    results = {
        'sensor.blur_sigma': {'gt': GT_BLUR, 'init': INIT_BLUR},
        'optics.focus_z': {'gt': GT_FOCUS, 'init': INIT_FOCUS},
        'sensor.bg_noise_std': {'gt': GT_NOISE, 'init': INIT_NOISE}
    }
    
    for name, data in results.items():
        final_val = final_params.get(name, -999.0)
        error = abs(final_val - data['gt'])
        print(f"{name:<20} | {data['gt']:<5.2f} | {data['init']:<5.2f} | {final_val:<5.2f} | {error:<5.2f}")

    # 6. Generate Final Image
    with torch.no_grad():
        final_img_raw = model(seed=42) # Check structural match
        save_image(final_img_raw / 255.0, f"{out_dir}/final_result.png")
        print(f"Saved Final image to {out_dir}/final_result.png")
        
    # 7. Plot Loss Curves and Parameter Evolution
    loss_vals = [l['loss'] for l in logs]
    steps = [l.get('step', i*5) for i, l in enumerate(logs)] # Use step from log if available, else infer
    
    # Loss Plot
    plt.figure()
    plt.plot(steps, loss_vals)
    plt.title("Calibration Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.savefig(f"{out_dir}/loss_curve.png")
    print(f"Saved Loss Curve to {out_dir}/loss_curve.png")
    
    # Parameter Evolution Plot
    plt.figure(figsize=(10, 6))
    for param_name in params_to_optimize:
        vals = [l['params'].get(param_name, 0.0) for l in logs]
        plt.plot(steps, vals, label=param_name)
        
        # Plot GT line
        gt_val = results[param_name]['gt']
        plt.axhline(y=gt_val, linestyle='--', alpha=0.5, label=f"{param_name} GT")
        
    plt.title("Parameter Evolution")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig(f"{out_dir}/param_evolution.png")
    print(f"Saved Parameter Evolution to {out_dir}/param_evolution.png")
    
    # 8. Save Parameter History to CSV
    csv_path = f"{out_dir}/param_history.csv"
    with open(csv_path, "w") as f:
        # Header
        headers = ["step", "loss"] + params_to_optimize
        f.write(",".join(headers) + "\n")
        
        # Rows
        for l in logs:
            step = l.get('step', -1)
            loss = l.get('loss', 0.0)
            row = [str(step), f"{loss:.6f}"]
            for pname in params_to_optimize:
                val = l['params'].get(pname, 0.0)
                row.append(f"{val:.6f}")
            f.write(",".join(row) + "\n")
            
    print(f"Saved Parameter History CSV to {csv_path}")

if __name__ == "__main__":
    validate_engine_full_cycle()
