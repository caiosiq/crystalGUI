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
    GT_NOISE = 12.0 # High noise to test Texture stage
    
    gt_config.sensor.blur_sigma = GT_BLUR
    gt_config.sensor.bg_noise_std = GT_NOISE
    
    # CRITICAL FIX: Set Aperture > 0 so focus_z actually matters!
    # A pinhole camera (aperture=0) has infinite DoF, making focus_z irrelevant.
    gt_config.sensor.aperture = 0.2 
    
    # Physics (Fixed for this test)
    gt_config.physics.rods.n_rods_rng_lo_hi = (50, 100, 10)
    
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
    init_config.physics.rods.n_rods_rng_lo_hi = (50, 100, 10) # Match physics count
    
    # Start far away
    INIT_BLUR = 8.0 # Start at 0 to test gradient robustness
    INIT_NOISE = 0.0
    
    init_config.sensor.blur_sigma = INIT_BLUR
    init_config.sensor.bg_noise_std = INIT_NOISE
    init_config.sensor.aperture = 0.2 # Match GT Aperture
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
    steps = [l.get('step', i*5) for i, l in enumerate(logs)]
    
    # --- ACADEMIC PLOT SETTINGS ---
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'lines.linewidth': 2.5,
        'figure.figsize': (8, 6),
        'font.family': 'Arial', # Use serif for academic papers
        'axes.grid': True,
        'grid.alpha': 0.3
    })

    # --- Combined Plot (2 Subplots, Shared X) ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={'height_ratios': [1, 1.5]})
    
    # 1. Loss Plot (Top)
    ax1.plot(steps, loss_vals, color='#D32F2F', label='Total Loss', linewidth=2.5)
    ax1.set_ylabel("Loss Value")
    ax1.set_title("Optimization Convergence & Parameter Recovery")
    ax1.legend(loc='upper right', frameon=True)
    ax1.grid(True, alpha=0.3)

    # 2. Parameter Evolution Plot (Bottom)
    colors = ['#1976D2', '#388E3C', '#FBC02D', '#7B1FA2']
    nice_names = {
        'sensor.blur_sigma': r'Blur $\sigma$',
        'sensor.bg_noise_std': r'Noise $\sigma$',
        'sensor.focus_z': r'Focus $z$'
    }

    for idx, param_name in enumerate(params_to_optimize):
        vals = [l['params'].get(param_name, 0.0) for l in logs]
        color = colors[idx % len(colors)]
        display_name = nice_names.get(param_name, param_name.split('.')[-1])
        
        # Plot optimization path
        ax2.plot(steps, vals, label=display_name, color=color, linewidth=2.5)
        
        # Plot GT line
        gt_val = results[param_name]['gt']
        ax2.axhline(y=gt_val, linestyle='--', alpha=0.7, color=color, linewidth=1.5)
        
        # Annotate GT
        y_range = max(max(vals), gt_val) - min(min(vals), gt_val)
        y_offset = y_range * 0.05 if y_range > 0 else 0.5
        ax2.text(steps[-1]*0.95, gt_val + y_offset, f"GT: {gt_val}", 
                horizontalalignment='right', verticalalignment='bottom', 
                color=color, fontsize=12, fontweight='bold')
        
    ax2.set_xlabel("Iteration Step")
    ax2.set_ylabel("Parameter Value")
    ax2.legend(loc='best', frameon=True, fancybox=True)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    combined_path = f"{out_dir}/optimization_combined.png"
    plt.savefig(combined_path, dpi=300)
    print(f"Saved Combined Plot to {combined_path}")
    
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
