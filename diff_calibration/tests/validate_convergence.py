
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import yaml
import numpy as np
from pathlib import Path
from PIL import Image

from osog.config import SynthConfig
from diff_calibration.diff_wrapper import DiffOSOG
from diff_calibration.loss.perceptual import VGGPerceptualLoss
from diff_calibration.loss.spectral import SpectralLoss
from diff_calibration.loss.patch import RandomCropLoss
from diff_calibration.loss.balancer import LossBalancer
from diff_calibration.calibrate import save_image

import csv

def validate_twin_study():
    """
    Twin Study: Can we recover known parameters?
    
    1. Generate a TARGET image with specific ground truth parameters (e.g., blur=2.0, noise=0.1).
    2. Initialize the optimizer with DIFFERENT parameters (e.g., blur=5.0, noise=0.01).
    3. Run the optimization loop.
    4. Check if the parameters converge back to the ground truth.
    """
    print("Starting Twin Study Validation...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = "diff_calibration/validation_output"
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "convergence_log.csv")
    
    # --- 1. Ground Truth Setup ---
    gt_config = SynthConfig()   
    gt_config.sensor.resolution = (256, 256) # Keep small for speed
    
    # CRITICAL: Reduce rod count for small resolution!
    # Default is (150, 600), which saturates a 256x256 image in Blaze mode.
    gt_config.physics.rods.n_rods_rng_lo_hi = (5, 10, 10)
    
    # Set Ground Truth Values
    GT_BLUR = 2.0
    GT_NOISE = 0.1
    # Shadow gain is tricky because it might be a tuple in config but we optimize scalar
    # Let's stick to scalar params for this test first to ensure basic loop works.
    
    gt_config.optics.blur_sigma = 0.0 # Unused in Blaze, use Sensor Blur
    gt_config.sensor.blur_sigma = GT_BLUR
    gt_config.optics.noise_scale = GT_NOISE
    gt_config.optics.mode = "blaze" # Enable Blaze for Physics Noise
    
    # Generate Target Image (using DiffOSOG just to be consistent, but could be standard pipeline)
    print(f"Generating Ground Truth Target (Blur={GT_BLUR}, Noise={GT_NOISE})...")
    gt_model = DiffOSOG(gt_config, device=device)
    
    # We need to freeze randomness for GT generation to be "The Target"
    # But DiffOSOG generates random particles every forward pass unless we fix seed.
    # CRITICAL: For calibration to work, we are trying to match STYLE (blur/noise), 
    # NOT exact particle positions.
    # So Target and Prediction will have DIFFERENT particles.
    # This is why we need Statistical Losses (Spectral/Gram), not Pixel MSE.
    
    # CRITICAL: DiffOSOG outputs [0, 255]. Torchvision image is [0, 1].
    # We must normalize GT to [0, 1] for loss calculation consistency later.
    with torch.no_grad():
        target_img_raw = gt_model() # (1, 3, 256, 256) [0, 255]
        target_img = target_img_raw / 255.0
        
    # Save target (save_image expects [0, 1])
    save_image(target_img, f"{out_dir}/target_gt.png")
    
    # --- 2. Optimizer Setup ---
    # Start far away from GT
    init_config = SynthConfig()
    init_config.sensor.resolution = (256, 256)
    init_config.physics.rods.n_rods_rng_lo_hi = (5, 10, 10) # Match GT density
    
    INIT_BLUR = 5.0 # Start very blurry
    INIT_NOISE = 0.01 # Start very clean
    
    init_config.optics.blur_sigma = 0.0
    init_config.sensor.blur_sigma = INIT_BLUR
    init_config.optics.noise_scale = INIT_NOISE
    init_config.optics.mode = "blaze"
    
    model = DiffOSOG(init_config, device=device)
    
    # Register params to tune
    # Tune Sensor Blur and Physics Noise
    # Note: 'blur_sigma' maps to 'sensor.blur_sigma' in DiffOSOG
    #       'optics.noise_scale' maps to 'optics.noise_scale'
    params_to_tune = ['blur_sigma', 'optics.noise_scale']
    model.register_active_params(params_to_tune)
    
    print(f"Initialized Model with: Blur={INIT_BLUR}, Noise={INIT_NOISE}")
    
    # --- 3. Loss & Optimizer ---
    # Learning rates: Blur needs larger steps, Noise needs smaller?
    optimizer = optim.Adam(model.parameters(), lr=0.01) 
    
    # Use full loss engine
    # Resize=False because we are already at 256x256
    vgg = VGGPerceptualLoss(resize=False).to(device)
    spectral = SpectralLoss(log_scale=True).to(device)
    
    # Balancer
    balancer = LossBalancer(['vgg', 'spectral'], weights={'vgg': 1.0, 'spectral': 0.5}, dynamic_scaling=True).to(device)
    
    # Track history
    history = {'blur': [], 'noise': [], 'loss': []}
    
    # Setup CSV logging
    print(f"Logging convergence data to {csv_path}...")
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['step', 'total_loss', 'vgg_loss', 'spectral_loss', 'blur_sigma', 'noise_scale']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # --- 4. Optimization Loop ---
        STEPS = 300
        print(f"Optimizing for {STEPS} steps...")
        print("Parameter names in model:", [n for n, p in model.named_parameters()])
        for step in range(STEPS):
            optimizer.zero_grad()
            # CRITICAL: Force the exact same rod layout every single step
            # By resetting the seed, the physics engine generates the same geometry,
            # allowing the optimizer to focus PURELY on the optics.
            torch.manual_seed(42) 
            np.random.seed(42)
            # Forward pass generates NEW particles every time
            pred_img_raw = model()
            
            # CRITICAL: Normalize to [0, 1] for loss
            pred_img = pred_img_raw / 255.0
            
            # Loss
            l_vgg = vgg(pred_img, target_img)
            l_spec = spectral(pred_img, target_img)
            
            loss, log_dict = balancer({'vgg': l_vgg, 'spectral': l_spec})
            
            # --- DEBUG: CHECK GRADIENT FLOW ---
            if step == 0:
                print(f"Pred Img requires_grad: {pred_img.requires_grad}")
                print(f"VGG Loss requires_grad: {l_vgg.requires_grad}")
                print(f"Total Loss requires_grad: {loss.requires_grad}")
            # ----------------------------------
            
            loss.backward()
            optimizer.step()
            
            # Clamp
            with torch.no_grad():
                for name, param in model.named_parameters():
                    param.clamp_(min=0.001)
            
            # Log
            current_params = {n: p.item() for n, p in model.named_parameters()}
            history['loss'].append(loss.item())
            
            # Extract specific param values
            # DiffOSOG uses underscores for parameter names
            # 'blur_sigma' -> 'sensor_blur_sigma' (Wait, DiffOSOG replaces dots with underscores)
            # 'blur_sigma' has no dots -> 'blur_sigma'
            # 'optics.noise_scale' -> 'optics_noise_scale'
            curr_blur = current_params.get('blur_sigma', 0.0)
            curr_noise = current_params.get('optics_noise_scale', 0.0)
            
            history['blur'].append(curr_blur)
            history['noise'].append(curr_noise)
            
            # Write to CSV
            writer.writerow({
                'step': step,
                'total_loss': loss.item(),
                'vgg_loss': l_vgg.item(),
                'spectral_loss': l_spec.item(),
                'blur_sigma': curr_blur,
                'noise_scale': curr_noise
            })
            if step % 5 == 0:
                csvfile.flush()
            
            if step % 10 == 0:
                print(f"Step {step}: Loss={loss.item():.4f} | Blur={curr_blur:.3f} (GT={GT_BLUR}) | Noise={curr_noise:.3f} (GT={GT_NOISE})")
            
    # --- 5. Analysis ---
    print("\n--- Validation Results ---")
    final_blur = history['blur'][-1]
    final_noise = history['noise'][-1]
    
    blur_err = abs(final_blur - GT_BLUR)
    noise_err = abs(final_noise - GT_NOISE)
    
    print(f"Final Blur: {final_blur:.3f} (Error: {blur_err:.3f})")
    print(f"Final Noise: {final_noise:.3f} (Error: {noise_err:.3f})")
    
    # Criteria: Blur within 0.5, Noise within 0.05
    if blur_err < 0.5 and noise_err < 0.05:
        print("SUCCESS: Parameters converged reasonably close to Ground Truth.")
    else:
        print("WARNING: Convergence not fully achieved. Check learning rates or loss weights.")
        
    # --- 6. Generate Final Image ---
    print("Generating Final Optimized Image...")
    # Use the optimized parameters to generate a final sample
    # Note: We must be careful not to use gradient tracking here if we want to save memory/speed
    with torch.no_grad():
        final_img_raw = model()
        final_img = final_img_raw / 255.0
        save_image(final_img, f"{out_dir}/final_optimized.png")
        print(f"Saved final optimized image to {out_dir}/final_optimized.png")
        
    # Plotting (Optional, saves to file)
    try:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.plot(history['blur'], label='Est Blur')
        plt.axhline(y=GT_BLUR, color='r', linestyle='--', label='GT Blur')
        plt.legend()
        plt.title('Blur Convergence')
        
        plt.subplot(1, 3, 2)
        plt.plot(history['noise'], label='Est Noise')
        plt.axhline(y=GT_NOISE, color='r', linestyle='--', label='GT Noise')
        plt.legend()
        plt.title('Noise Convergence')
        
        plt.subplot(1, 3, 3)
        plt.plot(history['loss'], label='Total Loss')
        plt.legend()
        plt.title('Loss Curve')
        
        plt.tight_layout()
        plt.savefig(f"{out_dir}/convergence_plot.png")
        print(f"Plot saved to {out_dir}/convergence_plot.png")
    except Exception as e:
        print(f"Plotting failed (no display?): {e}")

if __name__ == "__main__":
    validate_twin_study()
