
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import yaml
import os
import time
from pathlib import Path
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF

from osog.config import SynthConfig
from diff_calibration.diff_wrapper import DiffOSOG
from diff_calibration.loss.perceptual import VGGPerceptualLoss
from diff_calibration.loss.spectral import SpectralLoss
from diff_calibration.loss.patch import RandomCropLoss
from diff_calibration.loss.balancer import LossBalancer

def load_image(path, device, size=None):
    """Load image as (1, 3, H, W) tensor in range [0, 1]."""
    img = Image.open(path).convert('RGB')
    if size:
        img = img.resize(size, Image.BILINEAR)
    tensor = TF.to_tensor(img).unsqueeze(0).to(device)
    return tensor

def save_image(tensor, path):
    """Save (1, 3, H, W) tensor as image."""
    tensor = tensor.detach().cpu().squeeze(0)
    # Clamp to [0, 1] before saving to avoid weird artifacts
    tensor = torch.clamp(tensor, 0.0, 1.0)
    img = TF.to_pil_image(tensor)
    img.save(path)

def main():
    parser = argparse.ArgumentParser(description="Calibrate OSOG parameters to match a target image.")
    parser.add_argument('--target', type=str, required=True, help="Path to reference image.")
    # parser.add_argument('--config', type=str, default="config_base.yaml", help="Initial config file.")
    parser.add_argument('--out_dir', type=str, default="calibration_output", help="Output directory.")
    parser.add_argument('--steps', type=int, default=200, help="Optimization steps.")
    parser.add_argument('--lr', type=float, default=0.05, help="Learning rate.")
    # Default params to tune: blur, noise, shadow
    parser.add_argument('--params', nargs='+', default=['optics.blur_sigma', 'optics.noise_scale', 'optics.shadow_gain'], 
                        help="List of parameters to tune (dot notation).")
    parser.add_argument('--crop_size', type=int, default=256, help="Patch size for loss calculation.")
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 2. Initialize Model (Before loading target so we know resolution)
    # Load base config or defaults
    config = SynthConfig()
    # Force reasonable resolution for calibration (512x512 is good trade-off)
    config.sensor.resolution = (512, 512)
    
    model = DiffOSOG(config, device=device)
    
    # Register parameters to tune
    print(f"Registering parameters: {args.params}")
    model.register_active_params(args.params)
    
    # 1. Load Target (Resize to match config resolution)
    target_res = (config.sensor.resolution[1], config.sensor.resolution[0]) # W, H
    target_img = load_image(args.target, device, size=target_res)
    print(f"Loaded target: {target_img.shape}")
    
    # Save initial target for reference
    save_image(target_img, f"{args.out_dir}/target.png")
    
    # 3. Setup Optimizer
    # We might want different LRs for different params, but start simple.
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 4. Setup Loss Engine
    # VGG for Texture/Style
    # Resize=False because we handle resolution manually
    vgg = VGGPerceptualLoss(resize=False, use_gram=True).to(device)
    # Spectral for Blur/Noise
    spectral = SpectralLoss(log_scale=True).to(device)
    
    # Patch wrappers
    # Crop size must be smaller than image size
    crop_size = min(args.crop_size, target_res[0], target_res[1])
    vgg_patch = RandomCropLoss(vgg, crop_size=crop_size, num_crops=4) # Increase crops for stability
    spectral_patch = RandomCropLoss(spectral, crop_size=crop_size, num_crops=4)
    
    # Balancer
    # VGG is usually small (~0.1), Spectral is large (~1000 without normalization, ~10 with log).
    # We rely on dynamic scaling to fix this.
    balancer = LossBalancer(['vgg', 'spectral'], weights={'vgg': 1.0, 'spectral': 0.5}, dynamic_scaling=True).to(device)
    
    print("Starting optimization...")
    
    # 5. Optimization Loop
    for step in range(args.steps):
        optimizer.zero_grad()
        
        # Forward Pass
        # DiffOSOG generates (1, 3, H, W) in range [0, 255]
        pred_img_raw = model() 
        
        # CRITICAL: Scale to [0, 1] to match target image loaded by torchvision
        pred_img = pred_img_raw / 255.0
        
        # Compute Losses
        # Note: We pass full images to patch loss, it handles cropping internally
        # pred_img and target_img should be same size [1, 3, 512, 512] and range [0, 1]
        
        loss_vgg = vgg_patch(pred_img, target_img)
        loss_spec = spectral_patch(pred_img, target_img)
        
        losses = {'vgg': loss_vgg, 'spectral': loss_spec}
        total_loss, log_dict = balancer(losses)
        
        # Backward
        total_loss.backward()
        
        # Step
        optimizer.step()
        
        # Clamp params to valid ranges
        with torch.no_grad():
            for name, param in model.named_parameters():
                # Sigma/Scale must be positive
                if 'sigma' in name or 'scale' in name or 'gain' in name or 'density' in name:
                     param.clamp_(min=0.01)
                # Opacity/Reflectivity [0, 1]
                if 'opacity' in name or 'reflectivity' in name or 'roughness' in name:
                     param.clamp_(0.0, 1.0)
                     
        # Logging
        if step % 10 == 0:
            # Print current param values
            params_str = " | ".join([f"{n}: {p.item():.4f}" for n, p in model.named_parameters()])
            
            print(f"Step {step}: Loss {total_loss.item():.4f} | "
                  f"VGG: {log_dict['loss/vgg_weighted']:.4f} (s={log_dict.get('loss/vgg_scale', 1.0):.2f}) | "
                  f"Spec: {log_dict['loss/spectral_weighted']:.4f} (s={log_dict.get('loss/spectral_scale', 1.0):.2f})")
            print(f"Params: {params_str}")
            
        # Save intermediate
        if step % 20 == 0:
            save_image(pred_img, f"{args.out_dir}/step_{step:04d}.png")
            
    # Save final result
    save_image(pred_img, f"{args.out_dir}/final.png")
    print("Optimization done.")
    
    # Export params
    final_params = {n: p.item() for n, p in model.named_parameters()}
    print("Final Calibrated Parameters:")
    print(yaml.dump(final_params))

if __name__ == "__main__":
    main()
