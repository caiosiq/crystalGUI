import torch
import torch.nn.functional as F
import math
import random
import numpy as np
from typing import Optional, List, Tuple
from ..config import SynthConfig
from .utils.noise import generate_anisotropic_noise_2d

# Reuse the same PIL check/logic for the overlay part which happens on CPU
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False
import cv2

class SensorHeadTorch:
    def __init__(self, config: SynthConfig, device: str = "cpu"):
        self.cfg = config
        self.device = torch.device(device)

    def _gaussian_blur_2d(self, img: torch.Tensor, sigma: float) -> torch.Tensor:
        """
        Apply Gaussian blur to a (C, H, W) or (H, W) tensor.
        """
        # Handle tensor sigma
        # If sigma is tensor, we keep it as tensor for kernel calculation
        # But we need a float for k_size calculation (which is discrete)
        
        if torch.is_tensor(sigma):
             sigma_float = sigma.item() # Detaches for size calculation
             sigma_val = sigma # Keeps gradient for kernel calculation
        else:
             sigma_float = float(sigma)
             sigma_val = sigma_float
             
        if sigma_float <= 0:
             # Ensure sigma_val is treated as a tensor if possible
             if torch.is_tensor(sigma_val):
                 return img + 0.0 * sigma_val.sum()
             else:
                 return img
             
        if img.dim() == 2:
            img = img.unsqueeze(0) # (1, H, W)
            squeeze = True
        else:
            squeeze = False
            
        C, H, W = img.shape
        
        k_ideal = int(round(3 * sigma_float)) * 2 + 1
        max_k = min(H, W)
        if max_k % 2 == 0:
            max_k -= 1
        k_size = max(1, min(k_ideal, max_k))
        
        if k_size == 1:
            return img.squeeze(0) if squeeze else img

        pad = k_size // 2
        x = torch.arange(k_size, device=img.device, dtype=img.dtype) - pad
        
        # Use sigma_val (potentially tensor) here for gradients
        # Avoid float cast if tensor!
        kernel = torch.exp(-0.5 * (x / sigma_val) ** 2)
        kernel = kernel / (kernel.sum() + 1e-6)
        kernel = kernel.view(1, 1, -1)
        
        # Blur Rows
        # img: (C, H, W) -> view as (1, C*H, W) for conv1d
        inp_rows = img.view(1, C * H, W)
        k_rows = kernel.repeat(C * H, 1, 1) # (C*H, 1, K)
        out_rows = F.conv1d(inp_rows, k_rows, padding=pad, groups=C * H)
        out_rows = out_rows.view(C, H, W)
        
        # Blur Cols
        inp_cols = out_rows.view(C, H, W).transpose(1, 2).reshape(1, C * W, H)
        k_cols = kernel.repeat(C * W, 1, 1)
        out_cols = F.conv1d(inp_cols, k_cols, padding=pad, groups=C * W)
        
        out = out_cols.view(C, W, H).transpose(1, 2)
        
        return out.squeeze(0) if squeeze else out

    def _disk_blur_2d(self, img: torch.Tensor, sigma: float) -> torch.Tensor:
        """
        Applies a Disk Kernel (Bokeh) blur to simulate out-of-focus optics.
        Unlike Gaussian, this has a hard edge and uniform interior.
        """
        sigma_val = sigma
        if torch.is_tensor(sigma):
            sigma_val = sigma.item()
            
        if sigma_val <= 0.5:
            return img
            
        if img.dim() == 2:
            img = img.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
            
        C, H, W = img.shape
        
        # Kernel Radius
        r = int(math.ceil(sigma_val))
        k_size = 2 * r + 1
        
        # Generate Disk Kernel
        y, x = torch.meshgrid(
            torch.arange(k_size, device=self.device) - r,
            torch.arange(k_size, device=self.device) - r,
            indexing='ij'
        )
        dist_sq = x**2 + y**2
        
        # Soft Edge for Gradients (Sigmoid) instead of hard comparison
        # Hard: kernel = (dist_sq <= sigma**2).float()
        # Soft: Sigmoid around the edge.
        # Edge at dist = sigma.
        # dist_sq = dist^2.
        # We want 1 inside, 0 outside.
        # sigmoid( (sigma - dist) * temp )
        # dist = sqrt(dist_sq)
        
        # Use sigma (tensor) for gradients
        # dist is sqrt(x^2 + y^2). 
        # kernel = sigmoid((sigma - dist) * temp)
        
        # We assume sigma is float or tensor.
        # But wait, sigma_val was item() in original code which broke graph.
        
        # Re-check sigma logic
        if torch.is_tensor(sigma):
             sigma_use = sigma
        else:
             sigma_use = float(sigma)
             
        dist = torch.sqrt(dist_sq)
        
        # Temperature for sharpness. Higher = sharper edge.
        temp = 5.0
        kernel = torch.sigmoid((sigma_use - dist) * temp)
        
        # Normalize
        kernel = kernel / (kernel.sum() + 1e-6)
        kernel = kernel.view(1, 1, k_size, k_size)
        
        # Convolve
        pad = r
        out = F.conv2d(img.unsqueeze(0), kernel.repeat(C, 1, 1, 1), padding=pad, groups=C).squeeze(0)
        
        return out.squeeze(0) if squeeze else out

    def generate_soup_layer(self, h: int, w: int, count_range: Tuple[int, int], blur_sigma: float, opacity: float, rng: random.Random) -> torch.Tensor:
        """
        Generates a 'soup' of out-of-focus background particles using Procedural Anisotropic Noise.
        Replaces discrete object rendering with a continuous texture field.
        Returns: (1, H, W) tensor, values 0-1 (intensity/density).
        """
        opacity_val = opacity
        if torch.is_tensor(opacity):
            opacity_val = opacity.item()
            
        if opacity_val <= 0.001:
            # We return a tensor connected to opacity if possible, but if it's ~0, 
            # maybe just return zero with grad connection?
            # return torch.zeros(...) * opacity
            return torch.zeros(1, h, w, device=self.device) * opacity
            
        # 1. Map Parameters to Noise Space
        # Count -> Density Threshold
        # High count = Lower threshold (more visible noise)
        # Low count = Higher threshold (sparse noise)
        count = rng.randint(*count_range)
        if count <= 0:
            return torch.zeros(1, h, w, device=self.device)
            
        # Heuristic: Map count 0-200 to Threshold 2.0 -> 0.5
        # Noise is roughly N(0, 1). Values > 2.0 are rare. Values > 0.0 are 50%.
        # We want sparse background usually.
        # Let's say count=200 -> threshold=0.5. count=50 -> threshold=1.5.
        # Linear map: T = 1.8 - (count / 200.0)
        threshold = max(0.2, 1.8 - (count / 250.0))
        
        # Blur Sigma -> Frequency Scale
        # High blur = Low frequency (large blobs)
        # Low blur = High frequency (small grain)
        # scale ~ 1 / sigma
        # Let's say sigma=2.0 -> scale=4.0. sigma=5.0 -> scale=1.5.
        base_scale = 8.0 / (blur_sigma + 1.0)
        
        # Anisotropy (Flow Direction & Stretch)
        angle = 0.0
        stretch = 1.0
        
        # 1. Direction from Physics Flow
        if hasattr(self.cfg, 'physics') and self.cfg.physics.flow_enable:
            angle = self.cfg.physics.flow_direction
            
        # 2. Stretch from Sensor Config (UI Control)
        # Map 0.0 -> 1.0 (Isotropic)
        # Map 1.0 -> 8.0 (Highly Stretched)
        aniso = getattr(self.cfg.sensor, 'distractor_anisotropy', 0.0)
        stretch = 1.0 + aniso * 7.0
        
        # Legacy fallback: If flow is enabled but anisotropy is 0, maybe user expects some stretch?
        # But we now have explicit control. Let's stick to explicit control.
        # If user enables flow but leaves stretch at 0, soup is isotropic (no streaks).
        
        # 2. Generate Anisotropic Noise
        # scale_x (along flow) should be lower frequency (stretched) -> multiply scale by 1/stretch?
        # In my implementation: freq_x = scale_x.
        # To stretch along X, we want low frequency in X.
        # So scale_x = base_scale / stretch
        # scale_y = base_scale
        
        noise = generate_anisotropic_noise_2d(
            shape=(1, 1, h, w),
            scale_x=base_scale / stretch,
            scale_y=base_scale,
            angle_deg=angle,
            octaves=3,
            persistence=0.6,
            lacunarity=2.0,
            device=self.device,
            seed=rng.randint(0, 2**31 - 1)
        ).squeeze(0) # (1, H, W)
        
        # 3. Apply Threshold (Density)
        # Relu to keep only positive peaks
        soup = F.relu(noise - threshold)
        
        # Normalize peak to 1.0 (if any signal exists)
        if soup.max() > 1e-6:
            soup = soup / soup.max()
            
        # 4. Apply Optical Bokeh Blur
        # Even though noise is continuous, we apply the disk kernel 
        # to give it the characteristic optical "defocus" shape (hard edges on soft blobs)
        # But for "Deep Soup" (Zone 3), we might want it very soft.
        # The user said: "The overlapping disks completely destroy the local geometry."
        # So applying a large disk blur to the noise field is physically correct.
        
        # blur_sigma might be a tensor.
        blur_val = blur_sigma
        if torch.is_tensor(blur_sigma):
            blur_val = blur_sigma.item()
            
        if blur_val > 0.5:
             soup = self._disk_blur_2d(soup, blur_sigma)
             
        # 5. Final Opacity
        return soup * opacity

    def apply_background(self, h: int, w: int, rng: random.Random, seed: int) -> torch.Tensor:
        """
        Generates background directly on GPU.
        Returns: (3, H, W) tensor, values 0-255 float (not uint8 yet).
        """
        cfg = self.cfg
        dev = self.device
        
        # We use a torch generator for reproducibility if needed, or just random calls
        gen = torch.Generator(device=dev)
        gen.manual_seed(seed)
        
        # Base gray
        gmin, gmax = cfg.sensor.bg_gray_range
        
        # Override for Blaze: Must be DARK
        if cfg.optics.mode == "blaze":
             gmin, gmax = 0, 5
        
        # Override for Brightfield: Must be WHITE (or near white)
        # 255 is white
        # if cfg.optics.mode == "brightfield":
        #     gmin = 245
        #     gmax = 255
            
        # (1, H, W)
        base = torch.randint(gmin, gmax + 1, (1, h, w), device=dev, dtype=torch.float32, generator=gen)
        img = base.repeat(3, 1, 1) # (3, H, W)
        
        # Phase 4.4.2.1: Solvent Tint
        if hasattr(cfg.sensor, 'solvent_color'):
            # Solvent color is RGB tuple. We need to apply it to BGR tensor.
            # Convert RGB -> BGR
            c = cfg.sensor.solvent_color
            tint_bgr = [c[2], c[1], c[0]]
            tint = torch.tensor(tint_bgr, device=dev, dtype=torch.float32).view(3, 1, 1) / 255.0
            img = img * tint
        else:
            print("SensorConfig has no solvent_color")
        
        # Directional gradient (tilt)
        if cfg.sensor.tilt_enable:
            umin, umax = cfg.sensor.tilt_dir_deg
            ang = math.radians(rng.uniform(umin, umax))
            ux, uy = math.cos(ang), math.sin(ang)
            
            cx_shift = max(-0.5, min(0.5, float(cfg.sensor.tilt_center[0])))
            cy_shift = max(-0.5, min(0.5, float(cfg.sensor.tilt_center[1])))
            
            # Grids
            # linspace equivalent
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(-1.0 - 2 * cy_shift, 1.0 - 2 * cy_shift, h, device=dev),
                torch.linspace(-1.0 - 2 * cx_shift, 1.0 - 2 * cx_shift, w, device=dev),
                indexing='ij'
            )
            
            ramp = uy * grid_y + ux * grid_x
            max_val = torch.max(torch.abs(ramp))
            ramp = ramp / (max_val + 1e-6)
            
            ptp = rng.uniform(*cfg.sensor.tilt_ptp)
            img = img + 0.5 * ptp * ramp.unsqueeze(0)

        # Low-frequency illumination
        illum_ampl = cfg.sensor.illum_ampl
        is_illum_active = False
        if torch.is_tensor(illum_ampl):
             if illum_ampl > 0: is_illum_active = True
        elif illum_ampl > 0:
             is_illum_active = True
             
        if is_illum_active:
            sigma = cfg.sensor.illum_sigma
            # Check if sigma is tensor, extract value for heuristic logic (upsample vs blur)
            # But we must use tensor for actual blur calculation if it flows there
            # Here sigma is used for logic AND calculation.
            # If sigma is optimized, this `if sigma > 10` branch logic is non-differentiable switch.
            # But usually we don't optimize sigma here, mostly amplitude.
            
            sigma_val = sigma
            if torch.is_tensor(sigma): sigma_val = sigma.item()
            
            if sigma_val > 10:
                # Optimization: Generate small noise and upsample
                # This simulates "large blur" without the heavy convolution
                scale = 1.0 / (sigma_val * 0.5) # Heuristic scaling
                sh, sw = max(4, int(h * scale)), max(4, int(w * scale))
                
                # Generate small noise
                noise_small = torch.randn(1, 1, sh, sw, device=dev, generator=gen)
                
                # Upsample to full size (Bilinear gives smooth "blurred" look)
                noise = F.interpolate(noise_small, size=(h, w), mode='bilinear', align_corners=False).squeeze(0)
            else:
                # Standard path for small blurs (grain)
                noise = torch.randn(1, h, w, device=dev, generator=gen)
                if sigma_val > 0:
                    # Pass original sigma (tensor)
                    noise = self._gaussian_blur_2d(noise, sigma)
            
            rngv = noise.max() - noise.min()
            if rngv > 1e-6:
                noise = (noise - noise.min()) / (rngv + 1e-6)
                noise = (noise - 0.5) * 2.0
                # Remove float cast
                img = img + illum_ampl * noise
                
        # Vignette
        # Handle tensor vignette_strength
        vig_str = cfg.sensor.vignette_strength
        is_vig_active = False
        if torch.is_tensor(vig_str):
             if vig_str > 0: is_vig_active = True
        elif vig_str > 0:
             is_vig_active = True
             
        if is_vig_active:
            grid_y, grid_x = torch.meshgrid(
                torch.arange(h, device=dev, dtype=torch.float32),
                torch.arange(w, device=dev, dtype=torch.float32),
                indexing='ij'
            )
            cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
            r = torch.sqrt((grid_x - cx) ** 2 + (grid_y - cy) ** 2)
            r = r / (r.max() + 1e-6)
            
            # Remove float() cast to allow gradients
            vig = (1.0 - vig_str * (r ** 2))
            img = img * vig.unsqueeze(0)

        # DIC relief field
        if cfg.sensor.relief_field_enable:
            H_field = torch.randn(1, h, w, device=dev, generator=gen)
            sig = rng.uniform(*cfg.sensor.relief_field_sigma_px)
            
            H_field = self._gaussian_blur_2d(H_field, sig)
            
            phi = math.radians(rng.uniform(*cfg.sensor.relief_field_dir_deg))
            
            # Sobel
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=dev, dtype=torch.float32).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=dev, dtype=torch.float32).view(1, 1, 3, 3)
            
            # Padding for same size
            gx = F.conv2d(H_field.unsqueeze(0), sobel_x, padding=1)
            gy = F.conv2d(H_field.unsqueeze(0), sobel_y, padding=1)
            
            gx = gx.squeeze(0)
            gy = gy.squeeze(0)
            
            S = math.cos(phi) * gx + math.sin(phi) * gy
            S = S / (S.std() + 1e-6)
            
            if cfg.sensor.relief_field_extra_blur > 0:
                S = self._gaussian_blur_2d(S, cfg.sensor.relief_field_extra_blur)
                
            gain = rng.uniform(*cfg.sensor.relief_field_gain)
            # Remove float() cast
            img = img + gain * S
            
        # Background noise
        # Handle tensor bg_noise_std
        bg_noise = cfg.sensor.bg_noise_std
        is_noise_active = False
        if torch.is_tensor(bg_noise):
             if bg_noise > 0: is_noise_active = True
        elif bg_noise > 0:
             is_noise_active = True
             
        if is_noise_active:
            # Use torch.randn and multiply by tensor directly
            noise = bg_noise * torch.randn(1, h, w, device=dev, generator=gen)
            img = img + noise
            
        img = torch.clamp(img, 0, 255)
        
        # Expand to 3 channels (BGR) - ALREADY EXPANDED
        # img = img.repeat(3, 1, 1)
        return img

    def apply_fouling(self, img: torch.Tensor, rng: random.Random) -> torch.Tensor:
        """
        Apply Lens Fouling (Dirt/Biofilm) on top of the image.
        Physically this should be on the lens/sensor surface, so it occludes everything (bg + particles).
        """
        cfg = self.cfg
        if not cfg.sensor.fouling_enable:
            return img
            
        dev = img.device
        C, h, w = img.shape
        
        # We use a torch generator for reproducibility if needed
        # But we rely on passed rng for decisions, and generate tensors randomly
        # Let's make a local generator seeded from rng for tensor ops
        gen = torch.Generator(device=dev)
        gen.manual_seed(rng.randint(0, 2**31 - 1))
        
        # Fouling (Lens Dirt) - Discrete Blobs
        if rng.random() < cfg.sensor.fouling_prob:
            # Generate static blobs
            n_blobs = rng.randint(*cfg.sensor.fouling_count_range)
            if n_blobs > 0:
                fx = torch.randint(0, w, (n_blobs,), device=dev)
                fy = torch.randint(0, h, (n_blobs,), device=dev)
                
                foul_mask = torch.zeros(1, h, w, device=dev)
                foul_mask[0, fy, fx] = 1.0
                
                # Random sigma per frame or per blob? 
                # Original code used single frame_sigma for the whole mask blur
                frame_sigma = rng.uniform(*cfg.sensor.fouling_sigma_range)
                foul_mask = self._gaussian_blur_2d(foul_mask, frame_sigma)
                
                max_val = foul_mask.max()
                if max_val > 1e-6:
                    foul_mask = foul_mask / max_val
                
                strength = cfg.sensor.fouling_opacity
                img = img * (1.0 - strength * foul_mask)

        # Biofilm / Residue (Low-frequency overlay)
        # CHANGED: Now respects fouling_prob instead of hardcoded 0.4
        if rng.random() < cfg.sensor.fouling_prob: 
             # Generate low freq noise
             scale = 1.0 / 32.0
             sh, sw = max(4, int(h * scale)), max(4, int(w * scale))
             noise_bio = torch.randn(1, 1, sh, sw, device=dev, generator=gen)
             noise_bio = F.interpolate(noise_bio, size=(h, w), mode='bilinear', align_corners=False).squeeze(0)
             
             noise_bio = (noise_bio - noise_bio.min()) / (noise_bio.max() - noise_bio.min() + 1e-6)
             
             # Threshold to create "patches"
             bio_mask = torch.clamp((noise_bio - 0.5) * 3.0, 0.0, 1.0)
             
             # Apply texture to these patches
             tex = torch.randn(1, h, w, device=dev, generator=gen) * 0.1
             
             strength = 0.15 * cfg.sensor.fouling_opacity
             img = img * (1.0 - strength * bio_mask * (1.0 + tex))
             
        return img


    def apply_blur(self, img: torch.Tensor, sigma_override: Optional[float] = None) -> torch.Tensor:
        """
        img: (3, H, W)
        """
        # CRITICAL: Always read dynamic config value unless override is provided.
        # This allows injected tensors (with gradients) to flow through.
        # Do NOT assume self.cfg.sensor.blur_sigma is static.
        
        sigma = sigma_override if sigma_override is not None else self.cfg.sensor.blur_sigma
        
        # Check if sigma is a tensor (for gradients) or scalar
        # If scalar, check if > 0
        # If tensor, we assume it's valid and pass it down
        
        if torch.is_tensor(sigma):
             # Pass tensor directly to gaussian_blur_2d
             return self._gaussian_blur_2d(img, sigma)
        elif sigma > 0:
             return self._gaussian_blur_2d(img, sigma)
        
        return img

    def apply_dof(self, img: torch.Tensor, depth_map: torch.Tensor, focus_z: float, aperture: float) -> torch.Tensor:
        """
        Apply Shallow Depth of Field (Zone 2) using a simplified Circle of Confusion (CoC) model.
        """
        # Handle tensor inputs for focus_z and aperture
        aperture_val = aperture.item() if torch.is_tensor(aperture) else aperture
        
        if aperture_val <= 0.001:
            # Ensure aperture is treated as a tensor if possible
            if torch.is_tensor(aperture):
                return img + 0.0 * aperture.sum()
            else:
                return img
            
        # 1. Calculate CoC (Circle of Confusion)
        # CoC ~ |z - focus_z| * aperture
        # Use Tensors for gradient!
        coc = torch.abs(depth_map - focus_z) * aperture * 0.1
        
        # 2. Layered Approach (Trilinear Interpolation approximation)
        # Layer 0: Sharp (CoC < 0.5)
        # Layer 1: Medium Blur (CoC ~ 3.0)
        # Layer 2: Strong Blur (CoC ~ 8.0)
        
        # Define blur levels
        sigma_med = 3.0
        sigma_high = 8.0
        
        # Generate blurred layers (Use Disk Blur for Bokeh)
        img_med = self._disk_blur_2d(img, sigma_med)
        img_high = self._disk_blur_2d(img, sigma_high)
        
        # 3. Blending Weights
        # We want smooth transition.
        
        # CoC 0 -> 3
        t1 = torch.clamp((coc - 0.5) / (3.0 - 0.5), 0.0, 1.0)
        
        # CoC 3 -> 8
        t2 = torch.clamp((coc - 3.0) / (8.0 - 3.0), 0.0, 1.0)
        
        # Blend:
        # If t1=0, result = Sharp
        # If t1=1, result = Med
        # If t2=1, result = High
        
        out = img * (1.0 - t1) + img_med * t1 * (1.0 - t2) + img_high * t2
        
        return out

    def apply_chromatic_aberration(self, img: torch.Tensor, strength: float = 0.0) -> torch.Tensor:
        """
        Simulate lateral chromatic aberration (color fringing).
        """
        # Handle tensor
        s_val = strength.item() if torch.is_tensor(strength) else strength
        
        if s_val <= 0.001:
            # Ensure strength is treated as a tensor if possible
            if torch.is_tensor(strength):
                return img + 0.0 * strength.sum()
            else:
                return img
            
        C, H, W = img.shape
        dev = img.device
        
        # Grid (normalized -1 to 1)
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=dev),
            torch.linspace(-1, 1, W, device=dev),
            indexing='ij'
        )
        
        # Stack: (1, H, W, 2) for grid_sample
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        
        # Use strength (tensor) for scaling factors to preserve gradient
        k = strength * 2.0 / max(H, W)
        scale_b = 1.0 + k # Blue (0)
        scale_r = 1.0 - k # Red (2)
        
        # grid_sample doesn't support gradients w.r.t grid values well in all versions?
        # Actually it does. The grid coordinates depend on 'strength'.
        
        grid_b = base_grid / scale_b
        grid_r = base_grid / scale_r
        
        # BGR
        b = img[0].view(1, 1, H, W)
        g = img[1].view(1, 1, H, W)
        r = img[2].view(1, 1, H, W)
        
        # Resample
        b_new = F.grid_sample(b, grid_b, align_corners=False, padding_mode='reflection')
        r_new = F.grid_sample(r, grid_r, align_corners=False, padding_mode='reflection')
        
        # Combine
        out = torch.stack([b_new.squeeze(), g.squeeze(), r_new.squeeze()], dim=0)
        
        return out

    def _generate_star_kernel(self, k_size: int, n_spikes: int, angle_deg: float) -> torch.Tensor:
        """
        Generates a star-shaped PSF kernel.
        """
        # Ensure odd kernel size
        if k_size % 2 == 0: k_size += 1
        
        # Center
        c = k_size // 2
        
        # Grid
        y, x = torch.meshgrid(
            torch.arange(k_size, device=self.device) - c,
            torch.arange(k_size, device=self.device) - c,
            indexing='ij'
        )
        x = x.float()
        y = y.float()
        
        # Radial coordinates
        r = torch.sqrt(x**2 + y**2)
        
        # Kernel accumulator
        kernel = torch.zeros_like(r)
        
        base_angle = math.radians(angle_deg)
        
        for i in range(n_spikes):
            # Distribute spikes
            alpha = base_angle + i * (math.pi / n_spikes) * 2.0
            
            # Distance to line passing through center
            d = torch.abs(-math.sin(alpha) * x + math.cos(alpha) * y)
            
            # Spike profile across width: Gaussian (Sharp)
            spike_w = torch.exp(-0.5 * (d / 1.0)**2) 
            
            # Spike profile along length: Lorentzian-ish decay
            spike_l = 1.0 / (1.0 + 0.05 * r**2)
            
            kernel += spike_w * spike_l
            
        # Normalize
        kernel = kernel / (kernel.sum() + 1e-6)
        return kernel

    def apply_diffraction_spikes(self, img: torch.Tensor) -> torch.Tensor:
        """
        Adds diffraction spikes (bloom) to bright specular highlights.
        """
        cfg = self.cfg.sensor
        if not cfg.diffraction_spikes_enable:
            return img
            
        # 1. Extract highlights
        threshold = cfg.diffraction_spikes_threshold
        # Relu-based extraction
        mask = F.relu(img - threshold)
        
        if mask.max() <= 0:
            return img
            
        # 2. Generate Kernel
        k_size = int(cfg.diffraction_spikes_length * 2 + 1)
        kernel = self._generate_star_kernel(
            k_size, 
            cfg.diffraction_spikes_count, 
            cfg.diffraction_spikes_angle_deg
        )
        kernel = kernel.view(1, 1, k_size, k_size)
        
        # 3. Convolve
        C, H, W = img.shape
        pad = k_size // 2
        
        # Per-channel convolution
        bloom = F.conv2d(mask.unsqueeze(0), kernel.repeat(C, 1, 1, 1), padding=pad, groups=C).squeeze(0)
        
        # 4. Add back
        # Intensity scaling: The kernel is normalized, so bloom preserves energy of 'mask'.
        # 'mask' contains (intensity - threshold).
        # We multiply by user intensity factor.
        return torch.clamp(img + bloom * cfg.diffraction_spikes_intensity * 20.0, 0, 255)

    def apply_spectral_dispersion(self, img: torch.Tensor) -> torch.Tensor:
        """
        Simulate spectral dispersion by shifting channels based on intensity gradient.
        """
        cfg = self.cfg.sensor
        if not cfg.spectral_dispersion_enable:
            return img
            
        # Strength might be tensor if we optimize it
        s = cfg.spectral_dispersion_strength
        s_val = s.item() if torch.is_tensor(s) else s
        
        if s_val <= 0.001:
             # Ensure s is treated as a tensor if possible
             if torch.is_tensor(s):
                 return img + 0.0 * s.sum()
             else:
                 return img
             
        # Calculate luminance (0-1 range)
        # BGR coefficients
        lum = (0.114 * img[0] + 0.587 * img[1] + 0.299 * img[2]) / 255.0
        lum = lum.unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
        
        # Sobel Gradients
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=self.device, dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=self.device, dtype=torch.float32).view(1, 1, 3, 3)
        
        gx = F.conv2d(lum, sobel_x, padding=1)
        gy = F.conv2d(lum, sobel_y, padding=1)
        
        H, W = img.shape[1], img.shape[2]
        
        # Create base grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=self.device),
            torch.linspace(-1, 1, W, device=self.device),
            indexing='ij'
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0) # (1, H, W, 2)
        
        # Convert gradients to grid units [-1, 1]
        # Use s (tensor)
        gx_grid = gx.squeeze(0).squeeze(0) * (s * 2.0 / W)
        gy_grid = gy.squeeze(0).squeeze(0) * (s * 2.0 / H)
        
        disp = torch.stack([gx_grid, gy_grid], dim=-1).unsqueeze(0)
        
        # Apply Shifts (Opposite directions for Red and Blue)
        # BGR: Blue=0, Red=2
        
        b = img[0].view(1, 1, H, W)
        r = img[2].view(1, 1, H, W)
        
        # Padding mode reflection helps at edges
        b_new = F.grid_sample(b, base_grid - disp, align_corners=False, padding_mode='reflection')
        r_new = F.grid_sample(r, base_grid + disp, align_corners=False, padding_mode='reflection')
        
        # Combine
        out = torch.stack([b_new.squeeze(), img[1], r_new.squeeze()], dim=0)
        return out

    def apply_overlay_and_export(self, img_tensor: torch.Tensor, rng: random.Random, is_rgb: bool = False) -> np.ndarray:
        """
        Final step: downloads tensor to CPU, applies overlay (scalebar), returns numpy (H, W, 3) BGR uint8.
        """
        # Clamp and Convert
        img_tensor = torch.clamp(img_tensor, 0, 255).byte()
        # (3, H, W) -> (H, W, 3)
        img_cpu = img_tensor.permute(1, 2, 0).cpu().numpy()
        
        # If input was already RGB (polarization_rgb), we need to swap channels for CV2/Output which expects BGR
        if is_rgb:
            # RGB -> BGR
            img_cpu = img_cpu[..., ::-1]
        
        cfg = self.cfg
        if not cfg.sensor.scalebar.enable or rng.random() > cfg.sensor.scalebar.prob:
            return img_cpu
            
        # Re-implement overlay logic using CPU libraries on the numpy array
        # Just call the logic. I can copy the logic here or make a helper.
        # Since I am replacing the old class, I should implement the logic here.
        
        return self._draw_scalebar(img_cpu, rng)

    def _find_ttf(self) -> Optional[str]:
        from pathlib import Path
        cfg = self.cfg
        if cfg.sensor.scalebar.ttf and Path(str(cfg.sensor.scalebar.ttf)).exists():
            return str(cfg.sensor.scalebar.ttf)
        try:
            import matplotlib
            p = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
            if p.exists():
                return str(p)
        except Exception:
            pass
        for p in [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/google-noto/NotoSans-Regular.ttf",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        ]:
            if Path(p).exists():
                return p
        return None

    def _draw_scalebar(self, img_bgr: np.ndarray, rng: random.Random) -> np.ndarray:
        # Copied logic from original SensorHead
        cfg = self.cfg
        h, w = img_bgr.shape[:2]
        val = rng.randint(cfg.sensor.scalebar.value_range[0], cfg.sensor.scalebar.value_range[1])
        units = list(cfg.sensor.scalebar.units)
        idx = int(rng.random() * len(units)) if units else 0
        unit = units[idx] if units else ""
        text = f"{val} {unit}"
        font_px = int(rng.uniform(*cfg.sensor.scalebar.font_px))
        
        ttf = self._find_ttf() if PIL_AVAILABLE else None
        if ttf is None and ("μ" in unit):
            text = f"{val} um"
            
        if PIL_AVAILABLE:
            # Note: img_bgr is numpy uint8
            pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil)
            font = ImageFont.truetype(ttf, font_px) if ttf else ImageFont.load_default()
            
            L = int(rng.uniform(*cfg.sensor.scalebar.len_px))
            thick = int(rng.uniform(*cfg.sensor.scalebar.thick_px))
            margin = cfg.sensor.scalebar.margin_px
            corners = ["tl", "tr", "bl", "br"]
            corner = corners[int(rng.random() * len(corners))]
            c = int(rng.randint(cfg.sensor.scalebar.white_jit[0], cfg.sensor.scalebar.white_jit[1]))
            fill = (c, c, c)
            
            if corner == "bl":
                x0, y0 = margin, h - margin
                x1, y1 = x0 + L, y0
            elif corner == "br":
                x1, y1 = w - margin, h - margin
                x0, y0 = x1 - L, y1
            elif corner == "tl":
                x0, y0 = margin, margin
                x1, y1 = x0 + L, y0
            else:
                x1, y1 = w - margin, margin
                x0, y0 = x1 - L, y1
                
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            pad = 6
            
            if corner in ("bl", "br"):
                tx = x1 - tw if corner == "br" else x0
                ty = y0 - pad - th
            else:
                tx = x1 - tw if corner == "tr" else x0
                ty = y0 + pad
                
            tx = max(0, min(tx, w - tw))
            ty = max(0, min(ty, h - th))
            
            if cfg.sensor.scalebar.outline:
                sw = max(1, int(round(font_px * 0.08)))
                draw.text((tx, ty), text, font=font, fill=fill, stroke_width=sw, stroke_fill=(0, 0, 0))
            else:
                draw.text((tx, ty), text, font=font, fill=fill)
                
            if cfg.sensor.scalebar.outline:
                draw.line([(x0, y0), (x1, y1)], fill=(0, 0, 0), width=thick + 2)
            draw.line([(x0, y0), (x1, y1)], fill=fill, width=thick)
            
            return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        else:
            # Fallback
            L = int(rng.uniform(*cfg.sensor.scalebar.len_px))
            thick = int(rng.uniform(*cfg.sensor.scalebar.thick_px))
            margin = cfg.sensor.scalebar.margin_px
            bl = rng.random() < 0.5
            if bl:
                x0, y0 = margin, h - margin
                x1, y1 = x0 + L, y0
            else:
                x1, y1 = w - margin, h - margin
                x0, y0 = x1 - L, y1
            cv2.line(img_bgr, (x0, y0), (x1, y1), (0, 0, 0), thickness=thick + 2, lineType=cv2.LINE_AA)
            cv2.line(img_bgr, (x0, y0), (x1, y1), (255, 255, 255), thickness=thick, lineType=cv2.LINE_AA)
            return img_bgr
