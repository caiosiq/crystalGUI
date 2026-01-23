import torch
import torch.nn.functional as F
import math
import random
import numpy as np
from typing import Optional, List, Tuple
from ..config import SynthConfig

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
        if sigma <= 0:
            return img
            
        if img.dim() == 2:
            img = img.unsqueeze(0) # (1, H, W)
            squeeze = True
        else:
            squeeze = False
            
        C, H, W = img.shape
        
        k_ideal = int(round(3 * sigma)) * 2 + 1
        max_k = min(H, W)
        if max_k % 2 == 0:
            max_k -= 1
        k_size = max(1, min(k_ideal, max_k))
        
        if k_size == 1:
            return img.squeeze(0) if squeeze else img

        pad = k_size // 2
        x = torch.arange(k_size, device=img.device, dtype=img.dtype) - pad
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
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
        # (1, H, W)
        base = torch.randint(gmin, gmax + 1, (1, h, w), device=dev, dtype=torch.float32, generator=gen)
        img = base # (1, H, W)
        
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
        if cfg.sensor.illum_ampl and cfg.sensor.illum_ampl > 0:
            sigma = cfg.sensor.illum_sigma
            if sigma > 10:
                # Optimization: Generate small noise and upsample
                # This simulates "large blur" without the heavy convolution
                scale = 1.0 / (sigma * 0.5) # Heuristic scaling
                sh, sw = max(4, int(h * scale)), max(4, int(w * scale))
                
                # Generate small noise
                noise_small = torch.randn(1, 1, sh, sw, device=dev, generator=gen)
                
                # Upsample to full size (Bilinear gives smooth "blurred" look)
                noise = F.interpolate(noise_small, size=(h, w), mode='bilinear', align_corners=False).squeeze(0)
            else:
                # Standard path for small blurs (grain)
                noise = torch.randn(1, h, w, device=dev, generator=gen)
                if sigma > 0:
                    noise = self._gaussian_blur_2d(noise, sigma)
            
            rngv = noise.max() - noise.min()
            if rngv > 1e-6:
                noise = (noise - noise.min()) / (rngv + 1e-6)
                noise = (noise - 0.5) * 2.0
                img = img + float(cfg.sensor.illum_ampl) * noise
                
        # Vignette
        if cfg.sensor.vignette_strength > 0:
            grid_y, grid_x = torch.meshgrid(
                torch.arange(h, device=dev, dtype=torch.float32),
                torch.arange(w, device=dev, dtype=torch.float32),
                indexing='ij'
            )
            cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
            r = torch.sqrt((grid_x - cx) ** 2 + (grid_y - cy) ** 2)
            r = r / (r.max() + 1e-6)
            
            vig = (1.0 - float(cfg.sensor.vignette_strength) * (r ** 2))
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
            img = img + float(gain) * S
            
        # Fouling (Lens Dirt / Biofilm)
        if cfg.sensor.fouling_enable and rng.random() < cfg.sensor.fouling_prob:
            # Generate static blobs
            n_blobs = rng.randint(*cfg.sensor.fouling_count_range)
            if n_blobs > 0:
                # Use large, blurry dots
                # Coordinates
                fx = torch.randint(0, w, (n_blobs,), device=dev)
                fy = torch.randint(0, h, (n_blobs,), device=dev)
                fsig = torch.rand(n_blobs, device=dev) * (cfg.sensor.fouling_sigma_range[1] - cfg.sensor.fouling_sigma_range[0]) + cfg.sensor.fouling_sigma_range[0]
                
                # Render blobs

                foul_mask = torch.zeros(1, h, w, device=dev)
                
                # Draw points (1.0)

                frame_sigma = rng.uniform(*cfg.sensor.fouling_sigma_range)
                
                # Draw random shapes/blobs
                # Low freq noise + threshold?
                # Or sparse points
                
                # Sparse points
                foul_mask[0, fy, fx] = 1.0
                
                # Blur
                foul_mask = self._gaussian_blur_2d(foul_mask, frame_sigma)
                
                # Normalize peak to 1? No, preserve accumulation.
                max_val = foul_mask.max()
                if max_val > 1e-6:
                    foul_mask = foul_mask / max_val
                
                # Apply as darkening (dirt blocks light)
                # or lightening (scattering)? Usually dark in brightfield.
                # opacity determines strength.
                
                # Invert mask so 1 = clear, 0 = dirt?
                # Currently mask has 1 at dirt center.
                strength = cfg.sensor.fouling_opacity
                
                # Darken: img = img * (1 - strength * mask)
                img = img * (1.0 - strength * foul_mask)

        # Biofilm / Residue (Low-frequency overlay)
        # Adds a structured, low-frequency noise texture (Perlin-like)
        # We simulate this by upsampling small noise + some thresholding
        if cfg.sensor.fouling_enable and rng.random() < 0.4: # 40% chance if fouling enabled
             # Generate low freq noise
             scale = 1.0 / 32.0
             sh, sw = max(4, int(h * scale)), max(4, int(w * scale))
             noise_bio = torch.randn(1, 1, sh, sw, device=dev, generator=gen)
             noise_bio = F.interpolate(noise_bio, size=(h, w), mode='bilinear', align_corners=False).squeeze(0)
             
             # Normalize 0..1
             noise_bio = (noise_bio - noise_bio.min()) / (noise_bio.max() - noise_bio.min() + 1e-6)
             
             # Threshold to create "patches"
             # Patches are where noise > 0.6
             bio_mask = torch.clamp((noise_bio - 0.5) * 3.0, 0.0, 1.0)
             
             # Apply texture to these patches
             # Texture is high frequency noise
             tex = torch.randn(1, h, w, device=dev, generator=gen) * 0.1
             
             # Apply: Darken areas with biofilm
             strength = 0.15 * cfg.sensor.fouling_opacity
             img = img * (1.0 - strength * bio_mask * (1.0 + tex))

        # Background noise
        if cfg.sensor.bg_noise_std and cfg.sensor.bg_noise_std > 0:
            noise = float(cfg.sensor.bg_noise_std) * torch.randn(1, h, w, device=dev, generator=gen)
            img = img + noise
            
        img = torch.clamp(img, 0, 255)
        
        # Expand to 3 channels (BGR)
        img = img.repeat(3, 1, 1)
        return img

    def apply_blur(self, img: torch.Tensor) -> torch.Tensor:
        """
        img: (3, H, W)
        """
        if self.cfg.sensor.blur_sigma > 0:
            return self._gaussian_blur_2d(img, self.cfg.sensor.blur_sigma)
        return img

    def apply_chromatic_aberration(self, img: torch.Tensor, strength: float = 0.0) -> torch.Tensor:
        """
        Simulate lateral chromatic aberration (color fringing).
        Shifts R channel outward and B channel inward radially.
        img: (3, H, W)
        strength: approximate pixel shift at corners
        """
        if strength <= 0.001:
            return img
            
        C, H, W = img.shape
        dev = img.device
        
        # Grid (normalized -1 to 1)
        # We need this every frame, but it's fast on GPU.
        # Ideally cache this if H/W don't change.
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=dev),
            torch.linspace(-1, 1, W, device=dev),
            indexing='ij'
        )
        
        # Stack: (1, H, W, 2) for grid_sample
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        
        # Scaling factors
        # strength=1 means ~1 pixel shift at edge?
        # grid goes -1 to 1. Width W corresponds to 2.0 in grid space.
        # 1 pixel = 2.0 / W
        # If we scale grid by (1 + k), the sampling moves outward.
        
        # Approximate scaling factor
        # k = strength / (W/2)
        # scale_r = 1.0 + k
        # scale_b = 1.0 - k
        
        k = strength * 2.0 / max(H, W)
        scale_r = 1.0 + k
        scale_b = 1.0 - k
        
        # R Channel: Sample from smaller grid (zoomed in -> features move OUT)
        # Wait, if we sample from coordinate 0.5 using grid 0.45, we get value from closer to center.
        # If we zoom IN to the image, the features move OUT.
        # To zoom IN, we need to sample a SMALLER area of the source.
        # So grid should be scaled DOWN (multiplied by < 1).
        # grid_r = base_grid / scale_r where scale_r > 1 -> grid becomes smaller?
        # Yes.
        
        grid_r = base_grid / scale_r
        grid_b = base_grid / scale_b # scale_b < 1 -> grid larger -> zoom out -> features move IN
        
        # Split channels (keep dimension for grid_sample)
        # (C, H, W) -> (1, 1, H, W) per channel
        r = img[0].view(1, 1, H, W)
        g = img[1].view(1, 1, H, W)
        b = img[2].view(1, 1, H, W)
        
        # Resample
        # padding_mode='reflection' avoids black borders
        r_new = F.grid_sample(r, grid_r, align_corners=False, padding_mode='reflection')
        b_new = F.grid_sample(b, grid_b, align_corners=False, padding_mode='reflection')
        
        # Combine
        # (1, 1, H, W) -> (H, W)
        out = torch.stack([r_new.squeeze(), g.squeeze(), b_new.squeeze()], dim=0)
        
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
