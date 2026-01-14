import torch
import torch.nn.functional as F
import math
import random
import numpy as np
from typing import Tuple, Union, Optional, Callable, List
from ..config import SynthConfig
from ..physics.particles import Rod, Debris, RodBatch, DebrisBatch
from ..physics.ghosts import GhostObject
from ..utils.math_torch import (
    noise1d_like, sin_wobble, kink, noisy_wobble, ragged_mask, smooth_cap, gaussian_blur_1d,
    noise1d_like_batch, sin_wobble_batch, kink_batch, noisy_wobble_batch, ragged_mask_batch
)

class DICModulatorTorch:
    def __init__(self, config: SynthConfig, device: str = "cpu"):
        self.cfg = config
        self.device = torch.device(device)
        
    def render(self, obj: Union[Rod, Debris, GhostObject], rng: random.Random, np_rng: np.random.RandomState) -> Tuple[Optional[torch.Tensor], int, int]:
        """
        Dispatch method to render any supported object type.
        Returns (patch_tensor_CHW, x_offset, y_offset).
        """
        if isinstance(obj, GhostObject):
            wrapped = obj.wrapped_obj
            if isinstance(wrapped, Rod):
                return self.render_rod(obj, rng, np_rng)
            elif isinstance(wrapped, Debris):
                return self.render_debris(obj, rng)
            else:
                raise NotImplementedError(f"DICModulatorTorch: Ghost wraps unknown type {type(wrapped)}")

        if isinstance(obj, Rod):
            return self.render_rod(obj, rng, np_rng)
        elif isinstance(obj, Debris):
            return self.render_debris(obj, rng)
        else:
            raise NotImplementedError(f"DICModulatorTorch does not know how to render {type(obj)}")

    def _get_warp_function(self, rod: Rod, seed: int) -> Optional[Callable[[torch.Tensor], torch.Tensor]]:
        mode = rod.shape_mode
        L = rod.L
        if mode == "wavy":
            amp = 0.6 + 0.004 * L
            return lambda u: sin_wobble(u, amp_px=amp, cycles=(0.7, 1.6), seed=seed)
        elif mode == "kink":
            return lambda u: kink(u, amp_px=1.0 + 0.006 * L, seed=seed)
        elif mode == "noisy":
            return lambda u: noisy_wobble(u, amp_px=1.0, corr=0.22, seed=seed)
        return None

    def _get_blur_sigma(self, z: float) -> float:
        ghost_sig = self.cfg.physics.ghosts.blur_sigma
        scale = max(0.1, ghost_sig) if ghost_sig > 0 else 2.0
        return abs(z) * scale

    def render_rod(self, rod: Rod, rng: random.Random, np_rng: np.random.RandomState) -> Tuple[Optional[torch.Tensor], int, int]:
        cfg = self.cfg
        
        # Unpack rod properties
        cx, cy, L, W, ang_deg = rod.cx, rod.cy, rod.L, rod.W, rod.angle_deg
        
        # Calculate bounding box with padding
        corners = rod.corners 
        # corners is numpy array, that's fine for bounds calc on CPU
        
        pad = max(6, int(max(cfg.optics.rod_halo_sigma, 3))) + 6
        
        x_min = int(np.floor(corners[:, 0].min())) - pad
        x_max = int(np.ceil(corners[:, 0].max())) + pad
        y_min = int(np.floor(corners[:, 1].min())) - pad
        y_max = int(np.ceil(corners[:, 1].max())) + pad
        
        w_patch = x_max - x_min
        h_patch = y_max - y_min
        
        if w_patch <= 0 or h_patch <= 0:
            return None, 0, 0
            
        # Create local coordinate grids on GPU
        # y: 0..h-1, x: 0..w-1
        # meshgrid 'ij' indexing produces y, x order
        yy, xx = torch.meshgrid(
            torch.arange(h_patch, device=self.device, dtype=torch.float32),
            torch.arange(w_patch, device=self.device, dtype=torch.float32),
            indexing='ij'
        )
        
        X = xx + x_min - cx
        Y = yy + y_min - cy
        
        th = math.radians(ang_deg)
        ct, st = math.cos(th), math.sin(th)
        u = (ct * X + st * Y) / (L / 2.0 + 1e-6)
        v = (-st * X + ct * Y)
        
        # Seeds for torch operations
        seed_warp = rng.randint(0, 2**31 - 1)
        seed_width = rng.randint(0, 2**31 - 1)
        seed_offset = rng.randint(0, 2**31 - 1)
        seed_edge = rng.randint(0, 2**31 - 1)
        seed_pol = rng.randint(0, 2**31 - 1)
        seed_rag = rng.randint(0, 2**31 - 1)
        
        # Apply Shape Warp
        v_warp_fn = self._get_warp_function(rod, seed_warp)
        if v_warp_fn:
            v_warp = v_warp_fn(u)
            v = v - v_warp
            
        if rod.curvature != 0.0:
            v = v + rod.curvature * (u * u - 1 / 3) * L

        # Tapered envelope
        taper = cfg.optics.taper_strength * (torch.abs(u) ** cfg.optics.taper_power)
        w_u = torch.maximum(torch.tensor(cfg.optics.min_width_ratio, device=self.device), 1.0 - taper) * (W + 1e-6)
        
        if rod.width_jit_amp > 0:
            w_u = w_u * (1.0 + noise1d_like(u, corr=0.22, amp=rod.width_jit_amp, seed=seed_width))

        sigma_v = (w_u * max(1e-6, cfg.optics.cross_soft_sigma))
        alpha_v = torch.exp(-0.5 * (v / sigma_v) ** 2)
        alpha_u = smooth_cap(u, a=0.78, b=1.00)
        alpha_fill = torch.clamp(alpha_v * alpha_u, 0.0, 1.0)
        
        # Base delta body
        g = np_rng.randn() # Keep using np_rng for scalar params to match logic
        delta_body = float(rod.delta) * (1.0 + 0.05 * g)
        
        layer = delta_body * alpha_fill

        # Phase contrast edge pair (Shadow)
        shadow_gain = rng.uniform(*cfg.optics.shadow_gain)
        if abs(rod.z) > 0.5:
             shadow_gain *= cfg.physics.ghosts.gain_mult

        shadow_width_mult = rng.uniform(*cfg.optics.shadow_width_mult)
        shadow_bias = rng.uniform(*cfg.optics.shadow_bias)
        shadow_offset_px = rng.uniform(*cfg.optics.shadow_offset_px)

        sigma_pc = torch.clamp(sigma_v * shadow_width_mult, min=0.6)
        sign = 1.0 if (rng.random() < 0.5) else -1.0
        
        offset_jit = 0.0
        if rod.offset_jit_amp > 0:
            # Note: noise1d_like expects u
            jit_noise = noise1d_like(u, 0.25, 1.0, seed=seed_offset)
            offset_jit = rod.offset_jit_amp * jit_noise
            
        v_shift = v - (shadow_offset_px * (1.0 + offset_jit) * sign)
        pc = (v_shift / sigma_pc) * torch.exp(-0.5 * (v_shift / sigma_pc) ** 2) / 0.60653066
        pc *= alpha_u
        
        polarity = 1.0 if (rng.random() < 0.5) else -1.0
        pc *= polarity
        
        if rod.edge_jit_amp > 0:
            pc = pc * (1.0 + rod.edge_jit_amp * noise1d_like(u, corr=0.22, amp=1.0, seed=seed_edge))
            
        if rod.polarity_flip_p > 0.0 and (rng.random() < rod.polarity_flip_p):
            flips = torch.sign(noise1d_like(u, corr=0.30, amp=1.0, seed=seed_pol))
            flips[flips == 0] = 1.0
            pc *= flips
            
        pc_pos = torch.maximum(pc, torch.tensor(0.0, device=self.device)) * (1.0 - shadow_bias)
        pc_neg = torch.minimum(pc, torch.tensor(0.0, device=self.device)) * (1.0 + shadow_bias)
        pc = pc_pos + pc_neg
        
        if rod.ragged_p > 0:
            mask_u = ragged_mask(u, p=rod.ragged_p, corr=rod.ragged_corr, seed=seed_rag)
            pc *= mask_u
            alpha_fill *= mask_u
            
        layer = layer + shadow_gain * pc

        # Halo
        if cfg.optics.rod_halo_sigma > 0 and cfg.optics.rod_halo_gain != 0:
            support = (alpha_fill > 0.12).float()
            
            # Using separable 1D blurs for efficiency and safety
            blurred = self._gaussian_blur_2d(support, cfg.optics.rod_halo_sigma)
                
            halo = torch.clamp(blurred - support, 0, 1)
            layer = layer + (delta_body * (cfg.optics.rod_halo_gain * 1.0)) * halo

        # Apply Depth Blur
        blur_sig = self._get_blur_sigma(rod.z)
        if blur_sig > 0.5:
            # Prefer our safe implementation
            layer = self._gaussian_blur_2d(layer, blur_sig)

        # Expand to 3 channels (H, W) -> (3, H, W)
        patch = torch.stack([layer, layer, layer], dim=0)
        
        return patch, x_min, y_min

    def render_debris(self, debris: Debris, rng: random.Random) -> Tuple[Optional[torch.Tensor], int, int]:
        r = debris.size_px
        x_min = int(round(debris.cx - r))
        x_max = int(round(debris.cx + r + 1))
        y_min = int(round(debris.cy - r))
        y_max = int(round(debris.cy + r + 1))
        
        w_patch = x_max - x_min
        h_patch = y_max - y_min
        
        if w_patch <= 0 or h_patch <= 0:
            return None, 0, 0
            
        # Coordinates
        yy, xx = torch.meshgrid(
            torch.arange(h_patch, device=self.device, dtype=torch.float32),
            torch.arange(w_patch, device=self.device, dtype=torch.float32),
            indexing='ij'
        )
        cx_rel = debris.cx - x_min
        cy_rel = debris.cy - y_min
        
        patch = torch.zeros((h_patch, w_patch), device=self.device, dtype=torch.float32)
        
        if debris.is_dash:
            # Simulate line segment
            Ld = r # half length
            ang_rad = math.radians(debris.angle_deg)
            dx, dy = math.cos(ang_rad), math.sin(ang_rad)
            
            # Line segment from p1 to p2
            p1x, p1y = cx_rel - Ld*dx, cy_rel - Ld*dy
            p2x, p2y = cx_rel + Ld*dx, cy_rel + Ld*dy
            
            # Distance from point (xx, yy) to segment (p1, p2)
            # Vector A = p1, B = p2, P = (xx, yy)
            # Project AP onto AB to find closest point on line
            
            abx, aby = p2x - p1x, p2y - p1y
            apx, apy = xx - p1x, yy - p1y
            
            len_sq = abx**2 + aby**2
            t = (apx * abx + apy * aby) / (len_sq + 1e-6)
            t = torch.clamp(t, 0.0, 1.0)
            
            # Closest point C
            cx_line = p1x + t * abx
            cy_line = p1y + t * aby
            
            # Distance squared
            dist_sq = (xx - cx_line)**2 + (yy - cy_line)**2
            dist = torch.sqrt(dist_sq)
            
            # Antialiased line width = 1
            intensity = torch.clamp(1.0 - (dist - 0.5), 0.0, 1.0) * float(debris.delta)
            patch = intensity
            
        else:
            # Circle
            dist2 = (xx - cx_rel)**2 + (yy - cy_rel)**2
            mask = (dist2 <= r*r).float()
            patch = mask * float(debris.delta)

        # Apply Depth Blur
        blur_sig = self._get_blur_sigma(debris.z)
        if blur_sig > 0.5:
            patch = self._gaussian_blur_2d(patch, blur_sig)

        patch_chw = torch.stack([patch, patch, patch], dim=0)
        return patch_chw, x_min, y_min

    def _gaussian_blur_2d(self, img: torch.Tensor, sigma: float) -> torch.Tensor:
        """
        Robust 2D Gaussian blur using separable 1D convolutions.
        Handles small patches by clamping kernel size.
        img: (H, W)
        """
        H, W = img.shape
        
        # 1. Calculate ideal kernel size based on sigma
        k_ideal = int(round(3 * sigma)) * 2 + 1
        
        # 2. CLAMP kernel size to be smaller than the smallest image dimension
        max_k = min(H, W)
        if max_k % 2 == 0:
            max_k -= 1
            
        k_size = min(k_ideal, max_k)
        k_size = max(1, k_size) # Ensure at least 1x1
        
        if k_size == 1:
            return img # No blur needed/possible

        # 3. Create Kernel
        pad = k_size // 2
        x = torch.arange(k_size, device=img.device, dtype=img.dtype) - pad
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / (kernel.sum() + 1e-6)
        kernel = kernel.view(1, 1, -1) # (1, 1, K)
        
        # 4. Blur Rows (Dimension W)
        inp_rows = img.view(1, H, W)
        kernel_rows = kernel.repeat(H, 1, 1) # (H, 1, K)
        out_rows = F.conv1d(inp_rows, kernel_rows, padding=pad, groups=H)
        
        # 5. Blur Cols (Dimension H)
        inp_cols = out_rows.view(H, W).t().view(1, W, H)
        kernel_cols = kernel.repeat(W, 1, 1) # (W, 1, K)
        out_cols = F.conv1d(inp_cols, kernel_cols, padding=pad, groups=W)
        
        return out_cols.view(W, H).t()

    def _gaussian_blur_batch(self, img: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
        """
        Batched 2D Gaussian blur with per-image sigma.
        img: (N, H, W)
        sigmas: (N,) or float
        """
        N, H, W = img.shape
        if isinstance(sigmas, float):
            sigmas = torch.full((N,), sigmas, device=img.device)
        
        # Clamp sigmas to avoid numerical issues
        sigmas = torch.clamp(sigmas, min=0.1)
        max_sigma = sigmas.max().item()
        
        # 1. Calculate ideal kernel size based on max sigma
        k_ideal = int(round(3 * max_sigma)) * 2 + 1
        
        # 2. CLAMP kernel size to be smaller than the smallest image dimension
        max_k = min(H, W)
        if max_k % 2 == 0:
            max_k -= 1
        k_size = min(k_ideal, max_k)
        k_size = max(1, k_size)
        
        if k_size == 1:
            return img

        # 3. Create Kernels (N, K)
        pad = k_size // 2
        x = torch.arange(k_size, device=img.device, dtype=img.dtype) - pad
        # x is (K,)
        # Expand x to (1, K) and sigmas to (N, 1) for broadcasting
        # result (N, K)
        kernels = torch.exp(-0.5 * (x.unsqueeze(0) / sigmas.unsqueeze(1)) ** 2)
        kernels = kernels / (kernels.sum(dim=1, keepdim=True) + 1e-6)
        
        # 4. Blur Rows (Dimension W)
        inp_rows = img.view(1, N * H, W)
        k_rows = kernels.unsqueeze(1).repeat(1, H, 1).view(N * H, 1, k_size)
        out_rows = F.conv1d(inp_rows, k_rows, padding=pad, groups=N * H)
        out_rows = out_rows.view(N, H, W)
        
        # 5. Blur Cols (Dimension H)
        inp_cols = out_rows.transpose(1, 2).reshape(1, N * W, H)
        k_cols = kernels.unsqueeze(1).repeat(1, W, 1).view(N * W, 1, k_size)
        out_cols = F.conv1d(inp_cols, k_cols, padding=pad, groups=N * W)
        
        return out_cols.view(N, W, H).transpose(1, 2)

    def render_rods_batch(self, rods: Union[List[Rod], RodBatch], rng: random.Random, np_rng: np.random.RandomState) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Vectorized rendering for a batch of Rods.
        Returns:
            patches: (N, 3, H, W)
            x_min: (N,)
            y_min: (N,)
        """
        if isinstance(rods, list):
            if not rods:
                return None, torch.empty(0), torch.empty(0)
            N = len(rods)
            
            # Basic properties
            t_cxs = torch.tensor([r.cx for r in rods], device=self.device, dtype=torch.float32)
            t_cys = torch.tensor([r.cy for r in rods], device=self.device, dtype=torch.float32)
            t_Ls = torch.tensor([r.L for r in rods], device=self.device, dtype=torch.float32)
            t_Ws = torch.tensor([r.W for r in rods], device=self.device, dtype=torch.float32)
            t_angles = torch.tensor([r.angle_deg for r in rods], device=self.device, dtype=torch.float32)
            t_deltas = torch.tensor([r.delta for r in rods], device=self.device, dtype=torch.float32)
            t_zs = torch.tensor([r.z for r in rods], device=self.device, dtype=torch.float32)
            t_curv = torch.tensor([r.curvature for r in rods], device=self.device, dtype=torch.float32)
            
            t_w_jit = torch.tensor([r.width_jit_amp for r in rods], device=self.device, dtype=torch.float32)
            t_off_jit = torch.tensor([r.offset_jit_amp for r in rods], device=self.device, dtype=torch.float32)
            t_edge_jit = torch.tensor([r.edge_jit_amp for r in rods], device=self.device, dtype=torch.float32)
            t_pol_p = torch.tensor([r.polarity_flip_p for r in rods], device=self.device, dtype=torch.float32)
            t_rag_p = torch.tensor([r.ragged_p for r in rods], device=self.device, dtype=torch.float32)
            t_rag_corr = torch.tensor([r.ragged_corr for r in rods], device=self.device, dtype=torch.float32)
            
            # Shape modes handling for list
            modes_list = [r.shape_mode for r in rods]
            SHAPE_MODE_MAP = {"straight": 0, "wavy": 1, "kink": 2, "noisy": 3}
            t_modes = torch.tensor([SHAPE_MODE_MAP.get(m, 0) for m in modes_list], device=self.device, dtype=torch.long)
            
            pad = max(6, int(max(self.cfg.optics.rod_halo_sigma, 3))) + 6
            all_corners = [r.corners for r in rods]
            x_mins_list, y_mins_list, ws_list, hs_list = [], [], [], []
            for c in all_corners:
                xm = int(np.floor(c[:, 0].min())) - pad
                xM = int(np.ceil(c[:, 0].max())) + pad
                ym = int(np.floor(c[:, 1].min())) - pad
                yM = int(np.ceil(c[:, 1].max())) + pad
                x_mins_list.append(xm); y_mins_list.append(ym)
                ws_list.append(xM - xm); hs_list.append(yM - ym)
            
            t_x_mins = torch.tensor(x_mins_list, device=self.device, dtype=torch.int32)
            t_y_mins = torch.tensor(y_mins_list, device=self.device, dtype=torch.int32)
            t_ws = torch.tensor(ws_list, device=self.device, dtype=torch.int32)
            t_hs = torch.tensor(hs_list, device=self.device, dtype=torch.int32)

        elif isinstance(rods, RodBatch):
            if rods.cx.numel() == 0:
                return None, torch.empty(0), torch.empty(0)
            
            N = rods.cx.shape[0]
            # Ensure on device
            if rods.cx.device != self.device:
                rods.to(self.device)
                
            t_cxs, t_cys = rods.cx, rods.cy
            t_Ls, t_Ws = rods.L, rods.W
            t_angles = rods.angle_deg
            t_deltas = rods.delta
            t_zs = rods.z
            t_curv = rods.curvature
            
            t_w_jit = rods.width_jit_amp
            t_off_jit = rods.offset_jit_amp
            t_edge_jit = rods.edge_jit_amp
            t_pol_p = rods.polarity_flip_p
            t_rag_p = rods.ragged_p
            t_rag_corr = rods.ragged_corr
            t_modes = rods.shape_mode
            
            # Calculate bounds on GPU from parameters
            th_rad = torch.deg2rad(t_angles)
            c, s = torch.abs(torch.cos(th_rad)), torch.abs(torch.sin(th_rad))
            
            box_w = t_Ls * c + t_Ws * s
            box_h = t_Ls * s + t_Ws * c
            
            pad = max(6, int(max(self.cfg.optics.rod_halo_sigma, 3))) + 6
            
            t_x_mins = torch.floor(t_cxs - box_w / 2.0).int() - pad
            t_y_mins = torch.floor(t_cys - box_h / 2.0).int() - pad
            
            x_maxs = torch.ceil(t_cxs + box_w / 2.0).int() + pad
            y_maxs = torch.ceil(t_cys + box_h / 2.0).int() + pad
            
            t_ws = x_maxs - t_x_mins
            t_hs = y_maxs - t_y_mins
            
        else:
             raise ValueError("rods must be list or RodBatch")
        
        cfg = self.cfg
        
        # Unified grid size
        max_w = min(t_ws.max().item(), 1024)
        max_h = min(t_hs.max().item(), 1024)
        
        dev = self.device
        
        # Grid Generation
        yy, xx = torch.meshgrid(
            torch.arange(max_h, device=dev, dtype=torch.float32),
            torch.arange(max_w, device=dev, dtype=torch.float32),
            indexing='ij'
        )
        xx = xx.unsqueeze(0).expand(N, -1, -1)
        yy = yy.unsqueeze(0).expand(N, -1, -1)
        
        X = xx + t_x_mins.view(N, 1, 1) - t_cxs.view(N, 1, 1)
        Y = yy + t_y_mins.view(N, 1, 1) - t_cys.view(N, 1, 1)
        
        # Coordinates
        th = torch.deg2rad(t_angles).view(N, 1, 1)
        ct, st = torch.cos(th), torch.sin(th)
        
        L_half = (t_Ls.view(N, 1, 1) / 2.0) + 1e-6
        u = (ct * X + st * Y) / L_half
        v = (-st * X + ct * Y)
        
        # Shape Warp
        seed_warp = rng.randint(0, 2**31 - 1)
        v_warp = torch.zeros_like(v)
        
        # Wavy (1)
        is_wavy = (t_modes == 1).view(N, 1, 1)
        if is_wavy.any():
            amp = 0.6 + 0.004 * t_Ls.view(N, 1, 1)
            w = sin_wobble_batch(u, amp_px=amp, cycles=(0.7, 1.6), seed=seed_warp)
            v_warp = torch.where(is_wavy, w, v_warp)
            
        # Kink (2)
        is_kink = (t_modes == 2).view(N, 1, 1)
        if is_kink.any():
            amp = 1.0 + 0.006 * t_Ls.view(N, 1, 1)
            k = kink_batch(u, amp_px=amp, seed=seed_warp+1)
            v_warp = torch.where(is_kink, k, v_warp)
            
        # Noisy (3)
        is_noisy = (t_modes == 3).view(N, 1, 1)
        if is_noisy.any():
            n = noisy_wobble_batch(u, amp_px=1.0, corr=0.22, seed=seed_warp+2)
            v_warp = torch.where(is_noisy, n, v_warp)
            
        v = v - v_warp
        
        # Curvature
        if torch.any(t_curv != 0):
            v = v + t_curv.view(N, 1, 1) * (u * u - 1.0/3.0) * t_Ls.view(N, 1, 1)
            
        # Taper & Envelope
        taper = cfg.optics.taper_strength * (torch.abs(u) ** cfg.optics.taper_power)
        min_w = torch.tensor(cfg.optics.min_width_ratio, device=dev)
        w_u = torch.maximum(min_w, 1.0 - taper) * (t_Ws.view(N, 1, 1) + 1e-6)
        
        # Width Jitter
        if torch.any(t_w_jit > 0):
            jit = noise1d_like_batch(u, corr=0.22, amp=1.0, seed=rng.randint(0, 2**31-1))
            w_u = w_u * (1.0 + t_w_jit.view(N, 1, 1) * jit)
            
        sigma_v = w_u * max(1e-6, cfg.optics.cross_soft_sigma)
        alpha_v = torch.exp(-0.5 * (v / sigma_v) ** 2)
        alpha_u = smooth_cap(u, a=0.78, b=1.00)
        alpha_fill = torch.clamp(alpha_v * alpha_u, 0.0, 1.0)
        
        # Delta Body
        g = torch.randn(N, device=dev)
        delta_body = t_deltas * (1.0 + 0.05 * g)
        delta_body = delta_body.view(N, 1, 1)
        
        layer = delta_body * alpha_fill
        
        # Shadow / DIC
        sh_gain = torch.empty(N, device=dev).uniform_(*cfg.optics.shadow_gain)
        is_ghost = torch.abs(t_zs) > 0.5
        sh_gain = torch.where(is_ghost, sh_gain * cfg.physics.ghosts.gain_mult, sh_gain)
        
        sh_width_mult = torch.empty(N, device=dev).uniform_(*cfg.optics.shadow_width_mult)
        sh_bias = torch.empty(N, device=dev).uniform_(*cfg.optics.shadow_bias)
        sh_offset_px = torch.empty(N, device=dev).uniform_(*cfg.optics.shadow_offset_px)
        
        sigma_pc = torch.clamp(sigma_v * sh_width_mult.view(N, 1, 1), min=0.6)
        sign = torch.where(torch.rand(N, device=dev) < 0.5, 1.0, -1.0).view(N, 1, 1)
        
        offset_jit = torch.zeros_like(u)
        if torch.any(t_off_jit > 0):
            jit_noise = noise1d_like_batch(u, 0.25, 1.0, seed=rng.randint(0, 2**31-1))
            offset_jit = t_off_jit.view(N, 1, 1) * jit_noise
            
        v_shift = v - (sh_offset_px.view(N, 1, 1) * (1.0 + offset_jit) * sign)
        
        norm_v = v_shift / sigma_pc
        pc = norm_v * torch.exp(-0.5 * norm_v ** 2) / 0.60653066
        pc *= alpha_u
        
        polarity = torch.where(torch.rand(N, device=dev) < 0.5, 1.0, -1.0).view(N, 1, 1)
        pc *= polarity
        
        if torch.any(t_edge_jit > 0):
            jit = noise1d_like_batch(u, corr=0.22, amp=1.0, seed=rng.randint(0, 2**31-1))
            pc = pc * (1.0 + t_edge_jit.view(N, 1, 1) * jit)
            
        if torch.any(t_pol_p > 0):
            flips = torch.sign(noise1d_like_batch(u, corr=0.30, amp=1.0, seed=rng.randint(0, 2**31-1)))
            flips[flips == 0] = 1.0
            do_flip = (torch.rand(N, device=dev) < t_pol_p).view(N, 1, 1)
            pc = torch.where(do_flip, pc * flips, pc)
            
        pc_pos = torch.maximum(pc, torch.tensor(0.0, device=dev)) * (1.0 - sh_bias.view(N, 1, 1))
        pc_neg = torch.minimum(pc, torch.tensor(0.0, device=dev)) * (1.0 + sh_bias.view(N, 1, 1))
        pc = pc_pos + pc_neg
        
        if torch.any(t_rag_p > 0):
            p_val = float(t_rag_p.max().item())
            if p_val > 0:
                mask_u = ragged_mask_batch(u, p=p_val, corr=float(t_rag_corr.mean().item()), seed=rng.randint(0, 2**31-1))
                has_rag = (t_rag_p > 0).view(N, 1, 1)
                mask_u = torch.where(has_rag, mask_u, torch.ones_like(mask_u))
                pc *= mask_u
                alpha_fill *= mask_u
                
        layer = layer + sh_gain.view(N, 1, 1) * pc
        
        if cfg.optics.rod_halo_sigma > 0 and cfg.optics.rod_halo_gain != 0:
            support = (alpha_fill > 0.12).float()
            blurred = self._gaussian_blur_batch(support, cfg.optics.rod_halo_sigma)
            halo = torch.clamp(blurred - support, 0, 1)
            layer = layer + (delta_body * cfg.optics.rod_halo_gain) * halo
            
        ghost_sig = cfg.physics.ghosts.blur_sigma
        scale = max(0.1, ghost_sig) if ghost_sig > 0 else 2.0
        blur_sigs = torch.abs(t_zs) * scale
        
        if torch.any(blur_sigs > 0.5):
            layer = self._gaussian_blur_batch(layer, blur_sigs)
            
        patch = layer.unsqueeze(1).repeat(1, 3, 1, 1)
        
        return patch, t_x_mins, t_y_mins

    def render_debris_batch(self, debris: DebrisBatch, rng: random.Random) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Vectorized rendering for debris.
        """
        if debris.cx.numel() == 0:
            return None, torch.empty(0), torch.empty(0)
            
        N = debris.cx.shape[0]
        if debris.cx.device != self.device:
            debris.to(self.device)
            
        t_cxs, t_cys = debris.cx, debris.cy
        t_zs = debris.z
        t_sizes = debris.size_px.float() # radius
        t_deltas = debris.delta
        t_angles = debris.angle_deg
        t_is_dash = debris.is_dash
        
        # Bounds
        pad = 2
        t_x_mins = torch.round(t_cxs - t_sizes).int()
        t_x_maxs = torch.round(t_cxs + t_sizes + 1.0).int()
        t_y_mins = torch.round(t_cys - t_sizes).int()
        t_y_maxs = torch.round(t_cys + t_sizes + 1.0).int()
        
        t_ws = t_x_maxs - t_x_mins
        t_hs = t_y_maxs - t_y_mins
        
        max_w = min(t_ws.max().item(), 512)
        max_h = min(t_hs.max().item(), 512)
        
        if max_w <= 0 or max_h <= 0:
             return None, torch.empty(0), torch.empty(0)
             
        dev = self.device
        
        # Grid
        yy, xx = torch.meshgrid(
            torch.arange(max_h, device=dev, dtype=torch.float32),
            torch.arange(max_w, device=dev, dtype=torch.float32),
            indexing='ij'
        )
        xx = xx.unsqueeze(0).expand(N, -1, -1)
        yy = yy.unsqueeze(0).expand(N, -1, -1)
        
        cx_rel = t_cxs.view(N, 1, 1) - t_x_mins.view(N, 1, 1)
        cy_rel = t_cys.view(N, 1, 1) - t_y_mins.view(N, 1, 1)
        
        patch = torch.zeros((N, max_h, max_w), device=dev, dtype=torch.float32)
        
        # Dash Mask
        # We need to handle Dash and Circle separately or vectorized.
        # Vectorized: compute both, select based on is_dash
        
        # Circle
        dist2 = (xx - cx_rel)**2 + (yy - cy_rel)**2
        mask_circle = (dist2 <= t_sizes.view(N, 1, 1)**2).float()
        
        # Dash
        # Line segment logic
        # Ld = r
        # p1, p2
        ang_rad = torch.deg2rad(t_angles).view(N, 1, 1)
        dx, dy = torch.cos(ang_rad), torch.sin(ang_rad)
        Ld = t_sizes.view(N, 1, 1)
        
        p1x = cx_rel - Ld*dx
        p1y = cy_rel - Ld*dy
        p2x = cx_rel + Ld*dx
        p2y = cy_rel + Ld*dy
        
        abx, aby = p2x - p1x, p2y - p1y
        apx, apy = xx - p1x, yy - p1y
        
        len_sq = abx**2 + aby**2
        t_line = (apx * abx + apy * aby) / (len_sq + 1e-6)
        t_line = torch.clamp(t_line, 0.0, 1.0)
        
        cx_line = p1x + t_line * abx
        cy_line = p1y + t_line * aby
        
        dist_sq_line = (xx - cx_line)**2 + (yy - cy_line)**2
        dist_line = torch.sqrt(dist_sq_line)
        
        intensity_line = torch.clamp(1.0 - (dist_line - 0.5), 0.0, 1.0)
        
        # Select
        is_dash_broad = t_is_dash.view(N, 1, 1)
        patch = torch.where(is_dash_broad, intensity_line, mask_circle)
        
        patch *= t_deltas.view(N, 1, 1)
        
        # Blur
        ghost_sig = self.cfg.physics.ghosts.blur_sigma
        scale = max(0.1, ghost_sig) if ghost_sig > 0 else 2.0
        blur_sigs = torch.abs(t_zs) * scale
        
        if torch.any(blur_sigs > 0.5):
            patch = self._gaussian_blur_batch(patch, blur_sigs)
            
        patch_chw = patch.unsqueeze(1).repeat(1, 3, 1, 1)
        
        return patch_chw, t_x_mins, t_y_mins
