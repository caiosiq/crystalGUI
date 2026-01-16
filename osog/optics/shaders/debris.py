from typing import Tuple, Union, List, Dict, Optional
import torch
import torch.nn.functional as F
import random
import numpy as np
from ...config import SynthConfig
from ...physics.particles import DebrisBatch
from ..utils import gaussian_blur_batch

class DebrisShader:
    def __init__(self, config: SynthConfig, device: torch.device):
        self.cfg = config
        self.device = device

    def render_batch(self, debris: DebrisBatch, rng: random.Random, return_aux: bool = False) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Vectorized rendering for debris.
        """
        if debris.cx.numel() == 0:
            return None, torch.empty(0), torch.empty(0), {}
            
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
             return None, torch.empty(0), torch.empty(0), {}
             
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
            patch = gaussian_blur_batch(patch, blur_sigs)
            
        patch_chw = patch.unsqueeze(1).repeat(1, 3, 1, 1)
        
        aux_dict = {}
        if return_aux:
             mask_val = (patch > 0.001).float().unsqueeze(1)
             aux_dict['mask'] = mask_val
             aux_dict['height'] = (patch * 0.1).unsqueeze(1)
             z_map = t_zs.view(N, 1, 1).expand(-1, max_h, max_w)
             aux_dict['depth'] = (z_map * mask_val.squeeze(1)).unsqueeze(1)

        return patch_chw, t_x_mins, t_y_mins, aux_dict
