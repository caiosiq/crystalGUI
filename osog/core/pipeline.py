import random
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import concurrent.futures

from ..config import SynthConfig
from ..physics.distribution import generate_distribution
from ..physics.particles import Rod, Agglomerate, Debris, RodBatch, DebrisBatch
from ..physics.ghosts import GhostObject
# from ..optics.sensor import SensorHead # Legacy removed
from .canvas import Canvas

import torch
from ..optics.dic_torch import DICModulatorTorch
from ..optics.sensor_torch import SensorHeadTorch

class Pipeline:
    def __init__(self, config: Dict[str, Any]):
        # Use the centralized loader which handles both flat (legacy) and nested (new) structures
        self.cfg = SynthConfig.from_dict(config)

        self.dic_head = None # Legacy head removed
        
        # Determine device
        use_gpu_config = self.cfg.canvas.use_gpu
        
        if use_gpu_config and torch.cuda.is_available():
            self.device_name = "cuda"
        else:
            self.device_name = "cpu"
            if use_gpu_config:
                print("Warning: GPU Rendering requested but CUDA not available. Using CPU.")
        
        self.dic_head_torch = DICModulatorTorch(self.cfg, device=self.device_name)
        self.sensor_head_torch = SensorHeadTorch(self.cfg, device=self.device_name)

    def generate(self, t: float, seed: Optional[int] = None, return_obbs: bool = False, parallel_workers: Optional[int] = None) -> Any:
        t0 = time.time()
        
        # 1. Setup Randomness
        if seed is None:
            rng = random.Random()
            rng.seed(random.SystemRandom().randint(0, 2**31 - 1))
            np_rng = np.random.RandomState(random.SystemRandom().randint(0, 2**31 - 1))
        else:
            rng = random.Random(seed)
            np_rng = np.random.RandomState(seed)
            
        # Ensure device is passed
        device = self.dic_head_torch.device

        # 2. Physics Generation (The "Reactor")
        rod_batch, debris_batch, agglomerates_meta = generate_distribution(self.cfg, t, rng, np_rng, device=device)
        
        t1 = time.time()

        # 3. Initialize Canvas on Device (The "Sensor" Background)
        # Generate background directly as tensor
        seed_bg = rng.randint(0, 2**31 - 1)
        canvas_tensor = self.sensor_head_torch.apply_background(self.cfg.canvas.height, self.cfg.canvas.width, rng, seed_bg)
        
        # 4. Rendering (The "Optics")
        with torch.no_grad():
            self._render_batch_gpu(canvas_tensor, rod_batch, debris_batch, rng)
            
            # 5. Sensor Artifacts (Blur) on Tensor
            canvas_tensor = self.sensor_head_torch.apply_blur(canvas_tensor)

        t2 = time.time()
        
        if self.device_name == "cuda":
            print(f"[GPU Profile] Total: {t2-t0:.4f}s | Physics: {t1-t0:.4f}s | Render: {t2-t1:.4f}s")
        else:
            print(f"[CPU Profile] Total: {t2-t0:.4f}s | Physics: {t1-t0:.4f}s | Render: {t2-t1:.4f}s")

        # 6. Overlay and Export (Download to CPU)
        # This converts to numpy BGR uint8
        img_np = self.sensor_head_torch.apply_overlay_and_export(canvas_tensor, rng)
        
        # Update Canvas object for compatibility
        canvas = Canvas(self.cfg.canvas.width, self.cfg.canvas.height)
        canvas.set_image(img_np)

        obbs = []
        if return_obbs:
            # Reconstruct OBBs from RodBatch if needed
            if rod_batch.cx.numel() > 0:
                cx = rod_batch.cx.cpu().numpy()
                cy = rod_batch.cy.cpu().numpy()
                L = rod_batch.L.cpu().numpy()
                W = rod_batch.W.cpu().numpy()
                ang = rod_batch.angle_deg.cpu().numpy()
                req = rod_batch.requires_label.cpu().numpy()
                
                for i in range(len(cx)):
                    if req[i]:
                        r = Rod(cx[i], cy[i], L[i], W[i], ang[i], 0, 0)
                        obbs.append(self._obj_to_dict(r))

            return canvas.image, obbs
            
        return canvas.image

    def _stamp_tensor_batch(self, canvas: torch.Tensor, patches: torch.Tensor, x_mins: torch.Tensor, y_mins: torch.Tensor):
        """
        Vectorized 'pasting' of patches onto canvas.
        Replaces the slow Python for-loop with a single CUDA kernel launch.
        
        Args:
            canvas: (3, H, W)
            patches: (N, 3, H_p, W_p)
            x_mins, y_mins: (N,) coordinates
        """
        if patches.numel() == 0:
            return

        N, C, H_p, W_p = patches.shape
        H, W = canvas.shape[1], canvas.shape[2]

        # 1. Create a coordinate grid for a single patch
        # device=canvas.device ensures these stay on GPU
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H_p, device=canvas.device),
            torch.arange(W_p, device=canvas.device),
            indexing='ij'
        ) # (Hp, Wp)

        # 2. Broadcast to global coordinates for ALL patches at once
        # y_mins is (N,) -> view as (N, 1, 1) to broadcast against (Hp, Wp)
        # result gy is (N, Hp, Wp) containing absolute Y coordinates for every pixel of every patch
        gy = y_mins.view(N, 1, 1).long() + grid_y.view(1, H_p, W_p)
        gx = x_mins.view(N, 1, 1).long() + grid_x.view(1, H_p, W_p)

        # 3. Masking: Find which pixels are actually inside the canvas
        # This handles edge clipping automatically without 'if' statements
        mask = (gy >= 0) & (gy < H) & (gx >= 0) & (gx < W)

        # Optimization: If mask is empty (all objects off screen), return early
        if not mask.any():
            return

        # 4. The "Splat": Add valid pixels to canvas
        # We loop over channels (only 3 iters) to keep memory usage lower than expanding fully to (N,3,H,W)
        for c in range(C):
            # Extract valid coordinates and values using the boolean mask
            # These become 1D tensors of length = (total_valid_pixels)
            valid_y = gy[mask]
            valid_x = gx[mask]
            valid_vals = patches[:, c, :, :][mask]

            # Atomic Add (accumulate=True) handles overlaps correctly
            canvas[c].index_put_((valid_y, valid_x), valid_vals, accumulate=True)

    def _render_batch_gpu(self, canvas_tensor: torch.Tensor, rod_batch: RodBatch, debris_batch: DebrisBatch, rng):
        """
        Fast GPU rendering path using Batches.
        Modifies canvas_tensor in-place.
        """
        t_start = time.time()
        
        H, W = canvas_tensor.shape[1], canvas_tensor.shape[2]
        
        # 1. Render Rods Batch
        t_rods_start = time.time()
        if rod_batch.cx.numel() > 0:
            patches, x_mins, y_mins = self.dic_head_torch.render_rods_batch(rod_batch, rng, None)
            
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t_rods_calc = time.time()
            
            if patches is not None:
                self._stamp_tensor_batch(canvas_tensor, patches, x_mins, y_mins)
                    
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t_rods_end = time.time()
            print(f"  [GPU Detail] Rods: Calc {t_rods_calc - t_rods_start:.4f}s | Stamp {t_rods_end - t_rods_calc:.4f}s")
            
        # 2. Render Debris Batch
        t_deb_start = time.time()
        if debris_batch.cx.numel() > 0:
            d_patches, d_x, d_y = self.dic_head_torch.render_debris_batch(debris_batch, rng)
            
            if d_patches is not None:
                self._stamp_tensor_batch(canvas_tensor, d_patches, d_x, d_y)
                    
        t_deb_end = time.time()
        if debris_batch.cx.numel() > 0:
             print(f"  [GPU Detail] Debris: {t_deb_end - t_deb_start:.4f}s")

        t_end = time.time()
        # No more download/clamp here, handled in apply_overlay_and_export
        print(f"  [GPU Detail] Total Render: {t_end - t_start:.4f}s")

    def _obj_to_dict(self, obj: Any) -> Dict[str, Any]:
        if isinstance(obj, Rod):
            corners = obj.corners.tolist()
            return {
                "cx": float(obj.cx),
                "cy": float(obj.cy),
                "L": float(obj.L),
                "W": float(obj.W),
                "angle_deg": float(obj.angle_deg),
                "corners": corners
            }
        return {}
