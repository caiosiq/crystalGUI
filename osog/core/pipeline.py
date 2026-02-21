
import random
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import torch

from ..config import SynthConfig
from ..physics.distribution import generate_distribution
from ..physics.particles import Rod, ParticleBatch, DebrisBatch
from ..optics.optical_engine import OpticalEngine
from ..optics.sensor_torch import SensorHeadTorch
from .canvas import Canvas

class Pipeline:
    def __init__(self, config: Union[Dict[str, Any], SynthConfig]):
        # Use the centralized loader which handles both flat (legacy) and nested (new) structures
        if isinstance(config, SynthConfig):
            self.cfg = config
        else:
            self.cfg = SynthConfig.from_dict(config)
        
        # Determine device
        use_gpu_config = self.cfg.canvas.use_gpu
        
        if use_gpu_config and torch.cuda.is_available():
            self.device_name = "cuda"
        else:
            self.device_name = "cpu"
            if use_gpu_config:
                print("Warning: GPU Rendering requested but CUDA not available. Using CPU.")
        
        self.optical_engine = OpticalEngine(self.cfg, device=self.device_name)
        self.sensor_head_torch = SensorHeadTorch(self.cfg, device=self.device_name)

    def generate(self, t: float, seed: Optional[int] = None, return_obbs: bool = False, parallel_workers: Optional[int] = None, return_heads: bool = False) -> Any:
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
        device = self.optical_engine.device

        # 2. Physics Generation (The "Reactor")
        # Generates batches of particles (Main + Ghosts + Clusters) and Debris
        particle_batch, debris_batch, agglomerates_meta = generate_distribution(self.cfg, t, rng, np_rng, device=device)
        
        t1 = time.time()

        # 3. Initialize Canvas on Device (The "Sensor" Background)
        # Generate background directly as tensor
        seed_bg = rng.randint(0, 2**31 - 1)
        canvas_tensor = self.sensor_head_torch.apply_background(self.cfg.canvas.height, self.cfg.canvas.width, rng, seed_bg)
        
        aux_canvases = None
        if return_heads:
            H, W = self.cfg.canvas.height, self.cfg.canvas.width
            # 1-channel auxiliary buffers
            aux_canvases = {
                'height': torch.zeros((1, H, W), device=device),
                'mask': torch.zeros((1, H, W), device=device),
                'depth': torch.zeros((1, H, W), device=device)
            }

        # 4. Rendering (The "Optics")
        with torch.no_grad():
            self._render_batch_gpu(canvas_tensor, particle_batch, debris_batch, rng, aux_canvases=aux_canvases)
            
            # 5. Sensor Artifacts (Blur, Chromatic Aberration) on Tensor
            canvas_tensor = self.sensor_head_torch.apply_blur(canvas_tensor)
            canvas_tensor = self.sensor_head_torch.apply_chromatic_aberration(canvas_tensor, strength=self.cfg.sensor.chromatic_aberration_strength)

        t2 = time.time()
        
        if self.device_name == "cuda":
            print(f"[GPU Profile] Total: {t2-t0:.4f}s | Physics: {t1-t0:.4f}s | Render: {t2-t1:.4f}s")
        else:
            print(f"[CPU Profile] Total: {t2-t0:.4f}s | Physics: {t1-t0:.4f}s | Render: {t2-t1:.4f}s")

        # 6. Overlay and Export (Download to CPU)
        # This converts to numpy BGR uint8
        # If canvas_tensor is 3-channel (RGB) due to polarization_rgb, apply_overlay_and_export needs to handle it.
        # SensorHeadTorch usually assumes (3, H, W) is BGR or grayscale duplicated.
        # But render_batch_gpu might have produced true RGB.
        
        # Check if we are in RGB mode
        # Legacy RGB modes removed (polarization_rgb, fluorescence, confocal)
        # Assuming BGR/Grayscale standard for now
        is_rgb = False
        
        img_np = self.sensor_head_torch.apply_overlay_and_export(canvas_tensor, rng, is_rgb=is_rgb)
        
        # Update Canvas object for compatibility
        canvas = Canvas(self.cfg.canvas.width, self.cfg.canvas.height)
        canvas.set_image(img_np)

        # Process heads
        heads = {}
        if return_heads and aux_canvases:
            for k, v in aux_canvases.items():
                heads[k] = v.squeeze(0).cpu().numpy()

        obbs = []
        if return_obbs:
            # Reconstruct OBBs from ParticleBatch if needed
            # Only done if requested for labeling
            if particle_batch.cx.numel() > 0:
                cx = particle_batch.cx.cpu().numpy()
                cy = particle_batch.cy.cpu().numpy()
                L = particle_batch.L.cpu().numpy()
                W = particle_batch.W.cpu().numpy()
                ang = particle_batch.alpha.cpu().numpy()
                req = particle_batch.requires_label.cpu().numpy()
                sid = particle_batch.shape_id.cpu().numpy()
                beta = particle_batch.beta.cpu().numpy()
                gamma = particle_batch.gamma.cpu().numpy()
                H = particle_batch.H.cpu().numpy()
                z = particle_batch.z.cpu().numpy()
                
                for i in range(len(cx)):
                    if req[i]:
                        # Use keyword arguments to ensure correct field assignment
                        # We use Rod class as generic container for now
                        r = Rod(
                            cx=float(cx[i]), 
                            cy=float(cy[i]), 
                            L=float(L[i]), 
                            W=float(W[i]), 
                            angle_deg=float(ang[i]), 
                            delta=0.0, 
                            seed=0,
                            z=float(z[i]),
                            requires_label=True
                        )
                        # Manually attach extra 3D properties
                        r.beta = float(beta[i])
                        r.gamma = float(gamma[i])
                        r.H = float(H[i])
                        r.shape_id = int(sid[i])
                        
                        obbs.append(self._obj_to_dict(r))
            
            if return_heads:
                return canvas.image, obbs, heads
            return canvas.image, obbs
        
        if return_heads:
            return canvas.image, heads
            
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

    def _render_batch_gpu(self, canvas_tensor: torch.Tensor, particle_batch: ParticleBatch, debris_batch: DebrisBatch, rng, aux_canvases: Optional[Dict[str, torch.Tensor]] = None):
        """
        Fast GPU rendering path using Batches.
        Modifies canvas_tensor in-place.
        """
        t_start = time.time()
        
        do_aux = (aux_canvases is not None)
        
        # 1. Render Main Particles Batch
        t_rods_start = time.time()
        if particle_batch.cx.numel() > 0:
            # 1a. Geometry Pass (G-Buffer)
            g_buffer, x_mins, y_mins, aux_r = self.optical_engine.geometry_shader.render_batch(particle_batch, rng)
            
            if g_buffer is not None:
                # 1b. Optical Pass (Primary Mode)
                mode = self.cfg.optics.mode
                patches = self.optical_engine.render_optics(g_buffer, aux_r, rng, mode=mode)
                
                if torch.cuda.is_available(): torch.cuda.synchronize()
                t_rods_calc = time.time()
                
                # Stamp Primary
                self._stamp_tensor_batch(canvas_tensor, patches, x_mins, y_mins)
                
                # Stamp Aux Heads
                if do_aux:
                    # 1. Standard G-Buffer Channels
                    if 'height' in aux_canvases:
                        h_patch = g_buffer[:, 0:1]
                        self._stamp_tensor_batch(aux_canvases['height'], h_patch, x_mins, y_mins)
                    if 'mask' in aux_canvases:
                        m_patch = g_buffer[:, 1:2]
                        self._stamp_tensor_batch(aux_canvases['mask'], m_patch, x_mins, y_mins)
                    if 'depth' in aux_canvases and 'depth' in aux_r:
                        d_patch = aux_r['depth'].unsqueeze(1)
                        self._stamp_tensor_batch(aux_canvases['depth'], d_patch, x_mins, y_mins)
                        
                    # 2. Extra Optical Modalities (Multi-Head)
                    # Check if any aux key is a valid optical mode (excluding current mode)
                    known_modes = ["dic", "brightfield", "pvm", "blaze"]
                    for aux_mode in known_modes:
                        if aux_mode in aux_canvases and aux_mode != mode:
                            # Render this modality from the SAME G-Buffer
                            # Note: rng state might advance if sim_XXX uses it. 
                            # Ideally we fork rng or reuse if deterministic is needed, but for noise it's fine.
                            aux_patch = self.optical_engine.render_optics(g_buffer, aux_r, rng, mode=aux_mode)
                            self._stamp_tensor_batch(aux_canvases[aux_mode], aux_patch, x_mins, y_mins)
                        
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t_rods_end = time.time()
            print(f"  [GPU Detail] Particles: Calc {t_rods_calc - t_rods_start:.4f}s | Stamp {t_rods_end - t_rods_calc:.4f}s")
            
        # 2. Render Debris Batch
        t_deb_start = time.time()
        if debris_batch.cx.numel() > 0:
            d_patches, d_x, d_y, aux_d = self.optical_engine.render_debris_batch(debris_batch, rng, return_aux=do_aux)
            
            if d_patches is not None:
                self._stamp_tensor_batch(canvas_tensor, d_patches, d_x, d_y)
                
                if do_aux and aux_d:
                    for k, v in aux_d.items():
                        if k in aux_canvases:
                            self._stamp_tensor_batch(aux_canvases[k], v, d_x, d_y)
                    
        t_deb_end = time.time()
        if debris_batch.cx.numel() > 0:
             print(f"  [GPU Detail] Debris: {t_deb_end - t_deb_start:.4f}s")

        t_end = time.time()
        print(f"  [GPU Detail] Total Render: {t_end - t_start:.4f}s")

    def _obj_to_dict(self, obj: Any) -> Dict[str, Any]:
        if isinstance(obj, Rod):
            corners = obj.corners.tolist()
            return {
                "cx": float(obj.cx),
                "cy": float(obj.cy),
                "z": float(obj.z),
                "L": float(obj.L),
                "W": float(obj.W),
                "H": float(obj.H) if hasattr(obj, 'H') else float(obj.W * 0.5),
                "angle_deg": float(obj.angle_deg),
                "beta": float(getattr(obj, 'beta', 0.0)),
                "gamma": float(getattr(obj, 'gamma', 0.0)),
                "shape_id": int(getattr(obj, 'shape_id', 0)),
                "corners": corners
            }
        return {}
