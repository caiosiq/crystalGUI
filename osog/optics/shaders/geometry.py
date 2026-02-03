from typing import Tuple, Union, List, Dict, Optional
import torch
import torch.nn.functional as F
import random
import math
import numpy as np
from ...config import SynthConfig
from ...physics.particles import ParticleBatch, SHAPE_ROD, SHAPE_PLATE, SHAPE_CUBE, SHAPE_SPHERE, SHAPE_BUBBLE, SHAPE_DROPLET
from ...utils.math_torch import noise1d_like_batch, smooth_cap, sin_wobble_batch, kink_batch, noisy_wobble_batch
from ..utils import gaussian_blur_batch
from .texture import TextureShader

class GeometryShader:
    def __init__(self, config: SynthConfig, device: torch.device):
        self.cfg = config
        self.device = device
        self.texture_shader = TextureShader(config, device)

    def render_batch(self, batch: ParticleBatch, rng: random.Random) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Vectorized Geometry Pass.
        Generates the G-Buffer for a batch of particles.
        
        Returns: 
            g_buffer: (N, 4, H_patch, W_patch)
                Channel 0: Physical Height (microns)
                Channel 1: Mask (0.0 or 1.0)
                Channel 2: Material ID / RI / Delta (Encoded)
                Channel 3: Local Orientation (Radians)
            x_mins: (N,)
            y_mins: (N,)
            aux_dict: Additional data (z, etc.)
        """
        if batch.cx.numel() == 0:
            return None, torch.empty(0), torch.empty(0), {}
            
        N = len(batch)
        if batch.cx.device != self.device:
            batch.to(self.device)
            
        cfg = self.cfg
        dev = self.device
        
        # 2. 3D Projection (Analytic)
        beta_rad = torch.deg2rad(batch.beta)
        gamma_rad = torch.deg2rad(batch.gamma)
        
        cb, sb = torch.abs(torch.cos(beta_rad)), torch.abs(torch.sin(beta_rad))
        cg, sg = torch.abs(torch.cos(gamma_rad)), torch.abs(torch.sin(gamma_rad))
        
        # Projected Dimensions
        W_eff = batch.W * cg + batch.H * sg
        
        # For Rod/Sphere, override
        is_rod = (batch.shape_id == SHAPE_ROD)
        is_sphere = (batch.shape_id == SHAPE_SPHERE)
        is_bubble = (batch.shape_id == SHAPE_BUBBLE)
        is_droplet = (batch.shape_id == SHAPE_DROPLET)
        is_sphere_like = is_sphere | is_bubble | is_droplet
        
        W_eff = torch.where(is_rod | is_sphere_like, batch.W, W_eff)
        
        L_eff = batch.L * cb + batch.H * sb
        L_eff = torch.where(is_sphere_like, batch.L, L_eff) 
        
        # Effective Optical Thickness (Max path length)
        tilt_factor = torch.clamp(torch.max(cb, cg), 0.1, 1.0)
        H_eff = batch.H / tilt_factor
        H_eff = torch.where(is_sphere_like | is_rod, batch.H, H_eff)
        
        # 3. Bounding Box Calculation
        alpha_rad = torch.deg2rad(batch.alpha)
        ca, sa = torch.abs(torch.cos(alpha_rad)), torch.abs(torch.sin(alpha_rad))
        
        box_w = L_eff * ca + W_eff * sa
        box_h = L_eff * sa + W_eff * ca
        
        pad = max(6, int(max(cfg.optics.rod_halo_sigma, 3))) + 6
        
        t_x_mins = torch.floor(batch.cx - box_w / 2.0).int() - pad
        t_y_mins = torch.floor(batch.cy - box_h / 2.0).int() - pad
        
        x_maxs = torch.ceil(batch.cx + box_w / 2.0).int() + pad
        y_maxs = torch.ceil(batch.cy + box_h / 2.0).int() + pad
        
        t_ws = x_maxs - t_x_mins
        t_hs = y_maxs - t_y_mins
        
        # Unified grid
        max_w = min(t_ws.max().item(), 1024)
        max_h = min(t_hs.max().item(), 1024)
        
        # 4. Grid Generation
        yy, xx = torch.meshgrid(
            torch.arange(max_h, device=dev, dtype=torch.float32),
            torch.arange(max_w, device=dev, dtype=torch.float32),
            indexing='ij'
        )
        xx = xx.unsqueeze(0).expand(N, -1, -1)
        yy = yy.unsqueeze(0).expand(N, -1, -1)
        
        X = xx + t_x_mins.view(N, 1, 1) - batch.cx.view(N, 1, 1)
        Y = yy + t_y_mins.view(N, 1, 1) - batch.cy.view(N, 1, 1)
        
        # 5. Local Coordinates (u, v)
        th = torch.deg2rad(batch.alpha).view(N, 1, 1)
        ct, st = torch.cos(th), torch.sin(th)
        
        # Unnormalized Rotated Coordinates (aligned with alpha)
        # X_rot aligns with Length, Y_rot aligns with Width
        X_rot = (ct * X + st * Y)
        Y_rot = (-st * X + ct * Y)
        
        L_half = (L_eff.view(N, 1, 1) / 2.0) + 1e-6
        u = X_rot / L_half
        v = Y_rot
        
        # 6. SDF / Height Map Generation
        W_half = (W_eff.view(N, 1, 1) / 2.0) + 1e-6
        v_norm = v / W_half
        
        # --- Phase 4.4: 3D Box Rasterization (Exact Ray-Slab Intersection) ---
        is_box = (batch.shape_id == SHAPE_CUBE) | (batch.shape_id == SHAPE_PLATE)
        is_rod = (batch.shape_id == SHAPE_ROD)
        
        # Only compute detailed 3D physics if we have boxes OR rods
        if is_box.any() or is_rod.any():
            # Rotation Matrices components
            # We need M_inv = Rx(-gamma) * Ry(-beta) to transform World Ray to Box Frame
            # beta is rotation around Y (Pitch), gamma around X (Roll) relative to alpha-frame
            
            beta_rad = torch.deg2rad(batch.beta).view(N, 1, 1)
            gamma_rad = torch.deg2rad(batch.gamma).view(N, 1, 1)
            
            cb, sb = torch.cos(beta_rad), torch.sin(beta_rad)
            cg, sg = torch.cos(gamma_rad), torch.sin(gamma_rad)
            
            # Ray Direction in Box/Rod Frame: D_local = M_inv * (0, 0, 1)^T
            # Column 2 of M_inv
            dx_loc = -sb
            dy_local = sg * cb
            dz_local = cg * cb
            
            # Ray Origin in Box/Rod Frame (at Z=0): O_local = M_inv * (X_rot, Y_rot, 0)^T
            # Col 0 * X + Col 1 * Y
            ox_loc = cb * X_rot
            oy_loc = (sg * sb) * X_rot + cg * Y_rot
            oz_loc = (cg * sb) * X_rot - sg * Y_rot
            
            # Dimensions (Half-sizes)
            lx = batch.L.view(N, 1, 1) * 0.5
            ly = batch.W.view(N, 1, 1) * 0.5
            lz = batch.H.view(N, 1, 1) * 0.5
            
            # --- A. BOX INTERSECTION ---
            # Initialize with empty/zeros
            thickness_box = torch.zeros_like(X_rot)
            
            if is_box.any():
                # Avoid div by zero
                epsilon = 1e-6
                dx_b = torch.where(torch.abs(dx_loc) < epsilon, torch.tensor(epsilon, device=dev), dx_loc)
                dy_b = torch.where(torch.abs(dy_local) < epsilon, torch.tensor(epsilon, device=dev), dy_local)
                dz_b = torch.where(torch.abs(dz_local) < epsilon, torch.tensor(epsilon, device=dev), dz_local)
                
                # X Slabs
                t1x = (-lx - ox_loc) / dx_b
                t2x = (lx - ox_loc) / dx_b
                t_min_x = torch.min(t1x, t2x)
                t_max_x = torch.max(t1x, t2x)
                
                # Y Slabs
                t1y = (-ly - oy_loc) / dy_b
                t2y = (ly - oy_loc) / dy_b
                t_min_y = torch.min(t1y, t2y)
                t_max_y = torch.max(t1y, t2y)
                
                # Z Slabs
                t1z = (-lz - oz_loc) / dz_b
                t2z = (lz - oz_loc) / dz_b
                t_min_z = torch.min(t1z, t2z)
                t_max_z = torch.max(t1z, t2z)
                
                # Intersection
                t_enter = torch.maximum(torch.maximum(t_min_x, t_min_y), t_min_z)
                t_exit = torch.minimum(torch.minimum(t_max_x, t_max_y), t_max_z)
                
                thickness_box = torch.clamp(t_exit - t_enter, min=0.0)

            # --- B. ROD INTERSECTION (Cylinder along X-axis) ---
            # Equation: y^2 + z^2 = R^2 (R = ly = lz)
            # Ray: P(t) = O + tD
            # (Oy + t*Dy)^2 + (Oz + t*Dz)^2 = R^2
            # A*t^2 + B*t + C = 0
            
            thickness_rod = torch.zeros_like(X_rot)
            
            if is_rod.any():
                # R is half-width
                R = ly 
                
                # Quadratic Coeffs
                # A = Dy^2 + Dz^2
                # B = 2*(Oy*Dy + Oz*Dz)
                # C = Oy^2 + Oz^2 - R^2
                
                A = dy_local**2 + dz_local**2
                B = 2.0 * (oy_loc * dy_local + oz_loc * dz_local)
                C = oy_loc**2 + oz_loc**2 - R**2
                
                # Discriminant
                delta_sq = B**2 - 4.0 * A * C
                
                # Valid cylinder intersection if delta > 0
                # We use mask to avoid NaNs
                valid_cyl = (delta_sq > 0)
                
                sqrt_delta = torch.sqrt(torch.clamp(delta_sq, min=0.0))
                
                # Two intersections with infinite cylinder
                # t = (-B +/- sqrt(delta)) / (2A)
                # Avoid A=0 (ray parallel to axis)
                A_safe = torch.where(torch.abs(A) < 1e-6, torch.tensor(1.0, device=dev), A)
                
                t1_cyl = (-B - sqrt_delta) / (2.0 * A_safe)
                t2_cyl = (-B + sqrt_delta) / (2.0 * A_safe)
                
                t_min_cyl = torch.min(t1_cyl, t2_cyl)
                t_max_cyl = torch.max(t1_cyl, t2_cyl)
                
                # Clip by Length (X-slabs)
                # The rod ends at x = +/- lx
                # We reuse X-slab logic
                dx_r = torch.where(torch.abs(dx_loc) < 1e-6, torch.tensor(1e-6, device=dev), dx_loc)
                t1x = (-lx - ox_loc) / dx_r
                t2x = (lx - ox_loc) / dx_r
                t_min_x = torch.min(t1x, t2x)
                t_max_x = torch.max(t1x, t2x)
                
                # Intersection is overlap of Cylinder interval and X-slab interval
                t_enter = torch.maximum(t_min_cyl, t_min_x)
                t_exit = torch.minimum(t_max_cyl, t_max_x)
                
                # Check validity
                # Must be valid cylinder hit (delta>0) AND valid overlap (exit > enter)
                thickness_rod = torch.where(valid_cyl, torch.clamp(t_exit - t_enter, min=0.0), torch.zeros_like(X_rot))

        else:
            thickness_box = torch.zeros_like(X_rot)
            thickness_rod = torch.zeros_like(X_rot)

        # Texture Scaling
        tex_scale_u = L_eff.view(N, 1, 1) / 50.0
        tex_scale_v = W_eff.view(N, 1, 1) / 20.0
        
        # --- TEXTURE PASS (Phase 4.4.1.5) ---
        # Delegated to TextureShader
        roughness_map, transmission_map = self.texture_shader.generate_maps(
            batch, u, v_norm, tex_scale_u, tex_scale_v, max_h, max_w, rng
        )
        
        # --- B. Profile (Cross-Section) ---
        # ... (Rest of profile logic) ...
        is_rod = (batch.shape_id == SHAPE_ROD)
        is_sphere_like = (batch.shape_id == SHAPE_SPHERE) | (batch.shape_id == SHAPE_BUBBLE) | (batch.shape_id == SHAPE_DROPLET)
        is_round = is_rod | is_sphere_like
        is_round = is_round.view(N, 1, 1)
        
        is_cube = (batch.shape_id == SHAPE_CUBE).view(N, 1, 1)
        
        h_round = torch.sqrt(torch.clamp(1.0 - v_norm**2, 0.0, 1.0))
        h_flat = torch.where(torch.abs(v_norm) < 0.8, 1.0, 
                             torch.clamp((1.0 - torch.abs(v_norm)) / 0.2, 0.0, 1.0))
                             
        tumble_strength = torch.max(torch.abs(batch.beta), torch.abs(batch.gamma)).view(N, 1, 1) / 90.0
        tumble_strength = torch.clamp(tumble_strength, 0.0, 1.0)
        
        pyramid = 1.0 - torch.maximum(torch.abs(u), torch.abs(v_norm))
        pyramid = torch.clamp(pyramid, 0.0, 1.0)
        
        h_cube = h_flat * (1.0 - 0.6 * tumble_strength) + pyramid * (0.6 * tumble_strength)
        
        # REMOVED: h_round = h_round * tex_mod
        # REMOVED: h_flat = h_flat * tex_mod
        # REMOVED: h_cube = h_cube * tex_mod

        profile_v = torch.where(is_round, h_round, h_flat)
        profile_v = torch.where(is_cube, h_cube, profile_v)
        
        # --- C. Longitudinal Profile ---
        seed_tex = rng.randint(0, 2**31 - 1)
        break_roughness = noise1d_like_batch(v_norm * tex_scale_v, corr=0.2, amp=1.0, seed=seed_tex+1)
        u_limit = 1.0 - 0.15 * torch.abs(break_roughness)
        dist_to_edge = u_limit - torch.abs(u)
        sharpness_factor = (L_eff.view(N, 1, 1) * 0.5) + 1.0
        profile_u = torch.clamp(dist_to_edge * sharpness_factor, 0.0, 1.0) 
        
        h_u_sphere = torch.sqrt(torch.clamp(1.0 - u**2, 0.0, 1.0))
        
        is_cube_view = is_cube 
        if is_cube_view.any():
            u_flat = torch.where(torch.abs(u) < 0.8, 1.0, 
                             torch.clamp((1.0 - torch.abs(u)) / 0.2, 0.0, 1.0))
            profile_u = torch.where(is_cube_view, u_flat, profile_u)

        is_sphere_like_view = is_sphere_like.view(N, 1, 1)
        profile_u = torch.where(is_sphere_like_view, h_u_sphere, profile_u)

        # --- D. Shape Modes & Defects ---
        # NOTE: Shape modes currently only fully implemented for Rods.
        # Spheres/Cubes are mostly rigid, but we can apply some warping.
        
        shape_mode = batch.shape_mode.view(N, 1, 1)
        v_offset = torch.zeros_like(v)
        
        is_wavy = (shape_mode == 1)
        is_kink = (shape_mode == 2)
        is_noisy = (shape_mode == 3)
        
        # INCREASED AMPLITUDES FOR VISIBILITY
        u_scaled = u * tex_scale_u
        
        if is_wavy.any():
            # Sliding effect: Phase shift by random offset per particle
            phase_shift = batch.seed.view(N, 1, 1) % 100 * 0.1 
            amp = batch.W.view(N, 1, 1) * 0.5
            wobble = sin_wobble_batch(u_scaled + phase_shift)
            v_offset = torch.where(is_wavy, wobble * amp, v_offset)
            
        if is_kink.any():
            # Sliding effect for kinks
            kink_pos = (batch.seed.view(N, 1, 1) % 100 / 50.0) - 1.0 # -1 to 1
            amp = batch.W.view(N, 1, 1) * 1.0
            kink_off = kink_batch(u_scaled - kink_pos)
            v_offset = torch.where(is_kink, kink_off * amp, v_offset)
            
        if is_noisy.any():
            amp = batch.W.view(N, 1, 1) * 0.4
            noise_off = noisy_wobble_batch(u_scaled, corr=0.1)
            v_offset = torch.where(is_noisy, noise_off * amp, v_offset)

        if torch.any(batch.ragged_p > 0):
            rag_amp = batch.ragged_p.view(N, 1, 1) * batch.W.view(N, 1, 1) * 0.15
            rag_noise = noise1d_like_batch(u_scaled, corr=0.05)
            v_offset = v_offset + rag_noise * rag_amp
        
        # Apply the offset to v_norm (Coordinate Bending)
        v_norm_bent = (v - v_offset) / W_half
        
        # --- UNIVERSAL BENDING (All Shapes) ---
        # Previously only rods used v_norm_bent for profile calculation.
        # Now we apply it to everything.
        
        # Re-calculate profile height with bent coordinates
        h_round_bent = torch.sqrt(torch.clamp(1.0 - v_norm_bent**2, 0.0, 1.0))
        h_flat_bent = torch.where(torch.abs(v_norm_bent) < 0.8, 1.0, 
                             torch.clamp((1.0 - torch.abs(v_norm_bent)) / 0.2, 0.0, 1.0))
        
        # Cubes/Plates: Bend the top face?
        # For cubes, "bending" means the flat top face stays flat but moves in V.
        # h_flat_bent achieves this (it shifts the boundaries).
        # We also need to warp the 'pyramid' term for the bevels.
        pyramid_bent = 1.0 - torch.maximum(torch.abs(u), torch.abs(v_norm_bent))
        pyramid_bent = torch.clamp(pyramid_bent, 0.0, 1.0)
        
        h_cube_bent = h_flat_bent * (1.0 - 0.6 * tumble_strength) + pyramid_bent * (0.6 * tumble_strength)

        profile_v_bent = torch.where(is_round, h_round_bent, h_flat_bent)
        profile_v_bent = torch.where(is_cube, h_cube_bent, profile_v_bent)
        
        # Longitudinal Bending?
        # u is not modified, so length is straight.
        # But v is modified by u, so the object curves in the uv plane.
        # This is correct for "bending".
        
        # Spheres: 
        # r_sq = u**2 + v_norm_bent**2.
        # This distorts the circle into a bean shape or wobbly blob.
        
        if is_sphere_like_view.any():
             r_sq_bent = u**2 + v_norm_bent**2
             h_sphere_bent = torch.sqrt(torch.clamp(1.0 - r_sq_bent, 0.0, 1.0))
             height_map_bent = torch.where(is_sphere_like_view, h_sphere_bent, profile_v_bent * profile_u)
        else:
             height_map_bent = profile_v_bent * profile_u
             
        # ... (Rest of logic)
        
        # --- PHASE 4.4 Override ---
        # The 3D Ray Tracing (Box/Rod Intersection) currently IGNORES the bending!
        # It uses perfect cylinders/boxes.
        # To fix "Shape Mode" for Rods, we must fallback to the bent 2.5D profile 
        # when a shape deformation is active.
        
        has_deformation = (shape_mode > 0)
        
        # Use ray-traced physics by default...
        phys_height = H_eff.view(N, 1, 1) * height_map_bent
        
        # But if we have valid 3D intersection...
        if is_box.any():
             is_box_mask = is_box.view(N, 1, 1).expand(-1, max_h, max_w)
             # Fallback to bent 2.5D if deformed
             use_perfect_box = is_box_mask & (~has_deformation.view(N, 1, 1).expand(-1, max_h, max_w))
             phys_height = torch.where(use_perfect_box, thickness_box, phys_height)
             
        if is_rod.any():
             is_rod_mask = is_rod.view(N, 1, 1).expand(-1, max_h, max_w)
             # ONLY use perfect cylinder ray-tracing if NO deformation.
             # If deformed (wavy/kink), stick to the bent profile map (phys_height) calculated above.
             use_perfect_rod = is_rod_mask & (~has_deformation.view(N, 1, 1).expand(-1, max_h, max_w))
             
             phys_height = torch.where(use_perfect_rod, thickness_rod, phys_height)

        # --- G-Buffer Assembly ---
        
        # Channel 0: Physical Height
        g_height = phys_height
        
        # Channel 1: Mask (Binary)
        # Updated to check physical height directly
        g_mask = (phys_height > 0.001).float()
        
        # Channel 2: Refractive Index / Delta
        # We store 'delta' (RI difference) here.
        flip_mult = 1.0 - 2.0 * batch.polarity_flip_p.view(N, 1, 1)
        effective_delta = batch.delta.view(N, 1, 1) * flip_mult
        
        # Expand to grid size
        g_delta = effective_delta.expand(-1, max_h, max_w)
        # Apply mask
        g_delta = g_delta * g_mask
        
        # Channel 3: Local Orientation
        # For Rods/Plates: Global angle 'alpha'
        # For Spheres: Radial angle
        
        if is_sphere.any():
             local_ang = torch.atan2(Y, X)
             # If sphere, use local angle. Else use batch.alpha
             # This assumes we handle mixed batches carefully.
             # batch.alpha is (N,). local_ang is (N, H, W).
             batch_alpha_rad = torch.deg2rad(batch.alpha).view(N, 1, 1).expand(-1, max_h, max_w)
             
             is_sphere_map = is_sphere.view(N, 1, 1).expand(-1, max_h, max_w)
             g_orient = torch.where(is_sphere_map, local_ang, batch_alpha_rad)
        else:
             g_orient = torch.deg2rad(batch.alpha).view(N, 1, 1).expand(-1, max_h, max_w)
        
        g_orient = g_orient * g_mask
        
        # Stack G-Buffer: (N, 4, H, W)
        g_buffer = torch.stack([g_height, g_mask, g_delta, g_orient], dim=1)
        
        # Aux Dict
        aux_dict = {}
        # Depth
        z_map = batch.z.view(N, 1, 1).expand(-1, max_h, max_w)
        aux_dict['depth'] = z_map * g_mask
        
        # Opacity
        aux_dict['opacity'] = batch.opacity.view(N, 1, 1).expand(-1, max_h, max_w) * g_mask
        
        # Birefringence
        aux_dict['birefringence'] = batch.birefringence.view(N, 1, 1).expand(-1, max_h, max_w) * g_mask
        
        # Phase 4.3: Technicolor Props
        aux_dict['reflectivity'] = batch.reflectivity.view(N, 1, 1).expand(-1, max_h, max_w) * g_mask
        aux_dict['dispersion'] = batch.dispersion.view(N, 1, 1).expand(-1, max_h, max_w) * g_mask
        # Absorption Color is (N, 3). Expand to (N, 3, H, W)
        aux_dict['absorption_color'] = batch.absorption_color.view(N, 3, 1, 1).expand(-1, -1, max_h, max_w) * g_mask.unsqueeze(1)
        
        # Surface Roughness (for PVM)
        # Replaced scalar expansion with actual texture map
        aux_dict['surf_rough'] = roughness_map.expand(-1, max_h, max_w) * g_mask
        aux_dict['roughness_map'] = aux_dict['surf_rough'] # Alias for Phase 4.4.1.5
        
        # Transmission Map (for Phase 4.4.1.5)
        aux_dict['transmission_map'] = transmission_map.expand(-1, max_h, max_w) * g_mask
        
        # Shape ID (for specific shader overrides like bubbles)
        aux_dict['shape_id'] = batch.shape_id.view(N, 1, 1).expand(-1, max_h, max_w) * g_mask

        return g_buffer, t_x_mins, t_y_mins, aux_dict
