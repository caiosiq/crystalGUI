from typing import Tuple, Union, List, Dict, Optional
import torch
import torch.nn.functional as F
import random
import math
import numpy as np
import cv2
from ...config import SynthConfig
from ...physics.particles import ParticleBatch, SHAPE_ROD, SHAPE_PLATE, SHAPE_CUBE, SHAPE_SPHERE, SHAPE_BUBBLE, SHAPE_DROPLET, SHAPE_POLYHEDRA
from ...utils.math_torch import noise1d_like_batch, smooth_cap, sin_wobble_batch, kink_batch, noisy_wobble_batch
from ..utils import gaussian_blur_batch
from .texture import TextureShader

class GeometryShader:
    def __init__(self, config: SynthConfig, device: torch.device):
        self.cfg = config
        self.device = device
        self.texture_shader = TextureShader(config, device)

    def render_batch(self, batch: ParticleBatch, rng: random.Random, soft_edge_mode: bool = False) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Vectorized Geometry Pass.
        Generates the G-Buffer for a batch of particles.
        
        Args:
            soft_edge_mode: If True, uses Softplus/Sigmoid for edges to allow gradient flow 
                            outside the object boundaries (Differentiable Rendering).
        
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
            
        # DEBUG GRADIENT
        # print(f"DEBUG: batch.L grad: {batch.L.requires_grad}")
            
        N = len(batch)
        if batch.cx.device != self.device:
            batch.to(self.device)
            
        cfg = self.cfg
        dev = self.device
        
        # --- Differentiable Rendering Helpers ---
        # Used to smooth out hard clamps for gradients
        def soft_clamp_zero(x, beta=100.0):
            if soft_edge_mode:
                return F.softplus(x, beta=beta)
            return torch.clamp(x, min=0.0)

        def soft_clamp_unit(x, beta=100.0):
            if soft_edge_mode:
                # Approximate clamp(x, 0, 1) using softplus difference
                # clamp(x, 0, 1) ~ softplus(x) - softplus(x-1)
                return F.softplus(x, beta=beta) - F.softplus(x - 1.0, beta=beta)
            return torch.clamp(x, 0.0, 1.0)
            
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
        is_poly = (batch.shape_id == SHAPE_POLYHEDRA)
        is_sphere_like = is_sphere | is_bubble | is_droplet
        
        # Polyhedra: Assume spherical bounding box for safety (W = L = H = Size)
        # Or better: just treat like a cube/sphere for bounding box purposes.
        W_eff = torch.where(is_rod | is_sphere_like | is_poly, batch.W, W_eff)
        
        L_eff = batch.L * cb + batch.H * sb
        L_eff = torch.where(is_sphere_like | is_poly, batch.L, L_eff) 
        
        # Effective Optical Thickness (Max path length)
        tilt_factor = torch.clamp(torch.max(cb, cg), 0.1, 1.0)
        H_eff = batch.H / tilt_factor
        H_eff = torch.where(is_sphere_like | is_rod | is_poly, batch.H, H_eff)
        
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

        # Corner deformation params (applied last, after jitter/bending)
        corner_round_t = batch.corner_round.view(N, 1, 1)
        corner_bend_t = batch.corner_bend.view(N, 1, 1)
        is_angular_shape = (
            (batch.shape_id == SHAPE_CUBE)
            | (batch.shape_id == SHAPE_PLATE)
            | (batch.shape_id == SHAPE_ROD)
        ).view(N, 1, 1)
        angular_active = is_angular_shape & (
            (corner_round_t > 1e-6) | (corner_bend_t > 1e-6)
        )
        is_rod_pre = (batch.shape_id == SHAPE_ROD).view(N, 1, 1)
        
        # --- Phase 4.4: 3D Box Rasterization (Exact Ray-Slab Intersection) ---
        is_box = (batch.shape_id == SHAPE_CUBE) | (batch.shape_id == SHAPE_PLATE)
        is_rod = (batch.shape_id == SHAPE_ROD)
        is_poly = (batch.shape_id == SHAPE_POLYHEDRA)
        
        # Only compute detailed 3D physics if we have boxes OR rods OR polyhedra
        if is_box.any() or is_rod.any() or is_poly.any():
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
            
            # --- Phase 4.4.2.3.1: Micro-Topography (SDF Jitter) ---
            # Perturb ray origins to simulate rough/jagged surfaces for 3D shapes.
            # Driven by ragged_p (irregularity)
            ragged_amp = batch.ragged_p.view(N, 1, 1)
            
            if torch.any(ragged_amp > 0):
                 # Simple 2D Hash Noise
                 # Scale V to match U scale (approx microns)
                 v_scaled = v_norm * batch.W.view(N, 1, 1)
                 
                 # Seed offset per particle
                 p_seed = batch.seed.view(N, 1, 1).float()
                 
                 # Hash
                 dt = X_rot * 12.9898 + Y_rot * 78.233 + p_seed
                 noise_val = torch.frac(torch.sin(dt) * 43758.5453)
                 noise_val = (noise_val - 0.5) * 2.0 # -1 to 1
                 
                 # Jitter amount (in microns)
                 # 2.0 microns max jitter for ragged=1.0
                 jitter = noise_val * ragged_amp * 2.0 
                 
                 # Apply to Ray Origins (Effective spatial distortion)
                 ox_loc = ox_loc + jitter
                 oy_loc = oy_loc + jitter
                 oz_loc = oz_loc + jitter
            
            # Dimensions (Half-sizes)
            lx = batch.L.view(N, 1, 1) * 0.5
            ly = batch.W.view(N, 1, 1) * 0.5
            lz = batch.H.view(N, 1, 1) * 0.5
            
            # Constants
            epsilon = 1e-6
            
            # --- A. BOX INTERSECTION ---
            # Initialize with empty/zeros
            thickness_box = torch.zeros_like(X_rot)
            
            if is_box.any():
                # Avoid div by zero
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
                
                if soft_edge_mode:
                    # Soft clamp for gradients
                    thickness_box = soft_clamp_zero(t_exit - t_enter)
                else:
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
                
                if soft_edge_mode:
                     # Soft mask for cylinder validity?
                     # delta_sq is continuous. if delta_sq < 0, sqrt is NaN.
                     # We must use softplus(delta_sq) to avoid NaN and keep gradient?
                     # sqrt(softplus(delta_sq))
                     
                     sqrt_delta_soft = torch.sqrt(F.softplus(delta_sq, beta=20.0) + 1e-6)
                     
                     # Recalculate t1/t2 with soft delta
                     t1_cyl_s = (-B - sqrt_delta_soft) / (2.0 * A_safe)
                     t2_cyl_s = (-B + sqrt_delta_soft) / (2.0 * A_safe)
                     
                     t_min_cyl_s = torch.min(t1_cyl_s, t2_cyl_s)
                     t_max_cyl_s = torch.max(t1_cyl_s, t2_cyl_s)
                     
                     t_enter_s = torch.maximum(t_min_cyl_s, t_min_x)
                     t_exit_s = torch.minimum(t_max_cyl_s, t_max_x)
                     
                     # Valid overlap
                     raw_thick = t_exit_s - t_enter_s
                     thickness_rod = soft_clamp_zero(raw_thick)
                     
                     # Also need to fade out if missed cylinder?
                     # If delta_sq < 0, softplus is small pos -> small thickness.
                     # But physically it missed.
                     # This is fine for optimization (it pulls ray towards cylinder).
                     
                else:
                     thickness_rod = torch.where(valid_cyl, torch.clamp(t_exit - t_enter, min=0.0), torch.zeros_like(X_rot))

            # --- C. POLYHEDRA INTERSECTION (Procedural Planes) ---
            thickness_poly = torch.zeros_like(X_rot)
            
            if is_poly.any():
                # Procedural Plane Sculpting (Phase 4.4.1.7)
                
                # We need random planes defined in the LOCAL frame of the particle.
                # Since we already transformed the ray to the local frame (O_local, D_local),
                # we just need to define planes relative to (0,0,0).
                
                # Num Planes: Stored in batch.curv (Hack from generator)
                # Irregularity: Stored in batch.rag_p (Hack from generator)
                # Seed: batch.seed
                
                # We need to generate M planes for each particle.
                # Since we can't loop over N particles easily, we use a fixed max planes loop (e.g. 12).
                
                MAX_PLANES = 12
                
                # Initialize ceilings and floors
                # Start with a base bounding box (cube of size L) to ensure closure?
                # Or just start with +/- infinity.
                # Let's start with a base sphere or box to guarantee it's closed.
                # Actually, simpler: Initialize with a large bounding box.
                
                big_val = 1e5
                z_ceil = torch.full_like(X_rot, big_val)
                z_floor = torch.full_like(X_rot, -big_val)
                
                # We also need to clip by the standard X/Y/Z bounds (the size L, W, H)
                # This acts as the "block" from which we carve.
                # Use the Box Logic bounds as the starting canvas.
                
                dx_p = torch.where(torch.abs(dx_loc) < epsilon, torch.tensor(epsilon, device=dev), dx_loc)
                dy_p = torch.where(torch.abs(dy_local) < epsilon, torch.tensor(epsilon, device=dev), dy_local)
                dz_p = torch.where(torch.abs(dz_local) < epsilon, torch.tensor(epsilon, device=dev), dz_local)

                # Initial Box Bounds (Reuse logic)
                t1x = (-lx - ox_loc) / dx_p; t2x = (lx - ox_loc) / dx_p
                t1y = (-ly - oy_loc) / dy_p; t2y = (ly - oy_loc) / dy_p
                t1z = (-lz - oz_loc) / dz_p; t2z = (lz - oz_loc) / dz_p
                
                t_enter_box = torch.maximum(torch.maximum(torch.min(t1x, t2x), torch.min(t1y, t2y)), torch.min(t1z, t2z))
                t_exit_box = torch.minimum(torch.minimum(torch.max(t1x, t2x), torch.max(t1y, t2y)), torch.max(t1z, t2z))
                
                # Active range
                z_floor = torch.maximum(z_floor, t_enter_box)
                z_ceil = torch.minimum(z_ceil, t_exit_box)

                # Seed expansion for determinism
                
                num_planes_target = batch.curvature.view(N, 1, 1) # Stored here
                irregularity = batch.ragged_p.view(N, 1, 1) # Stored here
                
                for i in range(MAX_PLANES):
                    # Pseudo-random generation
                    # distinct seed for each plane iteration
                    p_seed = batch.seed.view(N, 1, 1) + i * 1337
                    
                    # Random Normal (nx, ny, nz)
                    # We want them somewhat distributed.
                    # Simple approach: Random spherical coords
                    
                    # Hash to 0-1
                    h1 = torch.frac(torch.sin(p_seed * 12.9898) * 43758.5453)
                    h2 = torch.frac(torch.sin(p_seed * 78.233) * 43758.5453)
                    
                    phi = h1 * 2.0 * math.pi
                    costheta = 2.0 * h2 - 1.0
                    sintheta = torch.sqrt(torch.clamp(1.0 - costheta**2, 0.0, 1.0))
                    
                    nx = sintheta * torch.cos(phi)
                    ny = sintheta * torch.sin(phi)
                    nz = costheta
                    
                    # Random Distance D from center
                    # If D is small, plane cuts deep. If D is large, plane barely grazes.
                    # We want faces to form a crystal. D ~ Size/2.
                    # Base distance = min_dim / 2
                    
                    min_dim = torch.min(batch.L, torch.min(batch.W, batch.H)).view(N, 1, 1)
                    
                    h3 = torch.frac(torch.sin(p_seed * 93.123) * 43758.5453)
                    # Dist varies from 0.3*Size to 0.5*Size based on irregularity?
                    # Actually for convex hull, planes should be roughly at radius R.
                    
                    dist_val = (min_dim * 0.4) + (min_dim * 0.4) * h3 * (0.5 + 0.5 * irregularity)
                    
                    # Plane Equation: nx*X + ny*Y + nz*Z + D = 0
                    # Ray: P = O + t*D_ray
                    # nx*(Ox + t*Dx) + ny*(Oy + t*Dy) + nz*(Oz + t*Dz) + dist = 0
                    # t * (nx*Dx + ny*Dy + nz*Dz) = -(nx*Ox + ny*Oy + nz*Oz + dist)
                    # t = - (N dot O + dist) / (N dot D_ray)
                    
                    denom = nx * dx_loc + ny * dy_local + nz * dz_local
                    numer = -(nx * ox_loc + ny * oy_loc + nz * oz_loc + dist_val)
                    
                    # Avoid division by zero (ray parallel to plane)
                    # If parallel, check if Origin is "inside" (numer > 0?)
                    # If denom ~ 0, t is infinite.
                    
                    denom_safe = torch.where(torch.abs(denom) < 1e-6, torch.tensor(1e-6, device=dev), denom)
                    t_plane = numer / denom_safe
                    
                    
                    # Apply only if i < num_planes
                    mask_plane = (i < num_planes_target).expand(-1, max_h, max_w)
                    
                    is_exit = (denom < 0)
                    is_enter = (denom > 0)
                    
                    # Update bounds
                    # If plane is not active, don't change z_ceil/z_floor
                    
                    z_ceil = torch.where(mask_plane & is_exit, torch.minimum(z_ceil, t_plane), z_ceil)
                    z_floor = torch.where(mask_plane & is_enter, torch.maximum(z_floor, t_plane), z_floor)
                
                # Final thickness
                if soft_edge_mode:
                    thickness_poly = soft_clamp_zero(z_ceil - z_floor)
                else:
                    thickness_poly = torch.clamp(z_ceil - z_floor, min=0.0)
                
                # Zero out if not poly
                thickness_poly = torch.where(is_poly.view(N, 1, 1).expand(-1, max_h, max_w), thickness_poly, torch.zeros_like(thickness_poly))

        else:
            thickness_box = torch.zeros_like(X_rot)
            thickness_rod = torch.zeros_like(X_rot)
            thickness_poly = torch.zeros_like(X_rot)

        # Texture Scaling
        tex_scale_u = L_eff.view(N, 1, 1) / 50.0
        tex_scale_v = W_eff.view(N, 1, 1) / 20.0
        
        # --- TEXTURE PASS (Phase 4.4.1.5) ---
        # Delegated to TextureShader
        roughness_map, transmission_map, turbidity_map = self.texture_shader.generate_maps(
            batch, u, v_norm, tex_scale_u, tex_scale_v, max_h, max_w, rng
        )
        
        # --- B. Profile (Cross-Section) ---
        # ... (Rest of profile logic) ...
        is_rod = (batch.shape_id == SHAPE_ROD)
        is_sphere_like = (batch.shape_id == SHAPE_SPHERE) | (batch.shape_id == SHAPE_BUBBLE) | (batch.shape_id == SHAPE_DROPLET)
        is_round = is_rod | is_sphere_like
        is_round = is_round.view(N, 1, 1)
        
        is_cube = (batch.shape_id == SHAPE_CUBE).view(N, 1, 1)
        is_plate = (batch.shape_id == SHAPE_PLATE).view(N, 1, 1)
        is_flat_angular = is_cube | is_plate

        def _gaussian_edge(dist_inside, sigma):
            """Smooth interior→exterior falloff (tapered Gaussian / erf-like)."""
            sigma = torch.clamp(sigma, min=1e-4)
            outside = torch.relu(-dist_inside)
            tail = torch.exp(-0.5 * (outside / sigma) ** 2)
            inside = soft_clamp_unit(dist_inside / sigma)
            return inside * tail

        def _smooth_box_footprint(abs_u, abs_v, cr, cb):
            """Rounded-rectangle footprint with Gaussian-soft edges (no noise)."""
            cr = torch.clamp(cr, 0.0, 1.0)
            cb = torch.clamp(cb, 0.0, 1.0)
            r_eff = 0.04 + cr * 0.40 + cb * 0.22
            blur = 0.05 + cr * 0.22 + cb * 0.10

            qx = abs_u - (1.0 - r_eff)
            qy = abs_v - (1.0 - r_eff)
            bx = torch.clamp(qx, min=0.0)
            by = torch.clamp(qy, min=0.0)
            outside_corner = torch.sqrt(bx ** 2 + by ** 2 + 1e-8) - r_eff

            inside_x = (1.0 - r_eff) - abs_u
            inside_y = (1.0 - r_eff) - abs_v
            inside = torch.minimum(inside_x, inside_y)

            sdf = torch.where((qx > 0) & (qy > 0), outside_corner, -inside)
            return _gaussian_edge(-sdf, blur)

        def _smooth_cap_profile(abs_u, cr, cb):
            """Gaussian-softened rod/plate end cap along u."""
            cr = torch.clamp(cr, 0.0, 1.0)
            cb = torch.clamp(cb, 0.0, 1.0)
            sigma = 0.08 + cr * 0.32 + cb * 0.14
            dist_in = 1.0 - abs_u
            return _gaussian_edge(dist_in, sigma)

        def _smooth_rod_cross_section(abs_v, cr, cb):
            """Gaussian-softened semi-cylindrical cross-section."""
            cr = torch.clamp(cr, 0.0, 1.0)
            cb = torch.clamp(cb, 0.0, 1.0)
            sigma = 0.07 + cr * 0.28 + cb * 0.12
            dist_in = 1.0 - abs_v
            return _gaussian_edge(dist_in, sigma)
        
        if soft_edge_mode:
            # Soft Profile for Sphere/Blob
            # h = sqrt(softplus(1 - v^2))
            h_round = torch.sqrt(soft_clamp_zero(1.0 - v_norm**2) + 1e-6)
            
            # h_flat = soft step
            # standard: where(|v|<0.8, 1, linear_decay)
            # soft: sigmoid
            # We want flat top then decay.
            # Sigmoid( (0.8 - |v|) * huge )?
            # Actually, let's keep the linear ramp but use soft_clamp
            ramp = (1.0 - torch.abs(v_norm)) / 0.2
            h_flat = torch.where(torch.abs(v_norm) < 0.8, torch.tensor(1.0, device=dev), 
                                 soft_clamp_unit(ramp))
        else:
            h_round = torch.sqrt(torch.clamp(1.0 - v_norm**2, 0.0, 1.0))
            h_flat = torch.where(torch.abs(v_norm) < 0.8, 1.0, 
                                 torch.clamp((1.0 - torch.abs(v_norm)) / 0.2, 0.0, 1.0))
                             
        tumble_strength = torch.max(torch.abs(batch.beta), torch.abs(batch.gamma)).view(N, 1, 1) / 90.0
        tumble_strength = torch.clamp(tumble_strength, 0.0, 1.0)
        
        if soft_edge_mode:
             pyramid = soft_clamp_unit(1.0 - torch.maximum(torch.abs(u), torch.abs(v_norm)))
        else:
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
        
        if soft_edge_mode:
            profile_u = soft_clamp_unit(dist_to_edge * sharpness_factor)
        else:
            profile_u = torch.clamp(dist_to_edge * sharpness_factor, 0.0, 1.0) 
        
        if soft_edge_mode:
            h_u_sphere = torch.sqrt(soft_clamp_zero(1.0 - u**2) + 1e-6)
        else:
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

        # --- Geometric Jitter (Phase 4.4.2) ---
        # Apply explicit jitters if configured
        
        # 1. Offset Jitter (Low Freq Wobble)
        if torch.any(batch.offset_jit_amp > 0):
             off_amp = batch.offset_jit_amp.view(N, 1, 1) * batch.W.view(N, 1, 1)
             # Low freq noise
             off_noise = noise1d_like_batch(u_scaled * 0.5, corr=0.8, seed=seed_tex + 100)
             v_offset = v_offset + off_noise * off_amp
             
        # 2. Edge Jitter (High Freq Roughness)
        if torch.any(batch.edge_jit_amp > 0):
             edge_amp = batch.edge_jit_amp.view(N, 1, 1) * batch.W.view(N, 1, 1) * 0.2
             # High freq noise
             edge_noise = noise1d_like_batch(u_scaled * 5.0, corr=0.1, seed=seed_tex + 200)
             v_offset = v_offset + edge_noise * edge_amp

        if torch.any(batch.ragged_p > 0):
            rag_amp = batch.ragged_p.view(N, 1, 1) * batch.W.view(N, 1, 1) * 0.15
            rag_noise = noise1d_like_batch(u_scaled, corr=0.05)
            v_offset = v_offset + rag_noise * rag_amp
        
        # 3. Width Jitter (Thickness Variation)
        width_mod = torch.ones_like(u)
        if torch.any(batch.width_jit_amp > 0):
             w_amp = batch.width_jit_amp.view(N, 1, 1)
             # Mid freq noise
             w_noise = noise1d_like_batch(u_scaled, corr=0.5, seed=seed_tex + 300)
             # Modulate around 1.0. e.g. 1.0 +/- 0.2
             width_mod = 1.0 + w_noise * w_amp
             width_mod = torch.clamp(width_mod, 0.1, 3.0)

         # Apply the offset to v_norm (Coordinate Bending)
        # v_norm_bent = (v - v_offset) / (W_half * width_mod)
        v_norm_bent = (v - v_offset) / (W_half * width_mod)
        
        # --- UNIVERSAL BENDING (All Shapes) ---
        # Previously only rods used v_norm_bent for profile calculation.
        # Now we apply it to everything.
        
        # Re-calculate profile height with bent coordinates
        h_round_bent = torch.sqrt(torch.clamp(1.0 - v_norm_bent**2, 0.0, 1.0))
        h_flat_bent = torch.where(torch.abs(v_norm_bent) < 0.8, 1.0, 
                             torch.clamp((1.0 - torch.abs(v_norm_bent)) / 0.2, 0.0, 1.0))
        
        # Cubes/Plates: Bend the top face?
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

        # --- Corner deformations (final pass, after jitter / longitudinal bending) ---
        if torch.any(angular_active):
            u_fin = u
            v_fin = v_norm_bent
            au_fin = torch.abs(u_fin)
            av_fin = torch.abs(v_fin)
            cr = corner_round_t * angular_active.float()
            cb = corner_bend_t * angular_active.float()

            box_flat = _smooth_box_footprint(au_fin, av_fin, cr, cb)
            box_pyramid = _smooth_box_footprint(au_fin, av_fin, cr * 0.65 + 0.15, cb * 0.65)
            h_cube_v = box_flat * (1.0 - 0.6 * tumble_strength) + box_pyramid * (0.6 * tumble_strength)

            h_plate_v = _smooth_cap_profile(av_fin, cr, cb)
            h_rod_v = _smooth_rod_cross_section(av_fin, cr, cb)
            profile_v_final = torch.where(is_rod_pre, h_rod_v, h_plate_v)
            profile_v_final = torch.where(is_cube, h_cube_v, profile_v_final)

            profile_u_final = _smooth_cap_profile(au_fin, cr, cb)

            height_map_corner = profile_v_final * profile_u_final
            height_map_bent = torch.where(angular_active, height_map_corner, height_map_bent)
             
        # ... (Rest of logic)
        
        # --- PHASE 4.4 Override ---
        # The 3D Ray Tracing (Box/Rod Intersection) currently IGNORES the bending!
        # It uses perfect cylinders/boxes.
        # To fix "Shape Mode" for Rods, we must fallback to the bent 2.5D profile 
        # when a shape deformation is active.
        
        has_deformation = (shape_mode > 0) | (
            angular_active & ((corner_round_t > 1e-6) | (corner_bend_t > 1e-6))
        )
        
        # Use ray-traced physics by default...
        phys_height = H_eff.view(N, 1, 1) * height_map_bent
        
        # But if we have valid 3D intersection...
        if is_box.any():
             is_box_mask = is_box.view(N, 1, 1).expand(-1, max_h, max_w)
             # Fallback to bent 2.5D if deformed
             use_perfect_box = is_box_mask & (~has_deformation.view(N, 1, 1).expand(-1, max_h, max_w))
             
             if soft_edge_mode:
                 # In soft mode, we prefer the Profile approximation (phys_height)
                 # because it has smooth spatial gradients for L/W via u/v coordinates.
                 # Ray tracing (thickness_box) has sharp boundaries (dx=0 issues).
                 pass 
             else:
                 phys_height = torch.where(use_perfect_box, thickness_box, phys_height)
             
        if is_rod.any():
             is_rod_mask = is_rod.view(N, 1, 1).expand(-1, max_h, max_w)
             # ONLY use perfect cylinder ray-tracing if NO deformation.
             # If deformed (wavy/kink), stick to the bent profile map (phys_height) calculated above.
             use_perfect_rod = is_rod_mask & (~has_deformation.view(N, 1, 1).expand(-1, max_h, max_w))
             
             if soft_edge_mode:
                 # Force profile mode for gradients
                 pass
             else:
                 phys_height = torch.where(use_perfect_rod, thickness_rod, phys_height)
             
        if is_poly.any():
             is_poly_mask = is_poly.view(N, 1, 1).expand(-1, max_h, max_w)
             phys_height = torch.where(is_poly_mask, thickness_poly, phys_height)

        # --- G-Buffer Assembly ---
        
        # Channel 0: Physical Height
        g_height = phys_height
        
        # Channel 1: Mask (Binary)
        if soft_edge_mode:
             # Soft Mask: Sigmoid based on height or optical path
             # If height > 0, mask -> 1.
             # We want a smooth transition at height=0.
             # sigmoid(height * temp)
             # Center at 0.001?
             # If height is negative (from softplus), we want mask -> 0.
             
             # Note: phys_height comes from soft_clamp_zero(thickness), so it's always positive.
             # But if it's very small, mask should be small.
             
             # However, soft_clamp_zero(x) -> log(1+exp(x)).
             # If x is largely negative (missed by far), height ~ 0.
             # If x is near 0, height ~ log(2).
             # Wait, softplus(0) = 0.69. That's huge!
             # We need shifted softplus for thickness? 
             # No, softplus is standard for ReLU approximation.
             # But ReLU(0) = 0. Softplus(0) != 0.
             # This means "grazing" rays will have thickness 0.69 microns!
             # This might be an issue.
             # Let's use F.softplus(x, beta) - F.softplus(0, beta) ??
             # Or just use large beta so softplus(0) is small?
             # Beta=50 -> softplus(0) = log(1+1)/50 = 0.69/50 = 0.014 microns.
             # That's acceptable.
             
             # Mask:
             # Sigmoid( (height - threshold) * temp )
             # If height = 0.014 (edge), we want mask ~ 0.5?
             # threshold = 0.01
             # temp = 100
             
             g_mask = torch.sigmoid((g_height - 0.01) * 10.0)
             
        else:
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
        aux_dict['is_ghost'] = batch.group_id == -1
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
        
        # Surface Roughness
        # Replaced scalar expansion with actual texture map
        aux_dict['surf_rough'] = roughness_map.expand(-1, max_h, max_w) * g_mask
        aux_dict['roughness_map'] = aux_dict['surf_rough'] # Alias for Phase 4.4.1.5
        
        # Transmission Map (for Phase 4.4.1.5)
        aux_dict['transmission_map'] = transmission_map.expand(-1, max_h, max_w) * g_mask
        
        # Phase 4.4.2.1: Turbidity (Volumetric Fog)
        # Now generated by TextureShader as a 3D Volumetric Map
        aux_dict['turbidity'] = turbidity_map * g_mask

        # Shape ID (for specific shader overrides like bubbles)
        aux_dict['shape_id'] = batch.shape_id.view(N, 1, 1).expand(-1, max_h, max_w) * g_mask

        # --- Phase 4.4.2: OBB Refinement for Polyhedra ---
        # The initial OBB for polyhedra is a loose bounding sphere (Size S).
        # We now have the exact 2D mask (g_mask). We can compute the Tight OBB 
        # in the particle's local frame (aligned with alpha) to give precise labels.
        
        if is_poly.any():
            # 1. Identify valid pixels for Polyhedra
            # OPTICAL EROSION: Only consider pixels with significant physical thickness.
            # Thin edges (< 0.5 microns) are practically invisible and should not drive the OBB.
            VISIBLE_THRESHOLD = 0.5 
            
            # g_height is channel 0 of g_buffer
            poly_mask = (torch.abs(g_height) > VISIBLE_THRESHOLD) & is_poly.view(N, 1, 1).expand(-1, max_h, max_w)
            
            # Check which particles have any pixels rendered
            has_pixels = poly_mask.any(dim=-1).any(dim=-1) # (N,)
            
            # Only update those that have pixels (avoid NaN/Inf)
            valid_indices = torch.nonzero(has_pixels & is_poly).squeeze(1)
            
            if valid_indices.numel() > 0:
                # Extract relevant slices to save memory/compute
                # (But here we are already using full tensors, so just masking is fine)
                
                # We need X_rot and Y_rot (Local coordinates aligned with Alpha)
                # min_u, max_u, min_v, max_v
                
                # Mask out invalid pixels with huge values
                INF = 1e9
                
                # Expand mask for broadcasting if needed (already (N, H, W))
                m = poly_mask
                
                # U (Length axis)
                u_masked_min = torch.where(m, X_rot, torch.tensor(INF, device=dev))
                u_masked_max = torch.where(m, X_rot, torch.tensor(-INF, device=dev))
                
                min_u = u_masked_min.amin(dim=(1, 2)) # (N,)
                max_u = u_masked_max.amax(dim=(1, 2)) # (N,)
                
                # V (Width axis)
                v_masked_min = torch.where(m, Y_rot, torch.tensor(INF, device=dev))
                v_masked_max = torch.where(m, Y_rot, torch.tensor(-INF, device=dev))
                
                min_v = v_masked_min.amin(dim=(1, 2))
                max_v = v_masked_max.amax(dim=(1, 2))
                
                # Compute new tight dimensions
                new_L = max_u - min_u
                new_W = max_v - min_v
                
                # Compute center shift in Local Frame
                center_u = (max_u + min_u) * 0.5
                center_v = (max_v + min_v) * 0.5
                
                # Rotate shift back to Global Frame to update cx, cy
                # shift_x = u*cos(a) - v*sin(a)
                # shift_y = u*sin(a) + v*cos(a)
                # ct, st are (N, 1, 1). We need (N,)
                ct_flat = ct.view(N)
                st_flat = st.view(N)
                
                shift_x = center_u * ct_flat - center_v * st_flat
                shift_y = center_u * st_flat + center_v * ct_flat
                
                # Update Batch Data IN PLACE
                # This ensures that the Labels returned by pipeline.generate() are tight.
                
                # Use indexing to only update valid polyhedra
                idx = valid_indices
                
                # Update L and W
                # Add a small padding? 
                # Labels usually should be tight. 
                # The OBB is defined as center + L/W.
                batch.L[idx] = new_L[idx]
                batch.W[idx] = new_W[idx]
                
                # Update Center
                batch.cx[idx] = batch.cx[idx] + shift_x[idx]
                batch.cy[idx] = batch.cy[idx] + shift_y[idx]
                
                # Note: We do NOT change alpha. The box remains aligned with the generated orientation.
                # This is the "Tight AABB in Local Frame" approach.

        return g_buffer, t_x_mins, t_y_mins, aux_dict
