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

class ParticleShader:
    def __init__(self, config: SynthConfig, device: torch.device):
        self.cfg = config
        self.device = device

    def render_batch(self, batch: ParticleBatch, rng: random.Random, return_aux: bool = False) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Vectorized rendering for a batch of Particles (Rods, Plates, Cubes, Spheres).
        Returns: (patches, x_mins, y_mins, aux_dict)
        """
        if batch.cx.numel() == 0:
            return None, torch.empty(0), torch.empty(0), {}
            
        N = len(batch)
        if batch.cx.device != self.device:
            batch.to(self.device)
            
        cfg = self.cfg
        dev = self.device
        
        # 2. 3D Projection (Analytic)
        # Calculate Effective (Projected) L and W based on orientation
        # beta (tumble) -> Rotation around W axis (shortens L)
        # gamma (roll) -> Rotation around L axis (mixes W and H)
        
        beta_rad = torch.deg2rad(batch.beta)
        gamma_rad = torch.deg2rad(batch.gamma)
        
        cb, sb = torch.abs(torch.cos(beta_rad)), torch.abs(torch.sin(beta_rad))
        cg, sg = torch.abs(torch.cos(gamma_rad)), torch.abs(torch.sin(gamma_rad))
        
        # Projected Dimensions
        # W_eff: The width seen from top.
        # For Rod: W stays W (cylinder). Gamma doesn't change profile much (still circle).
        # For Plate/Cube: W_eff = W*cos(gamma) + H*sin(gamma)
        W_eff = batch.W * cg + batch.H * sg
        
        # For Rod/Sphere, override
        is_rod = (batch.shape_id == SHAPE_ROD)
        is_sphere = (batch.shape_id == SHAPE_SPHERE)
        is_bubble = (batch.shape_id == SHAPE_BUBBLE)
        is_droplet = (batch.shape_id == SHAPE_DROPLET)
        is_sphere_like = is_sphere | is_bubble | is_droplet
        
        W_eff = torch.where(is_rod | is_sphere_like, batch.W, W_eff)
        
        # L_eff: Length seen from top.
        # For Rod/Plate: L_eff = L*cos(beta) + H*sin(beta)
        L_eff = batch.L * cb + batch.H * sb
        L_eff = torch.where(is_sphere_like, batch.L, L_eff) # Sphere L is Diameter
        
        # Effective Optical Thickness (Max path length)
        tilt_factor = torch.clamp(torch.max(cb, cg), 0.1, 1.0)
        H_eff = batch.H / tilt_factor
        H_eff = torch.where(is_sphere_like | is_rod, batch.H, H_eff)
        
        # 3. Bounding Box Calculation
        # Rotation alpha (in-plane)
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
        
        # u: along Length, v: along Width
        L_half = (L_eff.view(N, 1, 1) / 2.0) + 1e-6
        u = (ct * X + st * Y) / L_half
        v = (-st * X + ct * Y)
        
        # 6. SDF / Height Map Generation
        
        # We need a normalized v coordinate (-1 to 1) for profile
        W_half = (W_eff.view(N, 1, 1) / 2.0) + 1e-6
        v_norm = v / W_half
        
        # --- Texture Scaling (Fix for Stretching) ---
        # Scale noise coordinates by physical dimensions (pixels)
        # Standardize: 1 unit of noise = 50 pixels length / 20 pixels width
        tex_scale_u = L_eff.view(N, 1, 1) / 50.0
        tex_scale_v = W_eff.view(N, 1, 1) / 20.0
        
        # --- Material Texture Setup ---
        # 0=Smooth, 1=Striated, 2=Pitted, 3=Granular
        tex_type = batch.texture_type.view(N, 1, 1)
        roughness = batch.surf_roughness.view(N, 1, 1) # Material roughness
        grain = batch.grain_size.view(N, 1, 1)
        
        # Seed
        seed_tex = rng.randint(0, 2**31 - 1)
        
        # Helper masks
        is_striated = (tex_type == 1)
        is_pitted = (tex_type == 2)
        is_granular = (tex_type == 3)

        # --- Texture Helper ---
        def compute_tex_mod(u_in, v_norm_in, seed_offset=0):
            # Base tex_mod (1.0 = no change)
            mod_out = torch.ones_like(u_in)
            
            # 1. Striated (Fibrous)
            if is_striated.any():
                striations = noise1d_like_batch(v_norm_in * tex_scale_v, corr=0.1, amp=1.0, seed=seed_tex + seed_offset)
                str_depth = roughness * 0.5
                m = 1.0 - (str_depth * torch.abs(striations))
                mod_out = torch.where(is_striated, m, mod_out)
                
            # 2. Pitted (Amorphous)
            if is_pitted.any():
                u_sc = u_in * tex_scale_u * grain
                v_sc = v_norm_in * tex_scale_v * grain
                n1 = noise1d_like_batch(u_sc, corr=0.2, seed=seed_tex + seed_offset)
                n2 = noise1d_like_batch(v_sc, corr=0.2, seed=seed_tex + 100 + seed_offset)
                pits = (n1 * n2).abs()
                m = 1.0 - (roughness * pits)
                mod_out = torch.where(is_pitted, m, mod_out)
                
            # 3. Granular (Powder)
            if is_granular.any():
                u_sc = u_in * tex_scale_u * 2.0 * grain
                noise = torch.randn_like(u_in) * 0.5
                m = 1.0 - (roughness * noise.abs())
                mod_out = torch.where(is_granular, m, mod_out)
            
            # Legacy Ragged
            if torch.any(batch.ragged_p > 0):
                 rag_strength = batch.ragged_p.view(N, 1, 1) * 0.1
                 rag_noise = noise1d_like_batch(u_in * tex_scale_u, corr=0.05, seed=seed_tex + 50 + seed_offset)
                 mod_out = mod_out - (rag_strength * rag_noise.abs())
                 
            return mod_out

        # Apply Texture (Straight)
        tex_mod = compute_tex_mod(u, v_norm)

        # --- B. Profile (Cross-Section) ---
        # Same as before (Round vs Flat), but we ADD the striations to the surface
        
        # Masks
        is_rod = (batch.shape_id == SHAPE_ROD)
        is_sphere_like = (batch.shape_id == SHAPE_SPHERE) | (batch.shape_id == SHAPE_BUBBLE) | (batch.shape_id == SHAPE_DROPLET)
        is_round = is_rod | is_sphere_like
        is_round = is_round.view(N, 1, 1)
        
        is_cube = (batch.shape_id == SHAPE_CUBE).view(N, 1, 1)
        
        # Round Profile
        h_round = torch.sqrt(torch.clamp(1.0 - v_norm**2, 0.0, 1.0))
        
        # Flat Profile (Plate/Cube)
        h_flat = torch.where(torch.abs(v_norm) < 0.8, 1.0, 
                             torch.clamp((1.0 - torch.abs(v_norm)) / 0.2, 0.0, 1.0))
                             
        # Cube Internal Edge Logic ("X" pattern)
        tumble_strength = torch.max(torch.abs(batch.beta), torch.abs(batch.gamma)).view(N, 1, 1) / 90.0
        tumble_strength = torch.clamp(tumble_strength, 0.0, 1.0)
        
        # Pyramid profile
        u_norm = u / ((L_eff.view(N, 1, 1) / 2.0) + 1e-6)
        # Note: u was already normalized by L_half in step 5?
        # Step 5: u = (ct * X + st * Y) / L_half
        # So u IS u_norm (-1..1). The variable 'u' IS normalized.
        # But we want to re-verify usage below.
        
        pyramid = 1.0 - torch.maximum(torch.abs(u), torch.abs(v_norm))
        pyramid = torch.clamp(pyramid, 0.0, 1.0)
        
        h_cube = h_flat * (1.0 - 0.6 * tumble_strength) + pyramid * (0.6 * tumble_strength)
        
        # Apply Texture to profiles
        h_round = h_round * tex_mod
        h_flat = h_flat * tex_mod
        h_cube = h_cube * tex_mod

        # Select Profile
        profile_v = torch.where(is_round, h_round, h_flat)
        profile_v = torch.where(is_cube, h_cube, profile_v)
        
        # --- C. Longitudinal Profile (The "Sausage" Fix) ---
        # Old code used 'smooth_cap' which makes them look like pills.
        # New code uses 'Jagged Breaks'.
        
        # 1. Create a "Break Map" at the ends
        # Noise along V at the U-ends
        # Use scaled V for consistent jaggedness size
        break_roughness = noise1d_like_batch(v_norm * tex_scale_v, corr=0.2, amp=1.0, seed=seed_tex+1)
        
        # The physical end is at |u| = 1.0 (actually u is normalized by L_half? No, u is normalized by L_half in original code: u = ... / L_half)
        # So u goes from approx -1 to 1.
        
        # We want the crystal to end abruptly between 0.8 and 1.0 based on the break map.
        # "u_limit" defines the jagged edge for this specific slice of v
        u_limit = 1.0 - 0.15 * torch.abs(break_roughness) # Ends are between 0.85L and 1.0L
        
        # Jagged Cutoff
        dist_to_edge = u_limit - torch.abs(u)
        
        # Scale sharpness by length so it is always 1-2 pixels
        # u is normalized by L_half. So 1 unit of u = L_half pixels.
        # We want sharpness of ~1 pixel. So we need slope proportional to L_half.
        # sharp_slope = L_half * k. If dist is 1/L_half (1 pixel), result is k.
        
        sharpness_factor = (L_eff.view(N, 1, 1) * 0.5) + 1.0 # approx L_half
        profile_u = torch.clamp(dist_to_edge * sharpness_factor, 0.0, 1.0) 
        
        # Spheres still need round caps
        h_u_sphere = torch.sqrt(torch.clamp(1.0 - u**2, 0.0, 1.0))
        
        # Cube longitudinal
        is_cube_view = is_cube # Already (N,1,1)
        if is_cube_view.any():
            u_flat = torch.where(torch.abs(u) < 0.8, 1.0, 
                             torch.clamp((1.0 - torch.abs(u)) / 0.2, 0.0, 1.0))
             # Apply jaggedness to cubes too? Maybe less.
             # Let's keep cubes "perfect" for now or use same jagged logic?
             # Real cubes (NaCl) break squarely.
             # Let's stick to flat profile for cubes for now, or mix.
            profile_u = torch.where(is_cube_view, u_flat, profile_u)

        is_sphere_like_view = is_sphere_like.view(N, 1, 1)
        profile_u = torch.where(is_sphere_like_view, h_u_sphere, profile_u)

        # --- D. Shape Modes & Defects (With Texture Scaling) ---
        # 0: straight, 1: wavy, 2: kink, 3: noisy
        shape_mode = batch.shape_mode.view(N, 1, 1)
        v_offset = torch.zeros_like(v)
        
        is_wavy = (shape_mode == 1)
        is_kink = (shape_mode == 2)
        is_noisy = (shape_mode == 3)
        
        # Apply scaling to u for consistent wavelength
        u_scaled = u * tex_scale_u
        
        if is_wavy.any():
            amp = batch.W.view(N, 1, 1) * 0.25
            wobble = sin_wobble_batch(u_scaled) # Scaled U
            v_offset = torch.where(is_wavy, wobble * amp, v_offset)
            
        if is_kink.any():
            amp = batch.W.view(N, 1, 1) * 0.5
            kink_off = kink_batch(u_scaled) # Scaled U
            v_offset = torch.where(is_kink, kink_off * amp, v_offset)
            
        if is_noisy.any():
            amp = batch.W.view(N, 1, 1) * 0.2
            noise_off = noisy_wobble_batch(u_scaled, corr=0.1) # Scaled U
            v_offset = torch.where(is_noisy, noise_off * amp, v_offset)

        # --- Surface Roughness (With Texture Scaling) ---
        if torch.any(batch.ragged_p > 0):
            # ragged_p is 0.0-1.0 amplitude
            rag_amp = batch.ragged_p.view(N, 1, 1) * batch.W.view(N, 1, 1) * 0.15
            rag_noise = noise1d_like_batch(u_scaled, corr=0.05) # Scaled U
            v_offset = v_offset + rag_noise * rag_amp
        
        v_norm_bent = (v - v_offset) / W_half
        
        # Re-compute Profile with bent V
        h_round_bent = torch.sqrt(torch.clamp(1.0 - v_norm_bent**2, 0.0, 1.0))
        h_flat_bent = torch.where(torch.abs(v_norm_bent) < 0.8, 1.0, 
                             torch.clamp((1.0 - torch.abs(v_norm_bent)) / 0.2, 0.0, 1.0))
                             
        # Re-apply texture for bent coordinates
        # We reuse the helper to ensure consistent material look
        tex_mod_bent = compute_tex_mod(u, v_norm_bent)
        
        h_round_bent = h_round_bent * tex_mod_bent
        h_flat_bent = h_flat_bent * tex_mod_bent
        
        # Cube bending?
        # Let's ignore cubes for bending for now, or apply same logic.
        # Just update profile_v for round/flat shapes.
        
        profile_v_bent = torch.where(is_round, h_round_bent, h_flat_bent)
        
        # If no bending happened (v_offset=0), this is same as before.
        # Use profile_v_bent instead of profile_v.
        
        # Combine
        # Correct Sphere SDF logic again
        if is_sphere_like_view.any():
             # Spheres shouldn't bend usually.
             r_sq = u**2 + v_norm**2
             h_sphere_true = torch.sqrt(torch.clamp(1.0 - r_sq, 0.0, 1.0))
             
             # Keep original height_map for spheres.
             height_map_bent = torch.where(is_sphere_like_view, h_sphere_true, profile_v_bent * profile_u)
        else:
             height_map_bent = profile_v_bent * profile_u
             
        # --- Internal Inclusions (Volumetric Cloudiness) ---
        inclusions = batch.internal_inclusions.view(N, 1, 1)
        if torch.any(inclusions > 0):
             # Volumetric noise (Pseudo-2D)
             # High frequency, low correlation
             noise_u = noise1d_like_batch(u * tex_scale_u, corr=0.1, seed=seed_tex + 999)
             noise_v = noise1d_like_batch(v_norm * tex_scale_v, corr=0.1, seed=seed_tex + 888)
             inc_noise = noise_u * noise_v
             
             # Solvent Inclusions (Phase 4):
             # Create larger, smoother "pockets"
             # Use lower frequency noise
             pocket_u = noise1d_like_batch(u * tex_scale_u * 0.2, corr=0.5, seed=seed_tex + 777)
             pocket_v = noise1d_like_batch(v_norm * tex_scale_v * 0.2, corr=0.5, seed=seed_tex + 666)
             pockets = (pocket_u * pocket_v)
             
             # Combine: Fine inclusions + Large pockets
             total_noise = inc_noise + pockets * 2.0
             
             # Modulate height map: variation in optical path length
             # We scale by height so noise fades at edges
             height_map_bent = height_map_bent * (1.0 + total_noise * inclusions * 1.5)
             
             # Cracks/Fractures (Phase 4):
             # Sharp lines of discontinuity.
             # Use a "Lightning" pattern -> |Noise| < threshold
             crack_noise = noise1d_like_batch(u * tex_scale_u * 0.5, corr=0.8, seed=seed_tex + 555)
             # Threshold close to zero
             is_crack = (torch.abs(crack_noise) < 0.05).float()
             # Only occur rarely (based on inclusions param for now, or add specific defect param)
             # Let's say high inclusion count -> high defect probability
             has_crack = (inclusions > 0.5).float()
             
             # Apply crack (dark line)
             height_map_bent = height_map_bent * (1.0 - is_crack * has_crack * 0.5)

        # Replace height_map
        height_map = height_map_bent
        phys_height = H_eff.view(N, 1, 1) * height_map

        
        # 7. Optics Simulation
        # Mode Dispatch
        mode = cfg.optics.mode
        
        # Delta (Phase/Absorbance strength)
        # Apply Polarity Flip (Invert Delta)
        flip_mult = 1.0 - 2.0 * batch.polarity_flip_p.view(N, 1, 1)
        effective_delta = batch.delta.view(N, 1, 1) * flip_mult
        
        phase = phys_height * effective_delta
        
        if mode == "dic":
            # DIC Logic: Derivative of Phase
            
            # Shadow params
            sh_gain = torch.empty(N, device=dev).uniform_(*cfg.optics.shadow_gain)
            sh_gain = sh_gain.view(N, 1, 1)
            
            # Lighting Angle
            light_ang = math.radians(cfg.optics.lighting_angle_deg)
            lx, ly = math.cos(light_ang), math.sin(light_ang)
            
            # 2. Refraction Loss (The "Dark Edge" / "Planar" Fix)
            # Gradient X
            h_right = torch.roll(height_map, shifts=-1, dims=2)
            h_left  = torch.roll(height_map, shifts=1, dims=2)
            slope_x = (h_right - h_left) * 0.5 * W_half 
            
            # Gradient Y
            h_down = torch.roll(height_map, shifts=-1, dims=1)
            h_up   = torch.roll(height_map, shifts=1, dims=1)
            slope_y = (h_down - h_up) * 0.5 * W_half
            
            # Directional Slope
            slope = slope_x * lx + slope_y * ly
            
            edge_steepness = torch.sqrt(slope_x**2 + slope_y**2)
            scattering = torch.clamp(edge_steepness * 2.0 - 0.5, 0.0, 1.0) # Thresholded darkness
            
            # --- Corner Glint ---
            # Glint at sharp corners of the coordinate system
            # Corners are where |u| ~ 1 and |v_norm| ~ 1 (for plates/cubes)
            # We want this only for rectangular shapes (Rod/Plate/Cube)
            
            # Corner mask: (|u| > 0.8) & (|v_norm| > 0.8)
            # Distance from corner
            corner_dist = torch.sqrt(torch.clamp(torch.abs(u) - 0.8, 0.0, 1.0)**2 + torch.clamp(torch.abs(v_norm) - 0.8, 0.0, 1.0)**2)
            # Peak at corner
            corner_glint = torch.exp(-(corner_dist - 0.2)**2 * 50.0) 
            
            # Only apply to Plate and Cube
            is_angular = (batch.shape_id == SHAPE_PLATE) | (batch.shape_id == SHAPE_CUBE)
            is_angular = is_angular.view(N, 1, 1).float()
            
            # Modulate by slope to ensure it's on the edge
            corner_glint = corner_glint * edge_steepness * is_angular * 5.0
            
            # 3. Assemble Image
            # DIC Signal: d(Phase)/dx = d(Height)/dx * Delta
            # We approximate d(Height)/dx with 'slope'.
            # So Signal = slope * effective_delta * gain
            
            # FIXED: Normalize so that a typical delta (-0.15) gives a factor of ~1.0 
            # Dividing by -6.0 was too aggressive (it assumed delta was -6.0!) 
            # We divide by -0.15 to say "At delta -0.15, use 100% of the gain". 
            ref_delta = -0.15 
            delta_factor = effective_delta / ref_delta 
            
            # Calculate Base Signal (The bright/dark edges) 
            base_signal = slope * delta_factor * sh_gain 
            
            # Absorption (Refraction/Phase darkening)
            raw_absorption = phys_height * effective_delta * 0.05 
            absorption = torch.max(raw_absorption, torch.tensor(-0.5, device=dev)) 

            # Material Opacity (Light blocking)
            # batch.opacity is 0..1. 1 means fully black.
            opacity_map = batch.opacity.view(N, 1, 1)
            if torch.any(opacity_map > 0):
                # Strong darkening proportional to thickness
                # If opacity is 1.0 (metal), we want it BLACK except for glints
                op_term = -10.0 * phys_height * opacity_map
                absorption = absorption + op_term
                
            # Scattering / Metallic Glint
            # If high RI and high opacity, add specular highlights at edges
            # Approximation: high derivative = bright
            is_metal = (batch.opacity > 0.9) & (batch.refractive_index > 2.0)
            is_metal = is_metal.view(N, 1, 1).float()
            
            # Initialize scattering_intensity for metal
            scattering_intensity = torch.zeros_like(edge_steepness)
            
            if torch.any(is_metal > 0):
                 # Edge specular
                 glint = torch.clamp(edge_steepness - 0.2, 0.0, 1.0) * 5.0
                 # Add to scattering (which brightens the image)
                 scattering_intensity = scattering_intensity + glint * is_metal

            # Scattering logic (kept your new logic, it's fine) 
            # scattering_intensity = torch.clamp(torch.abs(delta_factor), 0.0, 1.0) 
            dark_edges = -1.0 * scattering * 0.8 * scattering_intensity
            
            # Add Corner Glint (Bright)
            # It's an additive light term
            layer = base_signal + absorption + dark_edges + corner_glint
            
            # --- Grain Boundaries ---
            # Dark lines where crystals overlap/intersect
            # We detect this by looking for "valleys" in the height map where multiple objects might meet
            # However, height_map is flattened.
            # A better heuristic for DIC:
            # If height is high but slope is low, it's a flat face.
            # If slope is high, it's an edge.
            # Real grain boundaries are often dark lines.
            # We can simulate this by darkening regions with very high negative curvature (crevices).
            
            # Laplacian of height ~ Curvature
            # Use Sobel to get 2nd derivatives
            # We already have slope_x, slope_y.
            # curv = d(slope_x)/dx + d(slope_y)/dy
            
            # d(slope_x)/dx
            sx_right = torch.roll(slope_x, shifts=-1, dims=2)
            sx_left  = torch.roll(slope_x, shifts=1, dims=2)
            dsx_dx = (sx_right - sx_left) * 0.5
            
            # d(slope_y)/dy
            sy_down = torch.roll(slope_y, shifts=-1, dims=1)
            sy_up   = torch.roll(slope_y, shifts=1, dims=1)
            dsy_dy = (sy_down - sy_up) * 0.5
            
            curvature = dsx_dx + dsy_dy
            
            # Grain boundaries (crevices) have positive curvature (concave up) in height map?
            # Height is 0 to 1 (convex object).
            # A valley between two hills has positive curvature (like a cup).
            # Peaks have negative curvature.
            
            # We want to darken high positive curvature (valleys).
            crevice_mask = torch.clamp(curvature - 0.05, 0.0, 1.0)
            
            # Only apply where there is actually height (ignore background)
            crevice_mask = crevice_mask * (height_map > 0.1).float()
            
            # Darken
            layer = layer - crevice_mask * 5.0
            
            # Apply Bubble/Droplet Overrides (Negative Classes)
            is_bubble_v = is_bubble.view(N, 1, 1)
            is_droplet_v = is_droplet.view(N, 1, 1)
            
            if is_bubble_v.any():
                # Bubble: Dark rim (refraction limit), clear center
                edge_val = torch.clamp(1.0 - height_map, 0.0, 1.0)
                
                # Fix: Mask out the square artifacts outside the bubble radius
                mask_inside = (height_map > 0.001).float()
                
                # Fix: Make rim negative (dark) instead of positive (bright)
                bubble_rim = -1.0 * (edge_val ** 6.0) * 4.0 # Sharp dark rim
                bubble_rim = bubble_rim * mask_inside
                
                absorption = torch.where(is_bubble_v, bubble_rim, absorption)
                
            if is_droplet_v.any():
                # Droplet: Subtle rim
                edge_val = torch.clamp(1.0 - height_map, 0.0, 1.0)
                mask_inside = (height_map > 0.001).float()
                
                # Fix: Mask and ensure correct sign
                droplet_rim = -1.0 * (edge_val ** 3.0) * 0.8
                droplet_rim = droplet_rim * mask_inside
                
                absorption = torch.where(is_droplet_v, droplet_rim, absorption)
            
            layer = base_signal + absorption + dark_edges
            
        elif mode == "brightfield":
            # Simple absorption
            # Darker where thicker (Phase is negative usually -> dark)
            layer = phase 
            
            # --- Diffraction Fringes (Airy Disks) ---
            # Simulate fringes around the edges
            # Edge is where height_map is low but > 0.
            # Pattern: oscillating dark/light bands decaying inward
            
            # Frequency of fringes
            k_fringe = 30.0
            # Decay inward
            decay = torch.exp(-height_map * 8.0)
            fringes = torch.sin(height_map * k_fringe) * decay * 0.3
            
            # Add to layer (phase is negative, so fringes modulate around it)
            layer = layer + fringes

            is_bubble_v = is_bubble.view(N, 1, 1)

            if is_bubble_v.any():
                # Bubble in brightfield: Dark rim, bright center (lensing)
                # Additive: "Dark" = negative, "Bright" = positive.
                edge_val = torch.clamp(1.0 - height_map, 0.0, 1.0)
                # Dark rim
                rim = -(edge_val ** 6.0) * 5.0
                # Bright center (Lensing)
                center = (height_map ** 4.0) * 2.0
                bubble_layer = rim + center
                layer = torch.where(is_bubble_v, bubble_layer, layer)
            
        elif mode == "polarization":
            # Maltese Cross
            # I ~ sin^2(2*theta) * sin^2(delta/2)
            # theta is angle between polarizer and local crystal axis.
            # Crystal axis is 'alpha'. Polarizer is 'cfg.polarizer_angle_deg'.
            
            pol_ang = math.radians(cfg.optics.polarizer_angle_deg)
            
            if is_sphere.any():
                # Radial coordinates
                local_ang = torch.atan2(Y, X) # Global angle of pixel
                # For spherulite, optical axis is radial.
                theta = local_ang - pol_ang
            else:
                # For rods, axis is constant 'alpha'
                theta = torch.deg2rad(batch.alpha).view(N, 1, 1) - pol_ang
                
            # Cross term
            cross = torch.sin(2 * theta) ** 2
            
            # Retardation term
            # sin^2(phase / 2)
            # Add material birefringence factor
            bire_strength = torch.abs(batch.birefringence.view(N, 1, 1))
            # Normalize: standard material has 0.0. fibrous 0.2. high 0.35.
            # Phase is already huge.
            # We want 'bire_strength' to modulate the color/intensity.
            # If bire=0 (isotropic), output should be black? Yes.
            
            # If birefringence is low, retardation is low.
            # Phase is proportional to thickness * delta.
            # Delta is usually refractive index difference.
            # Retardation = thickness * birefringence.
            # Our 'phase' variable is thickness * delta.
            # We should recalculate retardation properly using birefringence.
            
            retardation_val = phys_height * bire_strength * 20.0 # Gain
            
            retardation = torch.sin(retardation_val) ** 2
            
            layer = cross * retardation * 10.0 # Gain
            
        elif mode == "polarization_rgb":
            # --- Polychromatic Polarization (Michel-Levy) ---
            # Returns 3-channel RGB
            
            # 1. Calculate Angle Term (Crossed Polarizers)
            pol_ang = math.radians(cfg.optics.polarizer_angle_deg)
            if is_sphere.any():
                local_ang = torch.atan2(Y, X)
                theta = local_ang - pol_ang
            else:
                theta = torch.deg2rad(batch.alpha).view(N, 1, 1) - pol_ang
                
            # Intensity modulation from Crossed Polarizers: sin^2(2*theta)
            # We add a small 'leak' (0.05) so it's never perfectly black (extinction is rarely perfect)
            intensity_mod = torch.sin(2 * theta) ** 2 + 0.05
            
            # 2. Calculate Retardation (nm)
            # Retardation = Thickness * Birefringence
            # We assume phys_height is roughly proportional to microns.
            # Boost physical scale to push into 2nd/3rd order colors (550nm - 1600nm)
            # Previous 200.0 was too low (mostly 1st order gray).
            # 1000.0 * 0.05 (typical bire) * 1.0 (height) = 50nm (still low).
            # We need retardation ~ 1000nm.
            # So scale factor should be around 20000? 
            # If height=1.0, bire=0.05 -> ret = 20000 * 1 * 0.05 = 1000nm. Good.
            
            thickness_scale = 15000.0 
            bire = torch.abs(batch.birefringence.view(N, 1, 1))
            # If bire is 0 (isotropic), retardation is 0 -> black.
            retardation_nm = phys_height * thickness_scale * bire
            
            # 3. Spectral Interference (Approximation of Newton's Series)
            # We sum multiple wavelengths for each channel to simulate broad spectrum (White Light)
            # R channel: centered ~620nm
            # G channel: centered ~530nm
            # B channel: centered ~450nm
            
            def interference_intensity(ret_nm, lambda_nm):
                # I = sin^2(pi * R / lambda)
                return torch.sin(math.pi * ret_nm / lambda_nm) ** 2

            # Sample multiple wavelengths to broaden the spectrum and reduce "laser" look
            # Red: 600, 630, 660
            i_r = (interference_intensity(retardation_nm, 600.0) + 
                   interference_intensity(retardation_nm, 630.0) + 
                   interference_intensity(retardation_nm, 660.0)) / 3.0
                   
            # Green: 500, 530, 560
            i_g = (interference_intensity(retardation_nm, 500.0) + 
                   interference_intensity(retardation_nm, 530.0) + 
                   interference_intensity(retardation_nm, 560.0)) / 3.0
                   
            # Blue: 420, 450, 480
            i_b = (interference_intensity(retardation_nm, 420.0) + 
                   interference_intensity(retardation_nm, 450.0) + 
                   interference_intensity(retardation_nm, 480.0)) / 3.0
            
            # Combine
            rgb = torch.stack([i_r, i_g, i_b], dim=1) # (N, 3, H, W)
            
            # Modulate
            # Gain boosted to 15.0 for vibrancy
            rgb = rgb * intensity_mod.unsqueeze(1) * 15.0 
            
            layer = rgb


        elif mode == "fluorescence":
            # Fluorescence / Confocal
            # Dark background.
            # Signal ~ Volume * Efficiency
            # Volume ~ phys_height
            
            # Efficiency can be random per particle or material property
            # For now assume everything fluoresces a bit if enabled
            
            fluor_eff = torch.rand(N, device=dev).view(N, 1, 1) * 0.8 + 0.2
            
            # Emission
            signal = phys_height * fluor_eff
            
            # Color? Usually Green (FITC) or Red (TRITC).
            # Let's produce Green.
            # R=0, G=signal, B=0
            
            # (N, 3, H, W)
            zeros = torch.zeros_like(signal)
            layer = torch.stack([zeros, signal, zeros * 0.2], dim=1) # Slight blue tint
            
            # Add glow?
            # Blur will happen later.
            
        elif mode == "confocal":
            # Confocal: Optical Sectioning
            # Like Fluorescence, but strictly cuts out-of-focus light.
            # Signal ~ Volume * Efficiency
            
            fluor_eff = torch.rand(N, device=dev).view(N, 1, 1) * 0.8 + 0.2
            signal = phys_height * fluor_eff
            
            # Sectioning: Weight by distance from focus plane
            # Weight = exp(-(z - focus_z)^2 / sigma^2)
            # Sigma is very small (thin slice)
            
            dist = torch.abs(batch.z - cfg.optics.focus_z)
            # Narrow sigma
            section_weight = torch.exp(-(dist**2) / (0.05**2)) # 0.05 is thin slice
            section_weight = section_weight.view(N, 1, 1)
            
            signal = signal * section_weight
            
            # Green channel
            zeros = torch.zeros_like(signal)
            layer = torch.stack([zeros, signal, zeros], dim=1)

        elif mode == "shadowgraphy":
            # Shadowgraphy: Phase Contrast + Defocus
            # I ~ I0 * (1 - k * z * Laplacian(Phi))
            
            # 1. Base Transmission
            # If object is opaque, it blocks light.
            # If transparent (phase object), it transmits light but deflects it.
            
            # Simple absorption (opaque parts)
            base_trans = 1.0 - (batch.opacity.view(N, 1, 1) * height_map)
            
            # 2. Refractive/Diffractive Lensing
            # "Bright Center" effect for bubbles/spheres (positive lens)
            # "Dark Rim" (total internal reflection or high gradients)
            
            # Laplacian approximation using curvature
            # We already calculated curvature for grain boundaries! 
            # Re-calculate curvature from height_map for everything
            
            # Gradient
            h_right = torch.roll(height_map, shifts=-1, dims=2)
            h_left  = torch.roll(height_map, shifts=1, dims=2)
            slope_x = (h_right - h_left) * 0.5 * W_half 
            
            h_down = torch.roll(height_map, shifts=-1, dims=1)
            h_up   = torch.roll(height_map, shifts=1, dims=1)
            slope_y = (h_down - h_up) * 0.5 * W_half
            
            # Curvature (Laplacian)
            sx_right = torch.roll(slope_x, shifts=-1, dims=2)
            sx_left  = torch.roll(slope_x, shifts=1, dims=2)
            d2x = (sx_right - sx_left) * 0.5
            
            sy_down = torch.roll(slope_y, shifts=-1, dims=1)
            sy_up   = torch.roll(slope_y, shifts=1, dims=1)
            d2y = (sy_down - sy_up) * 0.5
            
            laplacian = d2x + d2y
            
            # Shadowgraphy Intensity Mod
            # Proportional to Defocus (Z distance)
            # Z is -1 to 1. Focus Z is cfg.optics.focus_z.
            # defocus = z - focus_z
            defocus = batch.z.view(N, 1, 1) - cfg.optics.focus_z
            
            # Enhance defocus effect
            # If defocus is 0, we see nothing (pure phase).
            # We add a small constant to simulate "always slightly out of focus" or imperfections
            defocus = defocus + 0.2 * torch.sign(defocus + 1e-6)
            
            # Shadowgraph term
            # I = 1 - epsilon * defocus * laplacian
            shadow_signal = -50.0 * defocus * laplacian
            
            # Clamp strong signals (caustics)
            shadow_signal = torch.clamp(shadow_signal, -0.8, 2.0)
            
            # Combine
            # Start with gray background (0.0 in this shader context implies additive/subtractive to base)
            # But here 'layer' is the final intensity map for the object.
            # In pipeline, we stamp this onto background.
            # Shadowgraphy objects can brighten (lensing) or darken (scattering).
            
            layer = shadow_signal * base_trans
            
            # 3. Bubbles/Droplets specific override (Strong lensing)
            if is_sphere_like.any():
                # Center bright spot
                lens_center = torch.exp(-10.0 * (u**2 + v_norm**2)) * 2.0
                # Dark rim
                rim = -1.0 * (1.0 - height_map)**4 * 5.0
                
                lens_effect = lens_center + rim
                
                # Apply only to spheres
                is_sl = is_sphere_like.view(N, 1, 1)
                layer = torch.where(is_sl, layer + lens_effect, layer)
            
            # 4. Diffraction Fringes (BOKEH / Airy)
            # Add rings if out of focus
            # Ring radius scales with defocus
            if cfg.optics.aperture > 0: # If DoF enabled
                abs_defocus = torch.abs(defocus)
                # Oscillating term based on distance from edge?
                # Or simply modulate existing signal
                fringes = torch.sin(height_map * 40.0 * abs_defocus) * 0.2
                layer = layer + fringes

        else:
            # Fallback
            layer = phase
        
        # 8. Jitter & Artifacts (Legacy)
        # Apply noise/jitter to the generated layer
        # Legacy jitter modes removed for 3D cleanliness
                 
        # 9. Blur (Depth of Field)
        # Use cfg.optics.aperture and cfg.optics.focus_z
        # If aperture > 0, apply blur based on |z - focus_z|
        # If ghost blur is set, add it (legacy)
        
        focus_dist = torch.abs(batch.z - cfg.optics.focus_z)
        blur_sigs = focus_dist * cfg.optics.aperture * 5.0 # Gain for visibility
        
        # Legacy Ghost Blur
        ghost_sig = cfg.physics.ghosts.blur_sigma
        if ghost_sig > 0:
             blur_sigs = blur_sigs + torch.abs(batch.z) * max(0.1, ghost_sig)
        
        # Apply blur
        # Note: layer might be (N, H, W) or (N, 3, H, W)
        # gaussian_blur_batch usually handles (N, C, H, W) or (N, H, W)
        
        if torch.any(blur_sigs > 0.5):
            if layer.dim() == 3: # (N, H, W)
                layer = gaussian_blur_batch(layer, blur_sigs)
            else: # (N, 3, H, W)
                # Blur each channel same amount
                # Reshape to (N*3, 1, H, W) or loop?
                # gaussian_blur_batch expects (N, H, W) or (N, 1, H, W) usually.
                # Let's loop channels or reshape.
                # Easier: Treat N*3 as batch dim for blur
                # blur_sigs is (N,). Need (N*3,)
                
                N_b, C_b, H_b, W_b = layer.shape
                layer_reshaped = layer.view(N_b * C_b, H_b, W_b)
                sigs_reshaped = blur_sigs.repeat_interleave(3)
                
                layer_blurred = gaussian_blur_batch(layer_reshaped, sigs_reshaped)
                layer = layer_blurred.view(N_b, C_b, H_b, W_b)
            
        # 10. Finalize
        # Ensure output is (N, 3, H, W)
        if layer.dim() == 3:
             patch = layer.unsqueeze(1).repeat(1, 3, 1, 1)
        else:
             patch = layer
        
        aux_dict = {}
        if return_aux:
            # height: phys_height (N, Hp, Wp) -> (N, 1, Hp, Wp)
            # Ensure phys_height is available (it is calculated above)
            aux_dict['height'] = phys_height.unsqueeze(1)
            
            # mask: (N, 1, Hp, Wp)
            # Binary mask where height > 0.001
            mask_val = (height_map > 0.001).float()
            aux_dict['mask'] = mask_val.unsqueeze(1)
            
            # depth: batch.z (N,) -> (N, 1, Hp, Wp)
            z_map = batch.z.view(N, 1, 1).expand(-1, height_map.shape[1], height_map.shape[2])
            aux_dict['depth'] = (z_map * mask_val).unsqueeze(1)

        return patch, t_x_mins, t_y_mins, aux_dict
