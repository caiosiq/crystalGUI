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
            
            # 2. Refraction Loss (The "Dark Edge" / "Planar" Fix)
            h_right = torch.roll(height_map, shifts=-1, dims=2)
            h_left  = torch.roll(height_map, shifts=1, dims=2)
            
            # Calculate slope from height map directly, replacing analytical slope
            # Scale by width to maintain slope magnitude
            slope = (h_right - h_left) * 0.5 * W_half 
            
            edge_steepness = torch.abs(slope)
            scattering = torch.clamp(edge_steepness * 2.0 - 0.5, 0.0, 1.0) # Thresholded darkness
            
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
            
            # OPTIONAL: Clamp absorption so thick particles don't become black holes 
            # If phys_height is large, this term can get huge. 
            # We limit the max darkness from absorption to -0.5 (50% grey) 
            raw_absorption = phys_height * effective_delta * 0.05 
            absorption = torch.max(raw_absorption, torch.tensor(-0.5, device=dev)) 

            # Scattering logic (kept your new logic, it's fine) 
            scattering_intensity = torch.clamp(torch.abs(delta_factor), 0.0, 1.0) 
            dark_edges = -1.0 * scattering * 0.8 * scattering_intensity
            
            layer = base_signal + absorption + dark_edges
            
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
            retardation = torch.sin(phase) ** 2
            
            layer = cross * retardation * 10.0 # Gain
            
        elif mode == "shadowgraphy":
            # Binary-ish silhouette
            # High contrast, inverted
            # layer should be negative (block light)
            layer = -5.0 * height_map
            # Add bokeh later? Pipeline handles blur.
            
        else:
            # Fallback
            layer = phase
        
        # 8. Jitter & Artifacts (Legacy)
        # Apply noise/jitter to the generated layer
        # Legacy jitter modes removed for 3D cleanliness
                 
        # 9. Blur (Depth of Field)
        ghost_sig = cfg.physics.ghosts.blur_sigma
        scale = max(0.1, ghost_sig) if ghost_sig > 0 else 2.0
        blur_sigs = torch.abs(batch.z) * scale
        
        if torch.any(blur_sigs > 0.5):
            layer = gaussian_blur_batch(layer, blur_sigs)
            
        # 10. Finalize
        patch = layer.unsqueeze(1).repeat(1, 3, 1, 1)
        
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
