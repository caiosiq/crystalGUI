from typing import Tuple, Dict, Optional
import torch
import torch.nn.functional as F
import random
import math
from ...config import SynthConfig
from ...physics.particles import ParticleBatch
from ..utils.noise import generate_fractal_noise_3d, generate_cellular_noise_3d
from ...utils.math_torch import noise1d_like_batch, smooth_cap

class TextureShader:
    def __init__(self, config: SynthConfig, device: torch.device):
        self.cfg = config
        self.device = device
        
    def generate_maps(self, 
                      batch: ParticleBatch, 
                      u: torch.Tensor, 
                      v_norm: torch.Tensor, 
                      tex_scale_u: torch.Tensor, 
                      tex_scale_v: torch.Tensor,
                      max_h: int, 
                      max_w: int,
                      rng: random.Random) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generates Geometric Crystal Textures (Growth Steps, Striations, Etching)
        AND Volumetric Inclusions (Clouds, Grain Boundaries).
        
        Returns:
            roughness_map: Surface roughness (0.0-1.0)
            transmission_map: Internal defects (0.0-1.0, where 0 blocks light)
            turbidity_map: Volumetric scattering density (0.0-1.0)
        """
        N = len(batch)
        seed_tex = rng.randint(0, 2**31 - 1)
        
        tex_type = batch.texture_type.view(N, 1, 1)
        roughness = batch.surf_roughness.view(N, 1, 1)
        grain = batch.grain_size.view(N, 1, 1)
        
        # Phase 4.4.2.3.1: Anisotropy
        anisotropy = batch.anisotropy.view(N, 1, 1)
        anisotropy_angle = batch.anisotropy_angle.view(N, 1, 1)
        
        # Phase 4.4.2.3.2: Volumetric Props
        # Use grain_size as a proxy for Fractal Scale if not explicit
        # High grain = Large features (Low freq)
        fractal_scale = grain * 2.0 + 0.1 
        
        # Define types
        is_striated = (tex_type == 1) # Deep Ridges (Quartz/Tourmaline)
        is_pitted   = (tex_type == 2) # Acid Etching / Geometric Pits
        is_stepped  = (tex_type == 3) # Growth Steps / Terraces (Replaces Granular)
        
        # --- 1. Base Micro-Structure ---
        # Real crystals are never perfectly mathematically smooth. 
        # We add a tiny bit of "tooth" so lighting always catches something.
        micro_grain = torch.randn_like(u) * 0.02
        noise_accum = 0.5 + micro_grain # Center at 0.5 for mid-gray
        
        # --- Phase 4.4.2.3.1: Anisotropic Surface Texture ---
        if torch.any(anisotropy > 0):
             # 1. Anisotropic Noise Generation
             # We want "streaks" that are long in one direction (U or rotated) and short in the other.
             # High Anisotropy -> High Freq V, Low Freq U
             
             # Frequency Scaling
             # As anisotropy -> 1.0, V freq increases (thinner lines), U freq decreases (longer lines)
             freq_v = 5.0 * (1.0 + anisotropy * 10.0) 
             freq_u = 0.5 / (1.0 + anisotropy * 10.0)
             
             # Rotation (Simulated by mixing U/V coords)
             if torch.any(anisotropy_angle != 0):
                 ang_rad = torch.deg2rad(anisotropy_angle)
                 # Simple 2D rotation of the domain before noise sampling
                 # u is in microns (large), v_norm is 0..1 (small). Scale v to match u roughly?
                 # Or just mix.
                 # Let's treat v as 0..W roughly. W is usually ~10-50. u is ~100.
                 # Approximate aspect ratio
                 aspect = 5.0 
                 v_scaled = v_norm * aspect * 20.0 # Make v comparable to u magnitude
                 
                 u_rot = u * torch.cos(ang_rad) - v_scaled * torch.sin(ang_rad)
                 v_rot = u * torch.sin(ang_rad) + v_scaled * torch.cos(ang_rad)
                 
                 # Use rotated coords for noise
                 coord_u = u_rot
                 coord_v = v_rot
             else:
                 coord_u = u
                 coord_v = v_norm * 20.0 # Arbitrary scaling to match freq
             
             # Generate Noise Components
             # 1D Noise along U (Longitudinal variation)
             n_long = noise1d_like_batch(coord_u * freq_u, corr=0.95, seed=seed_tex + 777)
             
             # 1D Noise along V (Cross-section variation)
             n_cross = noise1d_like_batch(coord_v * freq_v, corr=0.2, seed=seed_tex + 888)
             
             # Combine
             # Multiplication creates "broken" streaks. Addition creates continuous waves.
             # We want broken streaks (scratches).
             ani_noise = n_long * n_cross
             
             # Add to accumulator
             # We modulate the strength by the anisotropy parameter
             # And blend it into the base noise
             noise_accum = noise_accum + ani_noise * anisotropy * 0.8

        # --- TYPE 1: DEEP STRIATIONS (Sharp Ridges) ---
        if is_striated.any():
            # Instead of a sine wave, we use a sharp power function.
            # shape: | | | | instead of ~ ~ ~ ~
            
            freq = 20.0 * (1.0 + grain * 2.0)
            # Longitudinal lines (along U)
            # We vary the frequency slightly across V to make it look organic
            wobble = torch.sin(u * 2.0) * 0.1
            
            # Sharp Ridge Math: abs(sin(x))^power
            # High power = sharper lines
            ridges = torch.abs(torch.sin((v_norm + wobble) * freq * 3.14159)) 
            ridges = torch.pow(ridges, 8.0) # Power of 8 makes them very thin/sharp
            
            # Add some "skipping" (lines that fade in and out)
            skip_pattern = noise1d_like_batch(u * tex_scale_u, corr=0.5, seed=seed_tex)
            ridges = ridges * (0.5 + 0.5 * skip_pattern)
            
            # Add to accumulator (0.0 to 1.0 range)
            noise_accum = torch.where(is_striated, noise_accum + ridges * 0.8, noise_accum)

        # --- TYPE 2: CRYSTALLINE ETCHING (Geometric Patches) ---
        if is_pitted.any():
            # Instead of white noise, we want "patches" of roughness.
            # We assume the surface is smooth, but has "corroded" areas.
            
            # Low freq clouds to define "where" the etching is
            u_low = u * tex_scale_u * 2.0
            v_low = v_norm * tex_scale_v * 2.0
            mask_noise = noise1d_like_batch(u_low, corr=0.5, seed=seed_tex) * noise1d_like_batch(v_low, corr=0.5, seed=seed_tex+1)
            
            # High freq grit for the actual texture inside the pits
            grit = torch.randn_like(u)
            
            # Thresholding: "If noise > 0.3, apply grit, else smooth"
            # This creates distinct hard edges between smooth and rough
            is_etched = (mask_noise > (0.2 - grain * 0.4)).float() 
            
            etching = is_etched * grit * 0.5
            
            noise_accum = torch.where(is_pitted, noise_accum - etching, noise_accum)

        # --- TYPE 3: GROWTH STEPS (Terracing) ---
        if is_stepped.any():
            # This simulates the "staircase" growth of crystals.
            # Math: Step = floor(value * density) / density
            
            density = 5.0 + grain * 15.0 # How many steps?
            
            # Create a base gradient (like a hill)
            # Combine U and V so steps run diagonally or organically
            gradient = u * 0.5 + v_norm * 0.5
            
            # Add some warp so steps aren't perfectly straight lines
            warp = noise1d_like_batch(v_norm * 5.0, corr=0.2, seed=seed_tex+2) * 0.1
            
            # The Quantization Step
            # "floor" creates the abrupt jump in height
            steps = torch.floor((gradient + warp) * density) / density
            
            # "steps" is a height map.
            # To make it a ROUGHNESS map (bump map), we need the edges to pop.
            # The gradient of a stepped function is 0 everywhere except the edge.
            # We want the flat terraces to be slightly different heights or roughnesses.
            
            # We create a "sawtooth" pattern for visual relief
            sawtooth = torch.fmod((gradient + warp) * density, 1.0)
            
            noise_accum = torch.where(is_stepped, noise_accum + sawtooth * 0.6, noise_accum)

        # --- Apply User Roughness Scalar ---
        # If roughness slider is low, we dampen the effect, but we DON'T flatten it completely.
        # We clamp minimum roughness to ensure the normal map always has data to work with.
        
        safe_roughness = torch.clamp(roughness, 0.1, 2.0) # Minimum 0.1 intensity
        
        # Center around 0.0 before scaling, then shift back, so scaling creates contrast
        # noise_accum is roughly 0.0-1.0.
        final_roughness = (noise_accum - 0.5) * safe_roughness
        
        # Return to map format
        roughness_map = final_roughness

        # --- 2. Transmission Map (Internal Defects) ---
        # (Kept similar, but sharpened)
        inclusions = batch.internal_inclusions.view(N, 1, 1)
        transmission_map = torch.ones_like(u)
        
        if torch.any(inclusions > 0):
             # Sharper cracks
             n_u = noise1d_like_batch(u * tex_scale_u * 0.8, corr=0.3, seed=seed_tex + 999)
             n_v = noise1d_like_batch(v_norm * tex_scale_v * 0.8, corr=0.3, seed=seed_tex + 888)
             
             # "Veins" logic: abs(noise) close to 0
             veins = 1.0 - torch.abs(n_u * n_v) 
             veins = torch.pow(veins, 10.0) # Sharpen significantly
             
             cloud_density = veins * inclusions
             transmission_map = torch.clamp(1.0 - cloud_density, 0.0, 1.0)
             
        # --- 3. Volumetric Turbidity (Fractal Clouds) ---
        # Initialize with base turbidity scalar
        turbidity_map = batch.turbidity.view(N, 1, 1).expand(-1, max_h, max_w).clone()
        
        has_turbidity = (batch.turbidity > 0.01)
        if has_turbidity.any():
            # Generate low-res noise volume
            # We use a fixed small size for efficiency
            VOL_D, VOL_H, VOL_W = 8, 128, 128
            
            # Global frequency for the volume generation
            # We want roughly 2-4 cloud blobs across the volume
            noise_vol = generate_fractal_noise_3d(
                (N, 1, VOL_D, VOL_H, VOL_W), 
                frequency=3.0, 
                octaves=3, 
                device=self.device,
                seed=seed_tex + 555
            )
            
            # Sampling Grid
            # Map u, v to volume coordinates [-1, 1]
            # u is [-1, 1] relative to L_half.
            # We scale it by fractal_scale to zoom in/out of the noise volume.
            # High fractal_scale -> Zoom in -> Lower frequency features
            # Low fractal_scale -> Zoom out -> Higher frequency
            
            # Default scale: 1.0 covers the whole volume.
            # We want noise to be continuous.
            
            scale_factor = 1.0 / (fractal_scale.view(N, 1, 1) + 0.1)
            
            # Jitter offset to sample different parts of volume
            t_rng = torch.Generator(device=self.device)
            t_rng.manual_seed(seed_tex + 123)
            u_offset = torch.rand(N, 1, 1, device=self.device, generator=t_rng) * 2.0 - 1.0
            v_offset = torch.rand(N, 1, 1, device=self.device, generator=t_rng) * 2.0 - 1.0
            
            grid_u = u * scale_factor + u_offset
            grid_v = v_norm * scale_factor + v_offset
            
            # Sample at 3 depths to simulate volume integration
            # z in [-1, 1]
            z_slices = [-0.5, 0.0, 0.5]
            accum_noise = torch.zeros_like(u)
            
            for z_val in z_slices:
                # Construct (N, H, W, 3) grid
                # z needs to be broadcast
                grid_z = torch.full_like(grid_u, z_val)
                
                # Stack for grid_sample: (x, y, z)
                grid = torch.stack([grid_u, grid_v, grid_z], dim=-1).unsqueeze(1) # (N, 1, H, W, 3)
                
                # Sample
                # padding_mode='reflection' ensures continuity
                sample = F.grid_sample(noise_vol, grid, align_corners=False, padding_mode='reflection')
                # sample is (N, 1, 1, H, W)
                accum_noise += sample.squeeze(1).squeeze(1)
                
            accum_noise /= len(z_slices)
            
            # Modulate turbidity
            # Shift noise to [0, 1] range (approx)
            # Fractal noise is normalized to std=1, mean=0.
            # We want "clouds".
            cloud_mod = torch.clamp(accum_noise * 0.5 + 0.5, 0.0, 1.0)
            
            # Contrast curve
            cloud_mod = torch.pow(cloud_mod, 1.5)
            
            # Apply to turbidity map
            # We only modulate where turbidity is active
            turbidity_map = torch.where(has_turbidity.view(N, 1, 1).expand(-1, max_h, max_w), 
                                      turbidity_map * cloud_mod * 2.0, # *2.0 to maintain average brightness
                                      turbidity_map)

        # --- 4. Polycrystalline Aggregates (Voronoi) ---
        # Uses 'internal_inclusions' > 0.5 as trigger? 
        # Or just mix it in if inclusions > 0?
        # Let's use a specific range or flag.
        # Roadmap says "Polycrystalline Aggregates".
        # We'll enable it if inclusions > 0 AND grain_size > 0.
        
        has_poly = (inclusions > 0) & (grain > 0.1)
        if has_poly.any():
            # Generate Voronoi Volume
            VOL_D, VOL_H, VOL_W = 4, 256, 256
            
            # Scale depends on grain_size
            # High grain_size = Large Cells (Low Freq)
            # Low grain_size = Small Cells (High Freq)
            
            # We want Zoom to be High when grain is Low.
            # grain=0 -> Zoom=10 (Tiny cells)
            # grain=1 -> Zoom=1 (Big cells)
            # grain=5 -> Zoom=0.2 (Huge cells)
            
            # Formula: zoom = 10.0 / (grain * 9.0 + 1.0)
            # Check: g=0 -> 10/1 = 10. g=1 -> 10/10 = 1. g=5 -> 10/46 ~ 0.2.
            
            zoom_val = 10.0 / (grain.view(N, 1, 1) * 9.0 + 1.0)
            
            # Generate Voronoi Volume
            VOL_D, VOL_H, VOL_W = 4, 256, 256
            
            voronoi_vol = generate_cellular_noise_3d(
                (N, 1, VOL_D, VOL_H, VOL_W),
                scale=5.0, # Base scale in noise space
                jitter=1.0,
                device=self.device,
                seed=seed_tex + 333
            )
            
            grid_u = u * zoom_val
            grid_v = v_norm * zoom_val
            grid_z = torch.zeros_like(grid_u) # Surface slice
            
            grid = torch.stack([grid_u, grid_v, grid_z], dim=-1).unsqueeze(1)
            
            # Sample distance (Worley noise)
            dist_map = F.grid_sample(voronoi_vol, grid, padding_mode='reflection', align_corners=False).squeeze(1).squeeze(1)
            
            # Edges are where dist is Max? No.
            # Cellular noise returns distance to nearest center.
            # Center = 0. Edge = Max.
            # So Edges are high values.
            
            # Threshold to get sharp cracks
            # Dist is approx 0 to 0.5 (radius of cell).
            # We want edges.
            
            edge_intensity = smooth_cap(dist_map, 0.2, 0.4) # 0 at center, 1 at edge
            
            # Modulate Transmission (Dark Edges)
            # Stronger effect if inclusions is high
            poly_strength = inclusions.view(N, 1, 1).expand(-1, max_h, max_w)
            transmission_map = torch.where(has_poly.view(N, 1, 1).expand(-1, max_h, max_w),
                                         transmission_map * (1.0 - edge_intensity * poly_strength),
                                         transmission_map)
                                         
            # Modulate Roughness (Rough Edges)
            roughness_map = torch.where(has_poly.view(N, 1, 1).expand(-1, max_h, max_w),
                                      torch.max(roughness_map, edge_intensity * poly_strength * 0.5),
                                      roughness_map)

        return roughness_map, transmission_map, turbidity_map