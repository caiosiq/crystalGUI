from typing import Tuple, Dict, Optional
import torch
import random
import math
from ...config import SynthConfig
from ...physics.particles import ParticleBatch
from ...utils.math_torch import noise1d_like_batch

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
                      rng: random.Random) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates Geometric Crystal Textures (Growth Steps, Striations, Etching).
        """
        N = len(batch)
        seed_tex = rng.randint(0, 2**31 - 1)
        
        tex_type = batch.texture_type.view(N, 1, 1)
        roughness = batch.surf_roughness.view(N, 1, 1)
        grain = batch.grain_size.view(N, 1, 1)
        
        # Define types
        is_striated = (tex_type == 1) # Deep Ridges (Quartz/Tourmaline)
        is_pitted   = (tex_type == 2) # Acid Etching / Geometric Pits
        is_stepped  = (tex_type == 3) # Growth Steps / Terraces (Replaces Granular)
        
        # --- 1. Base Micro-Structure ---
        # Real crystals are never perfectly mathematically smooth. 
        # We add a tiny bit of "tooth" so lighting always catches something.
        micro_grain = torch.randn_like(u) * 0.02
        noise_accum = 0.5 + micro_grain # Center at 0.5 for mid-gray
        
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
             
        return roughness_map, transmission_map