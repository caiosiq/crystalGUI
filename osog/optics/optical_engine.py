import torch
import torch.nn.functional as F
import math
import random
import numpy as np
from typing import Tuple, Union, Optional, Callable, List, Dict
from ..config import SynthConfig
from ..physics.particles import Rod, Debris, ParticleBatch, DebrisBatch, SHAPE_BUBBLE, SHAPE_DROPLET, SHAPE_PLATE, SHAPE_CUBE
from .utils import gaussian_blur_batch
from .shaders.geometry import GeometryShader
from .shaders.debris import DebrisShader
from .shaders.texture import TextureShader

def wavelength_to_rgb(wavelength_nm: float) -> Tuple[float, float, float]:
    """Convert wavelength to RGB approximation."""
    w = float(wavelength_nm)
    if w >= 380 and w < 440:
        r, g, b = -(w - 440) / (440 - 380), 0.0, 1.0
    elif w >= 440 and w < 490:
        r, g, b = 0.0, (w - 440) / (490 - 440), 1.0
    elif w >= 490 and w < 510:
        r, g, b = 0.0, 1.0, -(w - 510) / (510 - 490)
    elif w >= 510 and w < 580:
        r, g, b = (w - 510) / (580 - 510), 1.0, 0.0
    elif w >= 580 and w < 645:
        r, g, b = 1.0, -(w - 645) / (645 - 580), 0.0
    elif w >= 645 and w <= 780:
        r, g, b = 1.0, 0.0, 0.0
    else:
        r, g, b = 0.0, 0.0, 0.0
    return r, g, b

class OpticalEngine:
    def __init__(self, config: SynthConfig, device: str = "cpu"):
        self.cfg = config
        self.device = torch.device(device)
        self.geometry_shader = GeometryShader(config, self.device)
        self.debris_shader = DebrisShader(config, self.device)
        self.texture_shader = TextureShader(config, self.device)
        
    @staticmethod
    def soft_clamp(x: torch.Tensor, min_val: float, max_val: float, temp: float = 1.0) -> torch.Tensor:
        """
        Differentiable Soft Clamp using Softplus Difference.
        Maps (-inf, inf) -> (min_val, max_val).
        Approximation: min + (softplus(x-min) - softplus(x-max))
        """
        # Beta controls sharpness. Higher = sharper.
        # temp is "softness", so beta ~ 1/temp.
        # Use a high base beta to ensure 0 maps close to 0.
        beta = 10.0 / (temp + 1e-6)
        
        range_val = max_val - min_val
        u = x - min_val
        
        # clamp(u, 0, range) ~ softplus(u) - softplus(u - range)
        clamped_u = F.softplus(u, beta=beta) - F.softplus(u - range_val, beta=beta)
        
        return min_val + clamped_u

    def _ghost_slope_gain(self, aux: Dict[str, torch.Tensor], n: int) -> torch.Tensor:
        """Per-particle slope multiplier (1.0 for mains, slope_gain for ghosts)."""
        is_ghost = aux.get('is_ghost')
        if is_ghost is None:
            return torch.ones(n, device=self.device)
        slope_gain = float(self.cfg.physics.ghosts.slope_gain)
        if slope_gain == 1.0 and not bool(torch.any(is_ghost)):
            return torch.ones(n, device=self.device)
        is_g = is_ghost.to(self.device)
        return torch.where(
            is_g,
            torch.full((n,), slope_gain, device=self.device),
            torch.ones(n, device=self.device),
        )

    def _height_gradients(
        self,
        height: torch.Tensor,
        aux: Dict[str, torch.Tensor],
        bump_scale: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Height gradients with optional roughness bump and ghost slope gain."""
        h_right = torch.roll(height, shifts=-1, dims=2)
        h_left = torch.roll(height, shifts=1, dims=2)
        h_down = torch.roll(height, shifts=-1, dims=1)
        h_up = torch.roll(height, shifts=1, dims=1)
        dx = (h_right - h_left) * 0.5
        dy = (h_down - h_up) * 0.5

        if bump_scale > 0:
            roughness_map = aux.get('roughness_map', torch.zeros_like(height))
            if roughness_map.dim() == 4:
                roughness_map = roughness_map.squeeze(1)
            if torch.any(roughness_map != 0):
                r_right = torch.roll(roughness_map, shifts=-1, dims=2)
                r_left = torch.roll(roughness_map, shifts=1, dims=2)
                r_down = torch.roll(roughness_map, shifts=-1, dims=1)
                r_up = torch.roll(roughness_map, shifts=1, dims=1)
                dr_dx = (r_right - r_left) * 0.5
                dr_dy = (r_down - r_up) * 0.5
                dx = dx + dr_dx * bump_scale
                dy = dy + dr_dy * bump_scale

        gain = self._ghost_slope_gain(aux, height.shape[0]).view(-1, 1, 1)
        return dx * gain, dy * gain
        
    def render_batch(self, particles: ParticleBatch, rng: random.Random, mode: str = "dic") -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Main entry point for rendering a batch of particles.
        1. Runs Geometry Pass -> G-Buffer
        2. Runs Optical Pass -> Image Patch
        """
        # 1. Geometry Pass
        g_buffer, x_mins, y_mins, aux = self.geometry_shader.render_batch(particles, rng)
        
        if g_buffer is None:
            return None, torch.empty(0), torch.empty(0), {}
        
        # 3. Optical Pass
        patch = self.render_optics(g_buffer, aux, rng, mode)
             
        # Aux output for Heads
        aux_out = {}
        if 'height' not in aux_out:
             # g_buffer[:, 0] is height
             aux_out['height'] = g_buffer[:, 0:1]
        if 'mask' not in aux_out:
             aux_out['mask'] = g_buffer[:, 1:2]
        if 'depth' in aux:
             aux_out['depth'] = aux['depth'].unsqueeze(1)
             
        return patch, x_mins, y_mins, aux_out

    def render_optics(self, g_buffer: torch.Tensor, aux: Dict[str, torch.Tensor], rng: random.Random, mode: str, soft_mode: bool = False) -> torch.Tensor:
        """
        Dispatch optical simulation on an existing G-Buffer.
        Returns: (N, 3, H, W) image patch.
        """
        if mode == "dic":
            image = self.sim_dic(g_buffer, aux, rng, soft_mode=soft_mode)
        elif mode == "brightfield":
            image = self.sim_brightfield(g_buffer, aux, soft_mode=soft_mode)
        elif mode == "blaze":
            image = self.sim_blaze(g_buffer, aux, rng, soft_mode=soft_mode)
        else:
            raise ValueError(f"Unknown optical mode: {mode}")

        # 3. Blur / Depth of Field
        image = self.apply_depth_of_field(image, aux)
        
        # 4. Format Output
        # Ensure (N, 3, H, W)
        if image.dim() == 3:
             patch = image.unsqueeze(1).repeat(1, 3, 1, 1)
        else:
             patch = image
             
        return patch

    def render_debris_batch(self, debris: DebrisBatch, rng: random.Random, return_aux: bool = False) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Vectorized rendering for debris.
        Delegates to DebrisShader.
        """
        return self.debris_shader.render_batch(debris, rng, return_aux)

    def sim_dic(self, g_buffer: torch.Tensor, aux: Dict[str, torch.Tensor], rng: random.Random, soft_mode: bool = False) -> torch.Tensor:
        """
        Simulate Differential Interference Contrast (DIC).
        Input: G-Buffer (Height, Mask, Delta, Orient)
        """
        height = g_buffer[:, 0]
        mask = g_buffer[:, 1]
        delta = g_buffer[:, 2] # Effective Delta (includes polarity)
        
        N, H, W = height.shape
        
        # Shadow params
        # Replace uniform_ with differentiable sampling: min + rand * (max - min)
        sh_gain_min = self.cfg.optics.shadow_gain[0]
        sh_gain_max = self.cfg.optics.shadow_gain[1]
        
        # Ensure we use the tensors if they are tensors
        # If they are scalars, this still works
        rand_val = torch.rand(N, device=self.device)
        sh_gain = sh_gain_min + rand_val * (sh_gain_max - sh_gain_min)
        sh_gain = sh_gain.view(N, 1, 1)
        
        # Lighting Angle
        light_ang = math.radians(self.cfg.optics.lighting_angle_deg)
        lx, ly = math.cos(light_ang), math.sin(light_ang)
        
        slope_x, slope_y = self._height_gradients(height, aux, bump_scale=2.0)
        
        # Directional Slope
        slope = slope_x * lx + slope_y * ly
        
        # Edge Steepness (for artifacts)
        edge_steepness = torch.sqrt(slope_x**2 + slope_y**2)
        if soft_mode:
            scattering = self.soft_clamp(edge_steepness * 2.0 - 0.5, 0.0, 1.0)
        else:
            scattering = torch.clamp(edge_steepness * 2.0 - 0.5, 0.0, 1.0)
        
        # Signal
        ref_delta = -0.15
        delta_factor = delta / ref_delta
        
        # Apply Transmission Map (Internal Defects) to Signal
        # Dark defects reduce signal amplitude
        transmission_map = aux.get('transmission_map', torch.ones_like(height))
        if transmission_map.dim() == 4: transmission_map = transmission_map.squeeze(1)
        
        # Turbidity adds noise/scattering (reduces contrast)
        turbidity = aux.get('turbidity', torch.zeros_like(height))
        if turbidity.dim() == 4: turbidity = turbidity.squeeze(1)
        
        # Combined "Clarity" factor
        clarity = transmission_map * torch.exp(-turbidity * 2.0)
        
        base_signal = slope * delta_factor * sh_gain * clarity
        
        # Absorption
        absorption = torch.max(height * delta * 0.05, torch.tensor(-0.5, device=self.device))
        
        # Opacity
        opacity = aux['opacity'].squeeze(1)
        if torch.any(opacity > 0):
            absorption = absorption - 10.0 * height * opacity
            
        dark_edges = -1.0 * scattering * 0.8
        
        # Corner Glint (Recalculate or store?)
        # For simplicity, skip complex corner glint in v1 port, or re-implement if critical.
        # The logic relies on (u,v) which are lost in G-Buffer unless we store them.
        # Phase 4.25 goal is Architecture. We can add glint back later or add (u,v) to G-Buffer.
        
        # Grain Boundaries (Curvature)
        sx_right = torch.roll(slope_x, shifts=-1, dims=2)
        sx_left  = torch.roll(slope_x, shifts=1, dims=2)
        d2x = (sx_right - sx_left) * 0.5
        
        sy_down = torch.roll(slope_y, shifts=-1, dims=1)
        sy_up   = torch.roll(slope_y, shifts=1, dims=1)
        d2y = (sy_down - sy_up) * 0.5
        
        curvature = d2x + d2y
        if soft_mode:
            crevice_mask = self.soft_clamp(curvature - 0.05, 0.0, 1.0) * mask
        else:
            crevice_mask = torch.clamp(curvature - 0.05, 0.0, 1.0) * mask
        
        layer = base_signal + absorption + dark_edges - crevice_mask * 5.0
        
        # Bubbles/Droplets overrides
        shape_id = aux['shape_id'].squeeze(1)
        is_bubble = (shape_id == SHAPE_BUBBLE)
        is_droplet = (shape_id == SHAPE_DROPLET)
        
        if is_bubble.any():
            if soft_mode:
                edge_val = self.soft_clamp(1.0 - height, 0.0, 1.0)
            else:
                edge_val = torch.clamp(1.0 - height, 0.0, 1.0)
            bubble_rim = -1.0 * (edge_val ** 6.0) * 4.0 * mask
            layer = torch.where(is_bubble, bubble_rim, layer)
            
        if is_droplet.any():
            if soft_mode:
                edge_val = self.soft_clamp(1.0 - height, 0.0, 1.0)
            else:
                edge_val = torch.clamp(1.0 - height, 0.0, 1.0)
            droplet_rim = -1.0 * (edge_val ** 3.0) * 0.8 * mask
            layer = torch.where(is_droplet, droplet_rim, layer)
        return layer
    def sim_brightfield(self, g_buffer: torch.Tensor, aux: Dict[str, torch.Tensor], soft_mode: bool = False) -> torch.Tensor:
        """
        Simulate Spectral Brightfield using Vector Physics (Lambertian + Fresnel).
        Now supports Chromatic Aberration (Dispersion) by rendering R/G/B passes with varying RI.
        """
        # --- UNPACK ---
        raw_height = g_buffer[:, 0] 
        mask = g_buffer[:, 1]
        base_delta = g_buffer[:, 2] # This is delta for Green (approx 550nm)
        
        # Get Dispersion (default to 0 if not present)
        # Dispersion represents the RI spread (n_blue - n_red)
        dispersion = aux.get('dispersion', torch.zeros_like(base_delta)).squeeze(1)
        
        # --- STEP 1: PHYSICAL VECTORS (Shared across wavelengths) ---
        # Instead of 2D slopes, we build real 3D Surface Normals.
        # Normal = Cross Product of tangent vectors.
        # Approx: N = (-dx, -dy, 1.0)
        dx, dy = self._height_gradients(raw_height, aux, bump_scale=5.0)
        
        # Construct 3D Normal Vector (N)
        # We need to balance 'pixel units' vs 'depth units'.
        # A scaling factor of 1.0 means a 45-degree slope is normal.
        Z_SCALE = 1.0 
        
        # Stack into (B, 3, H, W)
        normal_map = torch.stack([-dx, -dy, torch.ones_like(dx) * Z_SCALE], dim=1)
        
        # Normalize to length 1.0
        # This is crucial for correct lighting math.
        norm = torch.norm(normal_map, dim=1, keepdim=True)
        N = normal_map / (norm + 1e-6)

        # --- STEP 2: THE LIGHT VECTOR (L) ---
        # Here is where we control the physics.
        # Pure Brightfield: [0, 0, 1] -> Flat look, symmetric edges.
        # Oblique Brightfield: [0.3, -0.3, 0.9] -> 3D look, asymmetric shading.
        
        # Use Configured Light Direction (Phase 4.4.2)
        l_cfg = self.cfg.optics.light_direction
        light_vec = torch.tensor([l_cfg[0], l_cfg[1], l_cfg[2]], device=g_buffer.device)
        light_vec = light_vec / (torch.norm(light_vec) + 1e-6) # Normalize
        
        # Reshape for broadcasting (1, 3, 1, 1)
        L = light_vec.view(1, 3, 1, 1)

        # --- STEP 3: LIGHT INTERACTION (The Dot Product) ---
        # "N dot L" tells us how much light hits the surface.
        # 1.0 = Facing light perfectly (Bright)
        # 0.0 = Perpendicular to light (Dark Edge)
        # < 0.0 = Facing away (Shadow)
        
        incidence = torch.abs(torch.sum(N * L, dim=1)) # (B, H, W)
        
        # --- STEP 4: BASE REFRACTION CURVE ---
        # Real crystals aren't matte (Lambertian); they are glassy (Fresnel).
        # We map the incidence angle to transmission brightness.
        
        # 1. Bias: Shift so that 'flat' is 1.0 (Transparent)
        # 2. Contrast: How fast does it go dark when the angle gets steep?
        
        # steep slopes have incidence ~ 0.2
        # flat faces have incidence ~ 0.9
        
        # Tunable contrast curve
        # transmission_base = torch.clamp((incidence - 0.05) * 5, 0.0, 1.2)
        
        # --- PHASE 4.4.2.1: PHYSICAL FRESNEL (Refractive Index Matching) ---
        # T ~ (1 - R)^2
        # R = ((n1 cos_i - n2 cos_t) / (n1 cos_i + n2 cos_t))^2
        
        n1 = self.cfg.optics.medium_refractive_index
        n2 = n1 + base_delta # n_obj
        
        # Avoid n2=0 or negative
        if soft_mode:
            n2 = self.soft_clamp(n2, 1.0, 3.0)
        else:
            n2 = torch.clamp(n2, 1.0, 3.0)
        
        # Cos(theta_i) = incidence
        cos_i = incidence
        sin_i_sq = 1.0 - cos_i**2
        
        # Snell's Law: n1 sin_i = n2 sin_t
        # sin_t = (n1/n2) sin_i
        # cos_t = sqrt(1 - sin_t^2)
        
        n_ratio = n1 / n2
        sin_t_sq = (n_ratio**2) * sin_i_sq
        
        # Total Internal Reflection check
        # If sin_t_sq > 1.0, TIR occurs -> R=1.0, T=0.0
        is_tir = (sin_t_sq > 1.0).float()
        
        if soft_mode:
            cos_t = torch.sqrt(self.soft_clamp(1.0 - sin_t_sq, 0.0, 1.0))
        else:
            cos_t = torch.sqrt(torch.clamp(1.0 - sin_t_sq, 0.0, 1.0))
        
        # Fresnel Equations (Unpolarized average)
        # Rs = ((n1 cos_i - n2 cos_t) / (n1 cos_i + n2 cos_t))^2
        # Rp = ((n1 cos_t - n2 cos_i) / (n1 cos_t + n2 cos_i))^2
        
        rs_num = n1 * cos_i - n2 * cos_t
        rs_den = n1 * cos_i + n2 * cos_t
        Rs = (rs_num / (rs_den + 1e-6))**2
        
        rp_num = n1 * cos_t - n2 * cos_i
        rp_den = n1 * cos_t + n2 * cos_i
        Rp = (rp_num / (rp_den + 1e-6))**2
        
        R = 0.5 * (Rs + Rp)
        
        # Apply TIR
        R = torch.where(sin_t_sq > 1.0, torch.ones_like(R), R)
        
        # Transmission (Two interfaces: In and Out)
        # T_total = (1-R)^2 approx
        transmission_base = (1.0 - R)**2
        
        # Enhance contrast for visualization (Microscopes have finite NA which captures some refracted light)
        
        # Boost transmission slightly to avoid pitch black edges unless TIR
        transmission_base = transmission_base * 1.2
        if soft_mode:
            transmission_base = self.soft_clamp(transmission_base, 0.0, 1.2)
        else:
            transmission_base = torch.clamp(transmission_base, 0.0, 1.2)

        # --- CAUSTICS (Internal Focusing) ---
        # Crystals act as lenses. Convex shapes focus light (Hotspots).
        # We approximate this using surface curvature (Laplacian).
        # Top of hill (convex): Laplacian < 0. We want brightness > 0.
        h_right = torch.roll(raw_height, shifts=-1, dims=2)
        h_left = torch.roll(raw_height, shifts=1, dims=2)
        h_down = torch.roll(raw_height, shifts=-1, dims=1)
        h_up = torch.roll(raw_height, shifts=1, dims=1)
        d2x = h_right + h_left - 2.0 * raw_height
        d2y = h_down + h_up - 2.0 * raw_height
        laplacian = d2x + d2y
        
        # Caustic intensity: Focus light where surface is convex
        # We clamp to avoid extreme values
        if soft_mode:
            caustics = self.soft_clamp(-laplacian * 150.0, -0.5, 0.8)
        else:
            caustics = torch.clamp(-laplacian * 150.0, -0.5, 0.8)
        
        # --- FRESNEL RIM LIGHTING ---
        # 100% reflection at glancing angles.
        # Normal Z-component tells us how "flat" the surface is relative to view.
        # 1.0 = Flat (face-on), 0.0 = Edge (glancing).
        # We want brightness when nz is low.
        
        nz = N[:, 2] # (N, H, W)
        fresnel_rim = (1.0 - nz) ** 3.0 # Sharpness power 3.0
        if soft_mode:
            fresnel_rim = self.soft_clamp(fresnel_rim, 0.0, 1.0)
        else:
            fresnel_rim = torch.clamp(fresnel_rim, 0.0, 1.0)
        
        # --- SPECTRAL VECTORIZATION ---
        # We define relative delta shifts for R, G, B
        # Blue (High Freq) bends more -> Higher RI -> Higher Delta
        # Red (Low Freq) bends less -> Lower RI -> Lower Delta
        
        # Shifts: Red (-0.5), Green (0.0), Blue (+0.5)
        # Shape: (1, 3, 1, 1) for broadcasting against (N, 1, H, W)
        shifts = torch.tensor([-0.5, 0.0, 0.5], device=g_buffer.device).view(1, 3, 1, 1)
        
        OPTICAL_SCALE = 0.002
        
        # Expand inputs to (N, 1, H, W) to broadcast with shifts
        # base_delta: (N, H, W) -> (N, 1, H, W)
        # dispersion: (N, H, W) -> (N, 1, H, W)
        
        base_delta_exp = base_delta.unsqueeze(1)
        dispersion_exp = dispersion.unsqueeze(1)
        
        # Scale up dispersion for artistic visibility
        # Real dispersion is subtle (~0.01). We want visible rainbows.
        DISPERSION_SCALE_GEO = 20.0 # For refraction (edges)
        DISPERSION_SCALE_ABS = 150.0 # For volume absorption (body color) - Aggressive!
        
        chromatic_shift = shifts * dispersion_exp
        
        # Effective Delta for Refraction (Bending)
        eff_delta_geo = base_delta_exp + chromatic_shift * DISPERSION_SCALE_GEO
        
        # Effective Delta for Absorption (Color)
        eff_delta_abs = base_delta_exp + chromatic_shift * DISPERSION_SCALE_ABS
        
        # --- REFRACTION MODULATION ---
        # transmission_base: (N, H, W) -> (N, 1, H, W)
        trans_channel = torch.pow(transmission_base.unsqueeze(1), 1.0 + eff_delta_geo)
        
        # --- VOLUME ABSORPTION ---
        # Physical Model: Beer-Lambert with Solvent Displacement (Phase 4.4.2)
        # T_rel = exp(-h * (mu_particle - mu_solvent))
        
        transmission_map = aux.get('transmission_map', torch.ones_like(raw_height)).unsqueeze(1)
        turbidity = aux.get('turbidity', torch.zeros_like(raw_height)).unsqueeze(1)
        
        # 1. Solvent Mu
        mu_solvent = torch.zeros((1, 3, 1, 1), device=g_buffer.device)
        if hasattr(self.cfg.sensor, 'solvent_color'):
             c = self.cfg.sensor.solvent_color
             c_t = torch.tensor(c, device=g_buffer.device).float() / 255.0
             if soft_mode:
                 mu_solvent = -torch.log(self.soft_clamp(c_t, 1e-4, 1.0)).view(1, 3, 1, 1)
             else:
                 mu_solvent = -torch.log(torch.clamp(c_t, 1e-4, 1.0)).view(1, 3, 1, 1)
             
        # 2. Particle Mu
        # aux['absorption_color'] is usually (N, 3) or (N, 3, 1, 1)
        if 'absorption_color' in aux:
             ac = aux['absorption_color']
             if ac.dim() == 2: ac = ac.view(-1, 3, 1, 1)
             if soft_mode:
                 mu_particle = -torch.log(self.soft_clamp(ac, 1e-4, 1.0))
             else:
                 mu_particle = -torch.log(torch.clamp(ac, 1e-4, 1.0))
        else:
             mu_particle = torch.zeros_like(mu_solvent)
             
        # 3. Delta Mu (N, 3, 1, 1)
        delta_mu = mu_particle - mu_solvent
        
        # 4. Apply
        # Scale factor: 1.0 micron height -> visible color change
        ABS_SCALE = 0.5 
        
        vol_abs_channel = torch.exp(-raw_height.unsqueeze(1) * delta_mu * ABS_SCALE)
        
        # Add Turbidity (always lossy) & Transmission Map
        vol_abs_channel = vol_abs_channel * torch.exp(-raw_height.unsqueeze(1) * turbidity * 5.0)
        vol_abs_channel = vol_abs_channel * transmission_map
        
        # --- CAUSTICS ---
        # caustics: (N, H, W) -> (N, 1, H, W)
        caustic_factor = 1.0 + eff_delta_geo * 2.0
        
        # --- COMPOSE ---
        # Final Intensity: (N, 3, H, W)
        image_rgb = trans_channel * vol_abs_channel + caustics.unsqueeze(1) * caustic_factor
        
        # if 'absorption_color' in aux:
        #    image_rgb = image_rgb * aux['absorption_color']

        image_rgb = torch.clamp(image_rgb, 0.0, 1.2)
        delta_out = (image_rgb - 1.0) * 255.0
        delta_out = torch.clamp(delta_out, -255.0, 255.0)
        
        mask_3ch = mask.unsqueeze(1).repeat(1, 3, 1, 1)
        return delta_out * mask_3ch

    def sim_blaze(self, g_buffer: torch.Tensor, aux: Dict[str, torch.Tensor], rng: random.Random, soft_mode: bool = False) -> torch.Tensor:
        """
        Simulate Blaze / Darkfield Reflectance Probe using Waveguiding Physics.
        
        Mechanism A: Surface Glow (Roughness scattering grazing light)
        Mechanism B: Edge Leakage (TIR Failure at sharp edges - "Neon Outline")
        Mechanism C: Subsurface Scattering (Volume diffusion gated by Injection)
        """
        # --- ROBUST INPUT HANDLING ---
        height = g_buffer[:, 0]
        mask = g_buffer[:, 1]
        base_delta = g_buffer[:, 2]
        
        N_batch, H, W = height.shape
        device = height.device

        # --- PROPS ---
        reflectivity = aux.get('reflectivity', torch.full_like(height, 0.04))
        if reflectivity.dim() == 4: reflectivity = reflectivity.squeeze(1)
        if reflectivity.shape[-2:] != (H, W):
            reflectivity = F.interpolate(reflectivity.unsqueeze(1), size=(H,W), mode='nearest').squeeze(1)
            
        roughness = aux.get('surf_rough', torch.zeros_like(height))
        if roughness.dim() == 4: roughness = roughness.squeeze(1)
        if roughness.shape[-2:] != (H, W):
             roughness = F.interpolate(roughness.unsqueeze(1), size=(H,W), mode='bilinear').squeeze(1)

        turbidity = aux.get('turbidity', torch.zeros_like(height))
        if turbidity.dim() == 4: turbidity = turbidity.squeeze(1)
        if turbidity.shape[-2:] != (H, W):
             turbidity = F.interpolate(turbidity.unsqueeze(1), size=(H,W), mode='nearest').squeeze(1)
             
        dispersion = aux.get('dispersion', torch.zeros_like(height))
        if dispersion.dim() == 4: dispersion = dispersion.squeeze(1)
        if dispersion.shape[-2:] != (H, W):
             dispersion = F.interpolate(dispersion.unsqueeze(1), size=(H,W), mode='nearest').squeeze(1)

        # --- 1. GEOMETRY & CURVATURE ---
        h_pad = F.pad(height, (1, 1, 1, 1), mode='replicate')
        
        # GRADIENT (Slope) - "Injection Aperture"
        # We use a high scale to ensure even thin rods register as having "sides"
        GRAD_SCALE = 8.0 
        dx = (h_pad[:, 1:-1, 2:] - h_pad[:, 1:-1, :-2]) * 0.5 * GRAD_SCALE
        dy = (h_pad[:, 2:, 1:-1] - h_pad[:, :-2, 1:-1]) * 0.5 * GRAD_SCALE
        
        # CURVATURE (Laplacian) - Key for Edge Leakage
        # ∇²H = d²H/dx² + d²H/dy²
        # High curvature = Sharp Edge = Light Leakage point
        d2x = h_pad[:, 1:-1, 2:] + h_pad[:, 1:-1, :-2] - 2.0 * height
        d2y = h_pad[:, 2:, 1:-1] + h_pad[:, :-2, 1:-1] - 2.0 * height
        # Scale curvature for visibility. Clamp to 0 (Convex bumps glow, concave valleys don't)
        if soft_mode:
            curvature = self.soft_clamp(-(d2x + d2y) * 100.0, 0.0, 1.0)
        else:
            curvature = torch.clamp(-(d2x + d2y) * 100.0, 0.0, 1.0) 
        
        slope_mag = torch.sqrt(dx**2 + dy**2)
        
        # Perturb Normals with Roughness
        if torch.any(roughness != 0):
             r_pad = F.pad(roughness, (1, 1, 1, 1), mode='replicate')
             dr_dx = (r_pad[:, 1:-1, 2:] - r_pad[:, 1:-1, :-2]) * 0.5
             dr_dy = (r_pad[:, 2:, 1:-1] - r_pad[:, :-2, 1:-1]) * 0.5
             
             # Increase Bump Scale significantly to match PVM intensity
             # Real striations are deep grooves, not subtle scratches.
             BUMP_SCALE_BLAZE = 40.0 
             dx = dx + dr_dx * BUMP_SCALE_BLAZE
             dy = dy + dr_dy * BUMP_SCALE_BLAZE

        slope_gain = self._ghost_slope_gain(aux, N_batch).view(-1, 1, 1)
        dx = dx * slope_gain
        dy = dy * slope_gain
             
        dz = torch.ones_like(height)
        
        # Expand dx/dy for vectorization
        dx_base = dx.unsqueeze(1).unsqueeze(1)
        dy_base = dy.unsqueeze(1).unsqueeze(1)
        dz_base = dz.unsqueeze(1).unsqueeze(1)

        # --- 2. LIGHTING VECTORS ---
        # Grazing angle (15 deg) ensures light hits the sides of particles
        RING_ANGLE_DEG = 15.0 
        N_RING_SAMPLES = 8
        LIGHT_INTENSITY = 3500.0 
        
        theta_rad = math.radians(RING_ANGLE_DEG)
        sin_t, cos_t = math.sin(theta_rad), math.cos(theta_rad)
        
        phi = torch.linspace(0, 2*math.pi, N_RING_SAMPLES + 1, device=device)[:-1]
        lx = sin_t * torch.cos(phi)
        ly = sin_t * torch.sin(phi)
        lz = torch.full_like(lx, cos_t)
        
        L_ring = torch.stack([lx, ly, lz], dim=1).view(1, N_RING_SAMPLES, 3, 1, 1)

        # --- 3. SPECULAR FLASH (Surface Reflection) ---
        n_shifts = torch.tensor([-0.1, 0.0, 0.1], device=device).view(1, 3, 1, 1, 1)
        disp_mag = dispersion.unsqueeze(1).unsqueeze(1) * 20.0 
        
        factor = 1.0 + n_shifts * disp_mag
        dx_s = dx_base * factor
        dy_s = dy_base * factor
        dz_s = dz_base
        
        norm_s = torch.sqrt(dx_s**2 + dy_s**2 + dz_s**2)
        nx_s = dx_s / norm_s
        ny_s = dy_s / norm_s
        nz_s = dz_s / norm_s
        
        # Half Vector (Blinn-Phong)
        hx = L_ring[:, :, 0, :, :].unsqueeze(1)
        hy = L_ring[:, :, 1, :, :].unsqueeze(1)
        hz = L_ring[:, :, 2, :, :].unsqueeze(1) + 1.0
        h_norm = torch.sqrt(hx**2 + hy**2 + hz**2)
        hx, hy, hz = hx/h_norm, hy/h_norm, hz/h_norm
        
        ndoth = nx_s * hx + ny_s * hy + nz_s * hz
        
        # Wide specular lobe (40.0) matches "frosted glass" look
        spec_power = 40.0 * (1.0 - roughness * 0.5) 
        spec_power = spec_power.unsqueeze(1).unsqueeze(1)
        
        if soft_mode:
            spec = torch.pow(self.soft_clamp(ndoth, 0.0, 1.0), spec_power)
        else:
            spec = torch.pow(torch.clamp(ndoth, 0.0, 1.0), spec_power)
        specular_rgb = spec.mean(dim=2) 
        
        # Fresnel
        nz_base = dz / torch.sqrt(dx**2 + dy**2 + dz**2)
        n_med = self.cfg.optics.medium_refractive_index
        n_obj = n_med + base_delta
        R0 = ((n_obj - n_med) / (n_obj + n_med + 1e-6)) ** 2
        fresnel_surf = R0 + (1.0 - R0) * (1.0 - nz_base)**5
        
        specular_flash = specular_rgb * fresnel_surf.unsqueeze(1) * reflectivity.unsqueeze(1) * LIGHT_INTENSITY

        # --- 4. DIFFUSE GLOW COMPONENTS (The Physics Fix) ---

        # A. LIGHT INJECTION FACTOR
        # Light enters where the surface is steep (sides). Flat tops reflect light away.
        if soft_mode:
            injection_factor = self.soft_clamp(slope_mag * 0.5, 0.0, 1.0)
        else:
            injection_factor = torch.clamp(slope_mag * 0.5, 0.0, 1.0) 
        
        # --- Phase 4.4.2.3.1: Stochastic Injection ---
        # Modulate injection by surface texture to break uniformity.
        # Rough areas trap/scatter light into the volume more effectively.
        if torch.any(roughness != 0):
             stochastic_gate = 0.5 + 0.5 * roughness
             injection_factor = injection_factor * stochastic_gate

        # B. WAVEGUIDE / CORNER LEAKAGE (For Clear Crystals)
        # Light trapped inside leaks out at high curvature points (Edges/Corners).
        # It glows even if turbidity is zero.
        # (injection_factor * 0.5 + 0.1) allows for some "ambient" light trapping even on flat parts.
        waveguide_glow = (injection_factor * 0.5 + 0.1) * curvature * (LIGHT_INTENSITY * 0.08)

        # C. SURFACE ROUGHNESS GLOW (For Horizontal Rods)
        # Rough surfaces scatter grazing light into the camera. 
        # This makes the "Body" of the rod glow, not just the edges.
        surface_glow = roughness * slope_mag * (LIGHT_INTENSITY * 0.05)

        # D. SUBSURFACE SCATTERING (For Turbid Crystals)
        # Only happens if light enters (Injection) AND there is stuff to hit (Turbidity).
        # We decouple injection_factor slightly to allow ambient volume glow
        # SSS should happen even if surface is flat (diffuse entry)
        
        sss_mod = 1.0 + roughness * 3.0 
        
        # Use a softer injection factor for SSS (0.2 base)
        sss_injection = injection_factor * 0.8 + 0.2
        
        sss_glow = sss_injection * turbidity * height * (LIGHT_INTENSITY * 0.2) * sss_mod

        # --- COMPOSE ---
        diffuse_sum = waveguide_glow + surface_glow + sss_glow
        diffuse_base = diffuse_sum.unsqueeze(1).repeat(1, 3, 1, 1)
        
        signal = (specular_flash + diffuse_base) * mask.unsqueeze(1)
        
        # Noise
        noise_level = self.cfg.optics.noise_scale
        shot_noise = torch.randn_like(signal) * noise_level
        
        # Apply mask to noise as well to ensure transparent background
        shot_noise = shot_noise * mask.unsqueeze(1)
        
        if soft_mode:
            image = self.soft_clamp(signal + shot_noise, 0.0, 255.0)
        else:
            image = torch.clamp(signal + shot_noise, 0.0, 255.0)
        return image

    def apply_depth_of_field(self, image: torch.Tensor, aux: Dict[str, torch.Tensor]) -> torch.Tensor:
        depth = aux['depth'].squeeze(1)
        focus_z = self.cfg.optics.focus_z
        aperture = self.cfg.optics.aperture
        ghost_sig = float(self.cfg.physics.ghosts.blur_sigma)
        is_ghost = aux.get('is_ghost')

        particle_depth = depth.mean(dim=(1, 2))
        dist = torch.abs(particle_depth - focus_z)
        blur_sigs = dist * aperture * 5.0

        if is_ghost is not None and ghost_sig > 0 and torch.any(is_ghost):
            is_g = is_ghost.to(blur_sigs.device)
            ghost_blur = torch.maximum(
                blur_sigs,
                torch.tensor(ghost_sig, device=blur_sigs.device, dtype=blur_sigs.dtype),
            ) + dist * ghost_sig
            blur_sigs = torch.where(is_g, ghost_blur, blur_sigs)

        if not torch.any(blur_sigs > 0.5):
            return image

        if image.dim() == 3:
            image = gaussian_blur_batch(image, blur_sigs)
        else:
            N, C, H, W = image.shape
            img_reshaped = image.view(N * C, H, W)
            sigs_reshaped = blur_sigs.repeat_interleave(C)
            img_blurred = gaussian_blur_batch(img_reshaped, sigs_reshaped)
            image = img_blurred.view(N, C, H, W)

        return image
