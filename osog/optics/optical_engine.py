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
            
        # 2. Optical Pass
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

    def render_optics(self, g_buffer: torch.Tensor, aux: Dict[str, torch.Tensor], rng: random.Random, mode: str) -> torch.Tensor:
        """
        Dispatch optical simulation on an existing G-Buffer.
        Returns: (N, 3, H, W) image patch.
        """
        if mode == "dic":
            image = self.sim_dic(g_buffer, aux, rng)
        elif mode == "brightfield":
            image = self.sim_brightfield(g_buffer, aux)
        elif mode == "polarization":
            image = self.sim_polarization(g_buffer, aux)
        elif mode == "polarization_rgb":
            image = self.sim_polarization_rgb(g_buffer, aux)
        elif mode == "fluorescence":
            image = self.sim_fluorescence(g_buffer, aux)
        elif mode == "confocal":
            image = self.sim_confocal(g_buffer, aux)
        elif mode == "shadowgraphy":
            image = self.sim_shadowgraphy(g_buffer, aux)
        elif mode == "pvm":
            image = self.sim_pvm(g_buffer, aux, rng)
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

    def sim_dic(self, g_buffer: torch.Tensor, aux: Dict[str, torch.Tensor], rng: random.Random) -> torch.Tensor:
        """
        Simulate Differential Interference Contrast (DIC).
        Input: G-Buffer (Height, Mask, Delta, Orient)
        """
        height = g_buffer[:, 0]
        mask = g_buffer[:, 1]
        delta = g_buffer[:, 2] # Effective Delta (includes polarity)
        
        N, H, W = height.shape
        
        # Shadow params
        sh_gain = torch.empty(N, device=self.device).uniform_(*self.cfg.optics.shadow_gain)
        sh_gain = sh_gain.view(N, 1, 1)
        
        # Lighting Angle
        light_ang = math.radians(self.cfg.optics.lighting_angle_deg)
        lx, ly = math.cos(light_ang), math.sin(light_ang)
        
        # Gradients
        h_right = torch.roll(height, shifts=-1, dims=2)
        h_left  = torch.roll(height, shifts=1, dims=2)
        slope_x = (h_right - h_left) * 0.5 # Unit spacing assumption
        
        h_down = torch.roll(height, shifts=-1, dims=1)
        h_up   = torch.roll(height, shifts=1, dims=1)
        slope_y = (h_down - h_up) * 0.5
        
        # Directional Slope
        slope = slope_x * lx + slope_y * ly
        
        # Edge Steepness (for artifacts)
        edge_steepness = torch.sqrt(slope_x**2 + slope_y**2)
        scattering = torch.clamp(edge_steepness * 2.0 - 0.5, 0.0, 1.0)
        
        # Signal
        ref_delta = -0.15
        delta_factor = delta / ref_delta
        # Fix divide by zero if delta is 0? usually non-zero.
        
        base_signal = slope * delta_factor * sh_gain
        
        # Absorption
        absorption = torch.max(height * delta * 0.05, torch.tensor(-0.5, device=self.device))
        
        # Opacity
        opacity = aux['opacity'].squeeze(1)
        if torch.any(opacity > 0):
            absorption = absorption - 10.0 * height * opacity
            
        # Scattering / Dark Edges
        # is_metal logic? 
        # We need RI from somewhere? It's encoded in Delta? No.
        # We might need to add RI to G-Buffer or Aux.
        # For now, approximate scattering.
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
        crevice_mask = torch.clamp(curvature - 0.05, 0.0, 1.0) * mask
        
        layer = base_signal + absorption + dark_edges - crevice_mask * 5.0
        
        # Bubbles/Droplets overrides
        shape_id = aux['shape_id'].squeeze(1)
        is_bubble = (shape_id == SHAPE_BUBBLE)
        is_droplet = (shape_id == SHAPE_DROPLET)
        
        if is_bubble.any():
            edge_val = torch.clamp(1.0 - height, 0.0, 1.0)
            bubble_rim = -1.0 * (edge_val ** 6.0) * 4.0 * mask
            layer = torch.where(is_bubble, bubble_rim, layer)
            
        if is_droplet.any():
            edge_val = torch.clamp(1.0 - height, 0.0, 1.0)
            droplet_rim = -1.0 * (edge_val ** 3.0) * 0.8 * mask
            layer = torch.where(is_droplet, droplet_rim, layer)
        return layer
    def sim_brightfield(self, g_buffer: torch.Tensor, aux: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Simulate Brightfield using Vector Physics (Lambertian + Fresnel).
        Unifies Edge Detection and Face Shading into a single light interaction.
        """
        # --- UNPACK ---
        raw_height = g_buffer[:, 0] 
        mask = g_buffer[:, 1]
        delta_val = g_buffer[:, 2]
        
        # --- STEP 1: PHYSICAL VECTORS (The Correct Way) ---
        # Instead of 2D slopes, we build real 3D Surface Normals.
        # Normal = Cross Product of tangent vectors.
        # Approx: N = (-dx, -dy, 1.0)
        
        h_right = torch.roll(raw_height, shifts=-1, dims=2)
        h_left  = torch.roll(raw_height, shifts=1, dims=2)
        h_down  = torch.roll(raw_height, shifts=-1, dims=1)
        h_up    = torch.roll(raw_height, shifts=1, dims=1)
        
        # Calculate gradients
        dx = (h_right - h_left) * 0.5
        dy = (h_down - h_up) * 0.5
        
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
        
        # Let's use a "Relief" angle (mostly Z, slight top-left bias)
        light_vec = torch.tensor([0.4, -0.4, 1.2], device=g_buffer.device)
        light_vec = light_vec / torch.norm(light_vec) # Normalize
        
        # Reshape for broadcasting (1, 3, 1, 1)
        L = light_vec.view(1, 3, 1, 1)

        # --- STEP 3: LIGHT INTERACTION (The Dot Product) ---
        # "N dot L" tells us how much light hits the surface.
        # 1.0 = Facing light perfectly (Bright)
        # 0.0 = Perpendicular to light (Dark Edge)
        # < 0.0 = Facing away (Shadow)
        
        incidence = torch.abs(torch.sum(N * L, dim=1)) # (B, H, W)
        
        # --- STEP 4: REFRACTION CURVE ---
        # Real crystals aren't matte (Lambertian); they are glassy (Fresnel).
        # We map the incidence angle to transmission brightness.
        
        # 1. Bias: Shift so that 'flat' is 1.0 (Transparent)
        # 2. Contrast: How fast does it go dark when the angle gets steep?
        
        # steep slopes have incidence ~ 0.2
        # flat faces have incidence ~ 0.9
        
        # Tunable contrast curve
        transmission_surface = torch.clamp((incidence - 0.05) * 5, 0.0, 1.2)
        
        # Add the Delta (Refractive Index) influence
        # Higher index = More reflection/refraction = Darker edges
        transmission_surface = torch.pow(transmission_surface, 1.0 + delta_val)

        # --- STEP 5: VOLUME ABSORPTION (Beer-Lambert) ---
        # This answers your question: "Why use height?"
        # Because light travels THROUGH the crystal volume.
        # Thicker parts absorb more light. This is independent of surface angle.
        
        OPTICAL_SCALE = 0.002
        volume_absorption = torch.exp(-raw_height * OPTICAL_SCALE * (0.5 + delta_val))

        # --- STEP 6: COMPOSE ---
        # Final Light = Surface Interaction * Volume Absorption
        intensity = transmission_surface * volume_absorption

        # --- STEP 7: OUTPUT ---
        image_rgb = intensity.unsqueeze(1).repeat(1, 3, 1, 1)
        
        if 'absorption_color' in aux:
            image_rgb = image_rgb * aux['absorption_color']

        image_rgb = torch.clamp(image_rgb, 0.0, 1.2)
        delta_out = (image_rgb - 1.0) * 255.0
        delta_out = torch.clamp(delta_out, -255.0, 255.0)
        
        mask_3ch = mask.unsqueeze(1).repeat(1, 3, 1, 1)
        return delta_out * mask_3ch

    def sim_polarization(self, g_buffer: torch.Tensor, aux: Dict[str, torch.Tensor]) -> torch.Tensor:
        height = g_buffer[:, 0]
        orient = g_buffer[:, 3]
        bire = aux['birefringence'].squeeze(1)
        
        pol_ang = math.radians(self.cfg.optics.polarizer_angle_deg)
        theta = orient - pol_ang
        
        cross = torch.sin(2 * theta) ** 2
        
        # Retardation
        retardation_val = height * bire * 20.0
        retardation = torch.sin(retardation_val) ** 2
        
        return cross * retardation * 10.0

    def sim_polarization_rgb(self, g_buffer: torch.Tensor, aux: Dict[str, torch.Tensor]) -> torch.Tensor:
        height = g_buffer[:, 0]
        orient = g_buffer[:, 3]
        bire = aux['birefringence'].squeeze(1)
        
        pol_ang = math.radians(self.cfg.optics.polarizer_angle_deg)
        theta = orient - pol_ang
        
        intensity_mod = torch.sin(2 * theta) ** 2 + 0.05
        
        thickness_scale = 15000.0
        retardation_nm = height * thickness_scale * torch.abs(bire)
        
        def interference_intensity(ret_nm, lambda_nm):
             return torch.sin(math.pi * ret_nm / lambda_nm) ** 2

        i_r = (interference_intensity(retardation_nm, 600.0) + interference_intensity(retardation_nm, 630.0) + interference_intensity(retardation_nm, 660.0)) / 3.0
        i_g = (interference_intensity(retardation_nm, 500.0) + interference_intensity(retardation_nm, 530.0) + interference_intensity(retardation_nm, 560.0)) / 3.0
        i_b = (interference_intensity(retardation_nm, 420.0) + interference_intensity(retardation_nm, 450.0) + interference_intensity(retardation_nm, 480.0)) / 3.0
        
        rgb = torch.stack([i_r, i_g, i_b], dim=1)
        rgb = rgb * intensity_mod.unsqueeze(1) * 15.0
        return rgb

    def sim_fluorescence(self, g_buffer: torch.Tensor, aux: Dict[str, torch.Tensor]) -> torch.Tensor:
        height = g_buffer[:, 0]
        N, H, W = height.shape
        fluor_eff = torch.rand(N, device=self.device).view(N, 1, 1) * 0.8 + 0.2
        signal = height * fluor_eff
        zeros = torch.zeros_like(signal)
        return torch.stack([zeros, signal, zeros * 0.2], dim=1)

    def sim_confocal(self, g_buffer: torch.Tensor, aux: Dict[str, torch.Tensor]) -> torch.Tensor:
        height = g_buffer[:, 0]
        depth = aux['depth'].squeeze(1)
        N, H, W = height.shape
        
        fluor_eff = torch.rand(N, device=self.device).view(N, 1, 1) * 0.8 + 0.2
        signal = height * fluor_eff
        
        dist = torch.abs(depth - self.cfg.optics.focus_z)
        section_weight = torch.exp(-(dist**2) / (0.05**2))
        signal = signal * section_weight
        
        zeros = torch.zeros_like(signal)
        return torch.stack([zeros, signal, zeros], dim=1)

    def sim_shadowgraphy(self, g_buffer: torch.Tensor, aux: Dict[str, torch.Tensor]) -> torch.Tensor:
        height = g_buffer[:, 0]
        opacity = aux['opacity'].squeeze(1)
        depth = aux['depth'].squeeze(1)
        mask = g_buffer[:, 1]
        
        base_trans = 1.0 - (opacity * height)
        
        # Laplacian
        h_right = torch.roll(height, shifts=-1, dims=2)
        h_left  = torch.roll(height, shifts=1, dims=2)
        d2x = (h_right + h_left - 2*height) * 0.5 # 2nd deriv approx
        
        h_down = torch.roll(height, shifts=-1, dims=1)
        h_up   = torch.roll(height, shifts=1, dims=1)
        d2y = (h_down + h_up - 2*height) * 0.5
        
        laplacian = d2x + d2y
        
        defocus = depth - self.cfg.optics.focus_z
        defocus = defocus + 0.2 * torch.sign(defocus + 1e-6)
        
        shadow_signal = -50.0 * defocus * laplacian
        shadow_signal = torch.clamp(shadow_signal, -0.8, 2.0)
        
        layer = shadow_signal * base_trans
        
        # Bubbles override
        shape_id = aux['shape_id'].squeeze(1)
        is_bubble = (shape_id == SHAPE_BUBBLE)
        if is_bubble.any():
            # Approx U/V from height?
            # center bright spot
            # Just use height peak?
            lens_center = (height ** 4.0) * 2.0
            rim = -1.0 * (1.0 - height)**4 * 5.0
            layer = torch.where(is_bubble, layer + lens_center + rim, layer)
            
        return layer

    def sim_pvm(self, g_buffer: torch.Tensor, aux: Dict[str, torch.Tensor], rng: random.Random) -> torch.Tensor:
        """
        Reflectance (PVM) Mode.
        Physics: BRDF + Thin Film Interference + Laser Color.
        """
        height = g_buffer[:, 0]
        mask = g_buffer[:, 1]
        delta = g_buffer[:, 2] # (n - n_med)
        
        # Phase 4.3: Read new props
        reflectivity = aux.get('reflectivity', torch.full_like(height, 0.04))
        # Note: reflectivity is (N, 1, H, W) or (N, H, W) depending on how it was packed
        if reflectivity.dim() == 4: reflectivity = reflectivity.squeeze(1)
        
        roughness = aux.get('surf_rough', torch.zeros_like(height))
        if roughness.dim() == 4: roughness = roughness.squeeze(1)
        
        N, H, W = height.shape
        
        # 1. Compute Surface Normals
        # n = (-dh/dx, -dh/dy, 1) normalized
        h_right = torch.roll(height, shifts=-1, dims=2)
        h_left  = torch.roll(height, shifts=1, dims=2)
        dx = (h_right - h_left) * 0.5 * 20.0 
        
        h_down = torch.roll(height, shifts=-1, dims=1)
        h_up   = torch.roll(height, shifts=1, dims=1)
        dy = (h_down - h_up) * 0.5 * 20.0
        
        # Normal vector Z component
        dz = torch.ones_like(height)
        
        # Normalize
        norm = torch.sqrt(dx**2 + dy**2 + dz**2)
        nx, ny, nz = dx/norm, dy/norm, dz/norm
        
        # 2. Lighting Vector
        light_ang = math.radians(45.0) 
        lx, ly, lz = 0.2, -0.2, 0.95
        l_norm = math.sqrt(lx**2 + ly**2 + lz**2)
        lx, ly, lz = lx/l_norm, ly/l_norm, lz/l_norm
        
        # 3. Diffuse (Lambertian)
        diffuse = nx*lx + ny*ly + nz*lz
        diffuse = torch.clamp(diffuse, 0.0, 1.0)
        
        # 4. Specular (Sparkle)
        vx, vy, vz = 0.0, 0.0, 1.0 # View vector
        hx, hy, hz = lx+vx, ly+vy, lz+vz
        h_norm = math.sqrt(hx**2 + hy**2 + hz**2)
        hx, hy, hz = hx/h_norm, hy/h_norm, hz/h_norm
        
        dot_nh = nx*hx + ny*hy + nz*hz
        dot_nh = torch.clamp(dot_nh, 0.0, 1.0)
        
        # Roughness affects Specular Power (Glossiness)
        # Smooth (rough=0) -> Power 50 (Sharp)
        # Rough (rough=1) -> Power 2 (Broad)
        spec_power = 50.0 * (1.0 - roughness * 0.9)
        specular = dot_nh ** spec_power
        
        # Sparkle Mask
        sparkle_noise = torch.fmod(height * 1234.5678, 1.0)
        sparkle_thresh = 0.9 - 0.4 * roughness # Rougher -> More sparkles
        sparkle_mask = (sparkle_noise > sparkle_thresh).float()
        
        specular_intensity = specular * sparkle_mask * 5.0
        
        # Fresnel (Edge Glow)
        dot_nv = nz 
        fresnel = 1.0 - torch.abs(dot_nv)
        fresnel = torch.clamp(fresnel, 0.0, 1.0) ** 2.0 
        
        # 5. Thin-Film Interference (Iridescence)
        # Constructive interference when 2*n*d*cos(r) = m*lambda
        # We approximate for normal incidence/view: I ~ cos^2(k * n * d)
        lambda_nm = self.cfg.optics.laser_wavelength_nm
        # Convert height (microns) to nm -> * 1000
        # n_obj = n_med + delta
        n_obj = self.cfg.optics.medium_refractive_index + delta
        
        # Phase shift = 4 * pi * n * d / lambda
        phase = (4.0 * math.pi * n_obj * (height * 1000.0)) / lambda_nm
        interference = torch.cos(phase) ** 2
        
        # Modulate reflectivity by interference
        # Only strong for thin plates (where height is small and uniform)
        # We can blend it based on height or just apply globally
        # Effective Reflectance = Base + Amplitude * Interference
        eff_reflectivity = reflectivity * (0.5 + 0.5 * interference)
        
        # 6. Combine Intensity
        ambient = 0.1 + 0.2 * roughness # Rough surfaces scatter ambient light more
        
        # Apply reflectivity to diffuse/specular/fresnel components
        # High reflectivity materials (metals) shine brighter
        mat_gain = eff_reflectivity * 25.0 # Scale up for visibility (0.04 * 25 = 1.0)
        
        # Mix Diffuse and Specular based on roughness
        # Rough -> More diffuse
        diff_w = 0.4 + 0.6 * roughness
        spec_w = 1.0 - 0.5 * roughness
        
        intensity = ambient + (diffuse * diff_w + specular_intensity * spec_w + fresnel * 0.8) * mat_gain
        intensity = intensity * mask
        
        # 7. Apply Laser Color
        r, g, b = wavelength_to_rgb(lambda_nm)
        color_vec = torch.tensor([r, g, b], device=self.device).view(1, 3, 1, 1)
        
        # Expand intensity to RGB
        # (N, H, W) -> (N, 3, H, W)
        image_rgb = intensity.unsqueeze(1) * color_vec
        
        return image_rgb

    def apply_depth_of_field(self, image: torch.Tensor, aux: Dict[str, torch.Tensor]) -> torch.Tensor:
        depth = aux['depth'].squeeze(1)
        focus_z = self.cfg.optics.focus_z
        aperture = self.cfg.optics.aperture
        
        if aperture <= 0:
            return image
            
        dist = torch.abs(depth - focus_z)
        blur_sigs = dist * aperture * 5.0
        
        if torch.any(blur_sigs > 0.5):
            if image.dim() == 3:
                image = gaussian_blur_batch(image, blur_sigs)
            else:
                N, C, H, W = image.shape
                img_reshaped = image.view(N*C, H, W)
                sigs_reshaped = blur_sigs.repeat_interleave(C)
                img_blurred = gaussian_blur_batch(img_reshaped, sigs_reshaped)
                image = img_blurred.view(N, C, H, W)
                
        return image
