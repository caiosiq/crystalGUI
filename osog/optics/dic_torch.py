import torch
import torch.nn.functional as F
import math
import random
import numpy as np
from typing import Tuple, Union, Optional, Callable, List, Dict
from ..config import SynthConfig
from ..physics.particles import Rod, Debris, ParticleBatch, DebrisBatch, SHAPE_ROD
from ..physics.ghosts import GhostObject
from ..utils.math_torch import (
    noise1d_like, sin_wobble, kink, noisy_wobble, ragged_mask, smooth_cap, gaussian_blur_1d,
    noise1d_like_batch, sin_wobble_batch, kink_batch, noisy_wobble_batch, ragged_mask_batch
)
from .shaders.particle import ParticleShader
from .shaders.debris import DebrisShader

class DICModulatorTorch:
    def __init__(self, config: SynthConfig, device: str = "cpu"):
        self.cfg = config
        self.device = torch.device(device)
        self.particle_shader = ParticleShader(config, self.device)
        self.debris_shader = DebrisShader(config, self.device)
        
    def render(self, obj: Union[Rod, Debris, GhostObject], rng: random.Random, np_rng: np.random.RandomState) -> Tuple[Optional[torch.Tensor], int, int]:
        """
        Dispatch method to render any supported object type.
        """
        if isinstance(obj, GhostObject):
            wrapped = obj.wrapped_obj
            if isinstance(wrapped, Rod):
                return self.render_rod(obj, rng, np_rng)
            elif isinstance(wrapped, Debris):
                return self.render_debris(obj, rng)
            else:
                raise NotImplementedError(f"DICModulatorTorch: Ghost wraps unknown type {type(wrapped)}")

        if isinstance(obj, Rod):
            return self.render_rod(obj, rng, np_rng)
        elif isinstance(obj, Debris):
            return self.render_debris(obj, rng)
        else:
            raise NotImplementedError(f"DICModulatorTorch does not know how to render {type(obj)}")

    def render_rod(self, rod: Rod, rng: random.Random, np_rng: np.random.RandomState) -> Tuple[Optional[torch.Tensor], int, int]:
        
        cx = torch.tensor([rod.cx], device=self.device)
        cy = torch.tensor([rod.cy], device=self.device)
        z = torch.tensor([rod.z], device=self.device)
        L = torch.tensor([rod.L], device=self.device)
        W = torch.tensor([rod.W], device=self.device)
        H = W * 0.5 # Assume round
        ang = torch.tensor([rod.angle_deg], device=self.device)
        delta = torch.tensor([rod.delta], device=self.device)
        req = torch.tensor([rod.requires_label], device=self.device)
        
        # Defaults
        zero = torch.tensor([0.0], device=self.device)
        batch = ParticleBatch(
            cx=cx, cy=cy, z=z, L=L, W=W, H=H,
            alpha=ang, beta=zero, gamma=zero,
            delta=delta, requires_label=req,
            shape_id=torch.tensor([SHAPE_ROD], device=self.device, dtype=torch.long),
            curvature=torch.tensor([rod.curvature], device=self.device),
            width_jit_amp=torch.tensor([rod.width_jit_amp], device=self.device),
            edge_jit_amp=torch.tensor([rod.edge_jit_amp], device=self.device),
            offset_jit_amp=torch.tensor([rod.offset_jit_amp], device=self.device),
            ragged_p=torch.tensor([rod.ragged_p], device=self.device),
            ragged_corr=torch.tensor([rod.ragged_corr], device=self.device),
            polarity_flip_p=torch.tensor([rod.polarity_flip_p], device=self.device),
            shape_mode=torch.tensor([0], device=self.device, dtype=torch.long), # Map string later if needed
            seed=torch.tensor([rod.seed], device=self.device, dtype=torch.long),
            group_id=torch.tensor([0], device=self.device, dtype=torch.long)
        )
        
        # Map shape string to int
        SHAPE_MODE_MAP = {"straight": 0, "wavy": 1, "kink": 2, "noisy": 3}
        batch.shape_mode[0] = SHAPE_MODE_MAP.get(rod.shape_mode, 0)
        
        patches, x, y, _ = self.render_rods_batch(batch, rng, np_rng)
        if patches is None:
            return None, 0, 0
        return patches[0], int(x[0]), int(y[0])

    def render_debris(self, debris: Debris, rng: random.Random) -> Tuple[Optional[torch.Tensor], int, int]:
        # Legacy support via batch
        cx = torch.tensor([debris.cx], device=self.device)
        cy = torch.tensor([debris.cy], device=self.device)
        z = torch.tensor([debris.z], device=self.device)
        size = torch.tensor([debris.size_px], device=self.device)
        delta = torch.tensor([debris.delta], device=self.device)
        ang = torch.tensor([debris.angle_deg], device=self.device)
        dash = torch.tensor([debris.is_dash], device=self.device)
        seed = torch.tensor([debris.seed], device=self.device)
        
        batch = DebrisBatch(cx, cy, z, size, delta, ang, dash, seed)
        patches, x, y, _ = self.render_debris_batch(batch, rng)
        if patches is None:
            return None, 0, 0
        return patches[0], int(x[0]), int(y[0])

    def render_rods_batch(self, particles: Union[List[Rod], ParticleBatch], rng: random.Random, np_rng: np.random.RandomState, return_aux: bool = False) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Vectorized rendering for a batch of Particles.
        Delegates to ParticleShader.
        """
        if isinstance(particles, list):
            # Slow legacy path: convert list to batch
            # Not implemented in original code either
            if not particles: return None, torch.empty(0), torch.empty(0), {}
            return None, torch.empty(0), torch.empty(0), {}
            
        batch = particles
        return self.particle_shader.render_batch(batch, rng, return_aux)

    def render_debris_batch(self, debris: DebrisBatch, rng: random.Random, return_aux: bool = False) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Vectorized rendering for debris.
        Delegates to DebrisShader.
        """
        return self.debris_shader.render_batch(debris, rng, return_aux)
