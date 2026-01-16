
import torch
import random
from ...config import SynthConfig
from ..particles import DebrisBatch

def generate_debris(cfg: SynthConfig, w: int, h: int, generator: torch.Generator, rng: random.Random) -> DebrisBatch:
    """Generates background debris/noise."""
    
    def rand_uniform(shape, min_val, max_val):
        return torch.rand(shape, generator=generator) * (max_val - min_val) + min_val

    ds = cfg.physics.debris
    if ds.rate > 0:
        n_deb = int(w * h * float(ds.rate))
        if n_deb > 0:
            dx = torch.randint(0, w, (n_deb,), generator=generator).float()
            dy = torch.randint(0, h, (n_deb,), generator=generator).float()
            delta = rand_uniform(n_deb, ds.int_delta[0], ds.int_delta[1])
            z = rand_uniform(n_deb, -1.0, 2.0)
            is_dash = torch.rand(n_deb, generator=generator) < ds.dash_prob
            angle = rand_uniform(n_deb, -90.0, 90.0)
            size = torch.randint(ds.size_px[0], ds.size_px[1], (n_deb,), generator=generator)
            seeds = torch.randint(0, 2**31-1, (n_deb,), generator=generator)
            
            return DebrisBatch(
                cx=dx, cy=dy, z=z, size_px=size, delta=delta, angle_deg=angle, is_dash=is_dash, seed=seeds
            )

    # Empty
    e = torch.empty(0)
    return DebrisBatch(e,e,e,e,e,e,e.bool(),e.long())
