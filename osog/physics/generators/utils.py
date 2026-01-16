
import torch
import random
from ..constants import SHAPE_ROD, SHAPE_PLATE, SHAPE_CUBE, SHAPE_SPHERE, SHAPE_BUBBLE, SHAPE_DROPLET

def rand_uniform(shape, min_val, max_val, generator):
    return torch.rand(shape, generator=generator) * (max_val - min_val) + min_val

def gen_3d_params(n: int, cfg, generator: torch.Generator):
    """Generates 3D shape and orientation parameters."""
    if n == 0: 
        return torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0)
    
    # Shape ID
    r = torch.rand(n, generator=generator)
    s_id = torch.full((n,), SHAPE_ROD, dtype=torch.long)
    
    # Thresholds
    prob_plate = cfg.physics.rods.prob_plate
    prob_cube = cfg.physics.rods.prob_cube
    prob_sphere = cfg.physics.rods.prob_sphere
    prob_bubble = cfg.physics.rods.prob_bubble
    prob_droplet = cfg.physics.rods.prob_droplet

    t1 = prob_plate
    t2 = t1 + prob_cube
    t3 = t2 + prob_sphere
    t4 = t3 + prob_bubble
    t5 = t4 + prob_droplet
    
    s_id[r < t5] = SHAPE_DROPLET
    s_id[r < t4] = SHAPE_BUBBLE
    s_id[r < t3] = SHAPE_SPHERE
    s_id[r < t2] = SHAPE_CUBE
    s_id[r < t1] = SHAPE_PLATE
    
    # Thickness
    th_lo, th_hi = cfg.physics.rods.thickness_ratio_lo_hi
    thickness_ratio = rand_uniform(n, th_lo, th_hi, generator)
    
    # Orientation
    alpha = rand_uniform(n, -90.0, 90.0, generator)
    if cfg.physics.rods.enable_3d:
        beta = rand_uniform(n, -90.0, 90.0, generator) # Tumble
        gamma = rand_uniform(n, -180.0, 180.0, generator) # Roll
    else:
        beta = torch.zeros(n)
        gamma = torch.zeros(n)
        
    return s_id, thickness_ratio, alpha, beta, gamma
