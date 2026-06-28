
import math
from typing import Tuple, List, Dict, Any, Union
import random
import numpy as np
import torch
from ..config import SynthConfig
from .stage import apply_stage_to_config, params_for_t, sample_lambda, lambda_to_t, lerp, ensure_config
from .particles import ParticleBatch, DebrisBatch, Agglomerate
from .generators.main_generator import generate_main_particles
from .generators.ghost_generator import generate_ghosts
from .generators.debris_generator import generate_debris
from .generators.incrustation_generator import generate_incrustations
from .generators.fused_generator import generate_fused_clusters, generate_attached_bubbles, generate_coalesced_droplets


# Re-export stage helpers (used by batch jobs and legacy imports)
from .stage import sample_lambda, lambda_to_t, params_for_t, apply_stage_to_config, lerp


def generate_distribution(cfg: SynthConfig, t: float, rng: random.Random, np_rng: np.random.RandomState, device: str = 'cpu') -> Tuple[ParticleBatch, DebrisBatch, List[Agglomerate]]:
    """
    Main entry point for particle generation.
    Delegates to specific generators for Main Particles, Ghosts, Debris, and Clusters.
    """
    if isinstance(cfg, dict):
        cfg = ensure_config(cfg)

    cfg = apply_stage_to_config(cfg, t)
    
    w, h = cfg.canvas.width, cfg.canvas.height
    seed = rng.randint(0, 2**32 - 1)
    gen = torch.Generator(device='cpu')
    gen.manual_seed(seed)
    
    # 1. Generate Main Particles (Rods, Spheres, etc.)
    particles = generate_main_particles(cfg, w, h, gen, rng)
    
    # 2. Generate Fused Clusters (Agglomeration)
    fused = generate_fused_clusters(cfg, particles, gen, rng)
    
    # Merge fused into main particles
    for k in particles.keys():
        if fused[k]:
            particles[k].extend(fused[k])
            
    # 2.5 Generate Attached Bubbles (Phase 1)
    attached_bubbles = generate_attached_bubbles(cfg, particles, gen, rng)
    for k in particles.keys():
        if attached_bubbles[k]:
            particles[k].extend(attached_bubbles[k])
            
    # Phase 4.4.2.3.1: Generate Surface Incrustations
    incrustations = generate_incrustations(cfg, particles, gen, rng)
    for k in particles.keys():
        if incrustations[k]:
            particles[k].extend(incrustations[k])

    # 3. Generate Ghosts (Background noise)
    # Calculate total main particles to determine ghost fraction
    n_main = sum([len(t) for t in particles["cx"]])
    ghosts = generate_ghosts(cfg, n_main, w, h, gen, rng)
    
    # Merge ghosts
    for k in particles.keys():
        if ghosts[k]:
            particles[k].extend(ghosts[k])
            
    # 4. Assemble ParticleBatch
    if len(particles["cx"]) > 0:
        def cat(lst, dtype=torch.float32):
            return torch.cat(lst).to(dtype=dtype)
        
        # Flatten group ids properly; preserve -1 sentinel for ghost particles
        all_gids = []
        offset = 0
        for g in particles["group_id"]:
            if g.numel() > 0 and torch.all(g == -1):
                all_gids.append(g)
            else:
                all_gids.append(g + offset)
                offset += len(g)
            
        batch = ParticleBatch(
            cx=cat(particles["cx"]), cy=cat(particles["cy"]), z=cat(particles["z"]),
            L=cat(particles["L"]), W=cat(particles["W"]), H=cat(particles["H"]),
            alpha=cat(particles["alpha"]), beta=cat(particles["beta"]), gamma=cat(particles["gamma"]),
            delta=cat(particles["delta"]),
            requires_label=cat(particles["req_label"], dtype=torch.bool),
            shape_id=cat(particles["shape_id"], dtype=torch.long),
            curvature=cat(particles["curv"]),
            width_jit_amp=cat(particles["w_jit"]),
            edge_jit_amp=cat(particles["edge_jit"]),
            offset_jit_amp=cat(particles["off_jit"]),
            ragged_p=cat(particles["rag_p"]),
            ragged_corr=cat(particles["rag_corr"]),
            polarity_flip_p=cat(particles["pol_p"]),
            shape_mode=cat(particles["shape_mode"], dtype=torch.long),
            corner_round=cat(particles["corner_round"]),
            corner_bend=cat(particles["corner_bend"]),
            
            # New Material Fields
            refractive_index=cat(particles["ref_index"]),
            birefringence=cat(particles["birefringence"]),
            opacity=cat(particles["opacity"]),
            
            # Phase 4.3
            reflectivity=cat(particles["reflectivity"]),
            dispersion=cat(particles["dispersion"]),
            absorption_color=cat(particles["absorption_color"]),
            
            texture_type=cat(particles["tex_type"], dtype=torch.long),
            surf_roughness=cat(particles["surf_rough"]),
            grain_size=cat(particles["grain_size"]),
            internal_inclusions=cat(particles["inclusions"]),
            turbidity=cat(particles["turbidity"]), # Phase 4.4.2.1
            
            anisotropy=cat(particles["anisotropy"]),
            anisotropy_angle=cat(particles["anisotropy_angle"]),
            
            seed=cat(particles["seed"], dtype=torch.long),
            group_id=cat(all_gids, dtype=torch.long)
        )
    else:
        # Empty Batch
        e = torch.empty(0)
        batch = ParticleBatch(
            e,e,e,e,e,e,e,e,e,e,e.bool(),e.long(),e,e,e,e,e,e,e,e.long(),
            e, e, e, e, e, e, e.long(), e, e, e, e, # New fields incl Phase 4.3 + Turbidity
            e, e, # Anisotropy
            e, e, # Corner deformations
            e.long(),e.long()
        )
        
    if device != 'cpu':
        batch.to(device)

    # 5. Generate Debris
    debris_batch = generate_debris(cfg, w, h, gen, rng)
    if device != 'cpu':
        debris_batch.to(device)
        
    return batch, debris_batch, []

# Alias for compatibility if needed
generate_distribution_from_specs = generate_distribution
