from typing import Dict, Any, List, Tuple, Union, Optional
import math
import random
import numpy as np
import torch
from ..config import SynthConfig
from .particles import Rod, Agglomerate, RenderableObject, Debris, RodBatch, DebrisBatch
from .ghosts import GhostObject

SHAPE_MODE_MAP = {"straight": 0, "wavy": 1, "kink": 2, "noisy": 3}
REVERSE_SHAPE_MODE_MAP = {0: "straight", 1: "wavy", 2: "kink", 3: "noisy"}

def lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)

def sample_lambda(rng: random.Random, cfg: SynthConfig) -> float:
    # Handle dict config if passed (legacy)
    if isinstance(cfg, dict):
        cfg = SynthConfig.from_flat_dict(cfg)
        
    lo, hi = cfg.physics.stage_lambda_range
    log_lo, log_hi = math.log10(lo), math.log10(hi)
    return 10 ** rng.uniform(log_lo, log_hi)


def lambda_to_t(lmbda: float) -> float:
    t = (math.log10(max(1e-6, lmbda)) + 1.0) / 2.0
    return float(np.clip(t, 0.0, 1.0))


def params_for_t(cfg: Union[SynthConfig, Dict[str, Any]], t: float) -> Dict[str, Any]:
    if isinstance(cfg, dict):
        cfg = SynthConfig.from_flat_dict(cfg)
        
    t = float(max(0.0, min(1.0, t)))

    def _lo_and_hi_t(range_val):
        if isinstance(range_val, (list, tuple)):
            if len(range_val) == 3:
                lo, hi0, hi1 = range_val
                lo = float(lo)
                hi_t = float(lerp(float(hi0), float(hi1), t))
                return lo, max(lo, hi_t)
            elif len(range_val) == 2:
                lo, hi = range_val
                lo = float(lo); hi = float(hi)
                return lo, max(lo, hi)
        try:
            lo = float(range_val)
        except Exception:
            lo = 0.0
        return lo, lo

    n_lo, n_hi_t = _lo_and_hi_t(cfg.physics.rods.n_rods_rng_lo_hi)
    n_rods = int(round(lerp(n_lo, n_hi_t, t)))

    L_lo, L_hi_t = _lo_and_hi_t(cfg.physics.rods.rod_len_px_lo_hi)
    len_hi = float(L_hi_t)

    ar_lo, ar_hi_t = _lo_and_hi_t(cfg.physics.rods.rod_aspect_lo_hi)
    ar_hi_val = float(ar_hi_t)

    d_lo, d_hi_t = _lo_and_hi_t(cfg.physics.rods.rod_delta_rng)

    p_fused = lerp(cfg.physics.fused.p0, cfg.physics.fused.p1, t)

    return {
        "n_rods_min": int(round(n_lo)),
        "n_rods_max": int(round(n_hi_t)),
        "n_rods": int(n_rods),
        "rod_len_min": float(L_lo),
        "rod_len_max": float(len_hi),
        "rod_aspect_min": float(ar_lo),
        "rod_aspect_max": float(ar_hi_val),
        "rod_delta_min": float(d_lo),
        "rod_delta_max": float(d_hi_t),
        "p_fused": float(p_fused),
        "t": t,
    }

def generate_distribution(cfg: SynthConfig, t: float, rng: random.Random, np_rng: np.random.RandomState, device: str = 'cpu') -> Tuple[RodBatch, DebrisBatch, List[Agglomerate]]:
    """
    Vectorized generation of particle distribution.
    Returns:
        rods_batch: Tensor collection of all rods (singles + fused + ghosts)
        debris_batch: Tensor collection of debris
        agglomerates: Metadata list of agglomerate groupings (useful for OBBs)
    """
    if isinstance(cfg, dict):
        cfg = SynthConfig.from_flat_dict(cfg)
        
    p = params_for_t(cfg, t)
    
    # Setup Torch RNG
    # Extract a seed from the passed Python RNG to maintain determinism
    seed = rng.randint(0, 2**32 - 1)
    # We use a CPU generator for distribution logic to ensure reproducibility across platforms
    # and because distribution generation is lightweight compared to rendering.
    # We will move tensors to 'device' at the end.
    gen = torch.Generator(device='cpu')
    gen.manual_seed(seed)
    
    # Helper for uniform
    def rand_uniform(shape, min_val, max_val):
        return torch.rand(shape, generator=gen) * (max_val - min_val) + min_val

    n_rods_nominal = int(p.get("n_rods", 0))
    if "n_rods_min" in p and "n_rods_max" in p:
        # We can't use torch.randint with generator easily for single scalar?
        # Use python rng for n_rods count to match old logic
        try:
            n_rods_nominal = rng.randint(int(p["n_rods_min"]), int(p["n_rods_max"]))
        except ValueError:
            n_rods_nominal = int(p.get("n_rods", 0))

    w, h = cfg.canvas.width, cfg.canvas.height
    
    # -------------------------------------------------------------------------
    # 1. Main Rods (Singles + Agglomerates)
    # -------------------------------------------------------------------------
    rods_cx = []
    rods_cy = []
    rods_z = []
    rods_L = []
    rods_W = []
    rods_angle = []
    rods_delta = []
    rods_seed = []
    rods_group_id = []
    rods_req_label = []
    
    # Optional params
    rods_curv = []
    rods_w_jit = []
    rods_off_jit = []
    rods_edge_jit = []
    rods_pol_p = []
    rods_rag_p = []
    rods_rag_corr = []
    rods_shape = []

    # Agglomerate metadata for OBBs
    agglomerates_meta = []

    if cfg.physics.rods.enable and n_rods_nominal > 0:
        # Determine Fused vs Single
        # Vectorized decision
        p_fused = p["p_fused"]
        
        # We generate n_rods_nominal "seeds" or "parents".
        # Some become singles, some become agglomerates.
        is_fused = torch.rand(n_rods_nominal, generator=gen) < p_fused
        if not cfg.physics.fused.enable:
            is_fused[:] = False
            
        n_fused = is_fused.sum().item()
        n_single = n_rods_nominal - n_fused
        
        # --- Singles ---
        if n_single > 0:
            L = rand_uniform(n_single, p["rod_len_min"], p["rod_len_max"])
            asp = rand_uniform(n_single, p["rod_aspect_min"], p["rod_aspect_max"])
            W = torch.maximum(torch.tensor(2.0), L * asp)
            cx = rand_uniform(n_single, 0.05 * w, 0.95 * w)
            cy = rand_uniform(n_single, 0.05 * h, 0.95 * h)
            ang = rand_uniform(n_single, -90.0, 90.0)
            delta = rand_uniform(n_single, p["rod_delta_min"], p["rod_delta_max"])
            z = rand_uniform(n_single, -0.1, 0.1)
            seeds = torch.randint(0, 2**31-1, (n_single,), generator=gen)
            
            # Unique Group IDs for singles (e.g. 0 to n_single-1)
            # We'll offset these later
            g_ids = torch.arange(n_single)
            
            rods_cx.append(cx)
            rods_cy.append(cy)
            rods_z.append(z)
            rods_L.append(L)
            rods_W.append(W)
            rods_angle.append(ang)
            rods_delta.append(delta)
            rods_seed.append(seeds)
            rods_group_id.append(g_ids)
            rods_req_label.append(torch.ones(n_single, dtype=torch.bool))
            
            # Defaults for optional
            rods_curv.append(torch.zeros(n_single))
            rods_w_jit.append(torch.zeros(n_single))
            rods_off_jit.append(torch.zeros(n_single))
            rods_edge_jit.append(torch.zeros(n_single))
            rods_pol_p.append(torch.zeros(n_single))
            rods_rag_p.append(torch.zeros(n_single))
            rods_rag_corr.append(torch.full((n_single,), 0.2))
            rods_shape.append(torch.full((n_single,), SHAPE_MODE_MAP["straight"], dtype=torch.long))

        # --- Fused (Agglomerates) ---
        if n_fused > 0:
            # Parents
            p_L = rand_uniform(n_fused, p["rod_len_min"], p["rod_len_max"])
            p_asp = rand_uniform(n_fused, p["rod_aspect_min"], p["rod_aspect_max"])
            p_W = torch.maximum(torch.tensor(2.0), p_L * p_asp)
            p_cx = rand_uniform(n_fused, 0.05 * w, 0.95 * w)
            p_cy = rand_uniform(n_fused, 0.05 * h, 0.95 * h)
            p_ang = rand_uniform(n_fused, -90.0, 90.0)
            p_delta = rand_uniform(n_fused, p["rod_delta_min"], p["rod_delta_max"])
            p_z = rand_uniform(n_fused, -0.1, 0.1)
            
            # Number of arms per agglomerate (2 to 5)
            n_arms = torch.randint(2, 6, (n_fused,), generator=gen)
            
            # We need to replicate parent properties for each arm
            repeats = n_arms
            
            # Parent IDs for grouping
            # Start after singles
            start_id = n_single
            parent_ids = torch.arange(start_id, start_id + n_fused)
            
            # Expanded Parents
            ex_cx = p_cx.repeat_interleave(repeats)
            ex_cy = p_cy.repeat_interleave(repeats)
            ex_z = p_z.repeat_interleave(repeats)
            ex_L = p_L.repeat_interleave(repeats)
            ex_W = p_W.repeat_interleave(repeats)
            ex_ang = p_ang.repeat_interleave(repeats)
            ex_delta = p_delta.repeat_interleave(repeats)
            ex_gid = parent_ids.repeat_interleave(repeats)
            
            total_children = n_arms.sum().item()
            
            # Generate Child Deviations
            spread = rand_uniform(n_fused, 10.0, 55.0).repeat_interleave(repeats)
            d_ang = torch.randn(total_children, generator=gen) * (spread * 0.25)
            
            child_ang = ex_ang + d_ang
            
            # Child L, W variations
            # Normal distribution manually:
            # L_i = L + normal(0, 0.25*L) = L * (1 + normal(0, 0.25))
            noise_L = torch.randn(total_children, generator=gen) * 0.25
            child_L = torch.maximum(torch.tensor(8.0), ex_L * (1.0 + noise_L))
            
            noise_W = torch.randn(total_children, generator=gen) * 0.25
            child_W = torch.maximum(torch.tensor(3.0), ex_W * (1.0 + noise_W))
            
            # Position offset
            # r uniform(0, 0.25*W)
            r = rand_uniform(total_children, 0.0, 1.0) * (0.25 * ex_W)
            # Offset angle is a + 90
            off_ang_rad = torch.deg2rad(child_ang + 90.0)
            child_cx = ex_cx + r * torch.cos(off_ang_rad)
            child_cy = ex_cy + r * torch.sin(off_ang_rad)
            
            child_seeds = torch.randint(0, 2**31-1, (total_children,), generator=gen)
            
            rods_cx.append(child_cx)
            rods_cy.append(child_cy)
            rods_z.append(ex_z)
            rods_L.append(child_L)
            rods_W.append(child_W)
            rods_angle.append(child_ang)
            rods_delta.append(ex_delta)
            rods_seed.append(child_seeds)
            rods_group_id.append(ex_gid)
            rods_req_label.append(torch.ones(total_children, dtype=torch.bool))
            
            # Defaults
            rods_curv.append(torch.zeros(total_children))
            rods_w_jit.append(torch.zeros(total_children))
            rods_off_jit.append(torch.zeros(total_children))
            rods_edge_jit.append(torch.zeros(total_children))
            rods_pol_p.append(torch.zeros(total_children))
            rods_rag_p.append(torch.zeros(total_children))
            rods_rag_corr.append(torch.full((total_children,), 0.2))
            rods_shape.append(torch.full((total_children,), SHAPE_MODE_MAP["straight"], dtype=torch.long))
            
            # Store Agglomerate Metadata (for OBB reconstruction later if needed)
            # We don't have the Rod objects here, but we can reconstruct them later or 
            # the pipeline can just treat them as visual rods.
            # To support 'Agglomerate' object in pipeline list for OBBs, we need to handle that.
            # But we are moving to pure batch rendering. OBB generation usually happens
            # on the generated data. If the user wants GT OBBs, we need the logic.
            # For now, let's just generate the list of Agglomerate objects with dummy children?
            # Or better: pipeline.generate calls this, gets Batches.
            # If return_obbs is True, we need to convert Batches back to CPU objects or dicts.
            # We can use 'group_id' to reconstruct.

    # -------------------------------------------------------------------------
    # 2. Ghosts
    # -------------------------------------------------------------------------
    if cfg.physics.ghosts.enable and cfg.physics.ghosts.fraction > 0:
        n_ghosts = max(0, int(round(n_rods_nominal * float(cfg.physics.ghosts.fraction))))
        
        if n_ghosts > 0:
            Lg = rand_uniform(n_ghosts, 12.0, p["rod_len_max"])
            arg = rand_uniform(n_ghosts, 0.01, min(0.8, p["rod_aspect_max"] * 2))
            Wg = torch.maximum(torch.tensor(2.0), Lg * arg)
            cxg = rand_uniform(n_ghosts, 0.05 * w, 0.95 * w)
            cyg = rand_uniform(n_ghosts, 0.05 * h, 0.95 * h)
            angg = rand_uniform(n_ghosts, -90.0, 90.0)
            delt_g = rand_uniform(n_ghosts, cfg.physics.ghosts.delta_rng[0], cfg.physics.ghosts.delta_rng[1])
            z = rand_uniform(n_ghosts, 0.5, 2.0)
            
            seeds = torch.randint(0, 2**31-1, (n_ghosts,), generator=gen)
            
            # Ghosts have negative group ID to separate? Or just distinct.
            # Let's use -1 to indicate "No Group / Noise"
            g_ids = torch.full((n_ghosts,), -1, dtype=torch.long)
            
            rods_cx.append(cxg)
            rods_cy.append(cyg)
            rods_z.append(z)
            rods_L.append(Lg)
            rods_W.append(Wg)
            rods_angle.append(angg)
            rods_delta.append(delt_g)
            rods_seed.append(seeds)
            rods_group_id.append(g_ids)
            rods_req_label.append(torch.zeros(n_ghosts, dtype=torch.bool)) # Ghosts don't need labels
            
            # Ghost specific params
            rods_w_jit.append(torch.full((n_ghosts,), cfg.physics.ghosts.width_jit_amp))
            rods_off_jit.append(torch.full((n_ghosts,), cfg.physics.ghosts.offset_jit_amp))
            rods_edge_jit.append(torch.full((n_ghosts,), cfg.physics.ghosts.edge_jit_amp))
            rods_curv.append(rand_uniform(n_ghosts, cfg.physics.ghosts.curve_kappa_range[0], cfg.physics.ghosts.curve_kappa_range[1]))
            rods_rag_p.append(torch.full((n_ghosts,), cfg.physics.ghosts.ragged_p))
            rods_rag_corr.append(torch.full((n_ghosts,), cfg.physics.ghosts.ragged_corr))
            rods_pol_p.append(torch.zeros(n_ghosts)) # Not in config? Default 0
            
            # Shape modes
            # ["wavy", "kink", "noisy", "straight"]
            modes = torch.randint(0, 4, (n_ghosts,), generator=gen) # 0..3
            rods_shape.append(modes)

    # -------------------------------------------------------------------------
    # 3. Assemble RodBatch
    # -------------------------------------------------------------------------
    if len(rods_cx) > 0:
        def cat(lst, dtype=torch.float32):
            return torch.cat(lst).to(dtype=dtype)
            
        rod_batch = RodBatch(
            cx=cat(rods_cx),
            cy=cat(rods_cy),
            z=cat(rods_z),
            L=cat(rods_L),
            W=cat(rods_W),
            angle_deg=cat(rods_angle),
            delta=cat(rods_delta),
            requires_label=cat(rods_req_label, dtype=torch.bool),
            curvature=cat(rods_curv),
            width_jit_amp=cat(rods_w_jit),
            edge_jit_amp=cat(rods_edge_jit),
            offset_jit_amp=cat(rods_off_jit),
            ragged_p=cat(rods_rag_p),
            ragged_corr=cat(rods_rag_corr),
            polarity_flip_p=cat(rods_pol_p),
            shape_mode=cat(rods_shape, dtype=torch.long),
            seed=cat(rods_seed, dtype=torch.long),
            group_id=cat(rods_group_id, dtype=torch.long)
        )
    else:
        # Empty batch
        e = torch.empty(0)
        rod_batch = RodBatch(e,e,e,e,e,e,e,e.bool(),e,e,e,e,e,e,e,e.long(),e.long(),e.long())

    # -------------------------------------------------------------------------
    # 4. Debris
    # -------------------------------------------------------------------------
    deb_cx = []
    deb_cy = []
    deb_z = []
    deb_size = []
    deb_delta = []
    deb_angle = []
    deb_dash = []
    deb_seed = []
    
    if cfg.physics.debris.rate > 0:
        n_deb = int(w * h * float(cfg.physics.debris.rate))
        if n_deb > 0:
            dx = torch.randint(0, w, (n_deb,), generator=gen).float()
            dy = torch.randint(0, h, (n_deb,), generator=gen).float()
            delta = rand_uniform(n_deb, cfg.physics.debris.int_delta[0], cfg.physics.debris.int_delta[1])
            z = rand_uniform(n_deb, -1.0, 2.0)
            is_dash = torch.rand(n_deb, generator=gen) < cfg.physics.debris.dash_prob
            angle = rand_uniform(n_deb, -90.0, 90.0)
            size = torch.randint(cfg.physics.debris.size_px[0], cfg.physics.debris.size_px[1], (n_deb,), generator=gen)
            seeds = torch.randint(0, 2**31-1, (n_deb,), generator=gen)
            
            debris_batch = DebrisBatch(
                cx=dx, cy=dy, z=z, size_px=size, delta=delta, angle_deg=angle, is_dash=is_dash, seed=seeds
            )
        else:
             e = torch.empty(0)
             debris_batch = DebrisBatch(e,e,e,e,e,e,e.bool(),e.long())
    else:
         e = torch.empty(0)
         debris_batch = DebrisBatch(e,e,e,e,e,e,e.bool(),e.long())
         
    # Move to device if needed (pipeline handles this usually, but we can do it here)
    if device != 'cpu':
        rod_batch.to(device)
        debris_batch.to(device)
        
    return rod_batch, debris_batch, agglomerates_meta
