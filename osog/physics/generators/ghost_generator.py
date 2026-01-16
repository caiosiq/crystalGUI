
import torch
import random
from ...config import SynthConfig
from .utils import rand_uniform, gen_3d_params

def generate_ghosts(cfg: SynthConfig, n_total_main: int, w: int, h: int, generator: torch.Generator, rng: random.Random):
    """Generates out-of-focus ghost particles."""
    
    gs = cfg.physics.ghosts
    
    # Return lists to be appended to main buffers
    results = {
        "cx": [], "cy": [], "z": [],
        "L": [], "W": [], "H": [],
        "alpha": [], "beta": [], "gamma": [],
        "delta": [], "seed": [], "group_id": [],
        "req_label": [], "shape_id": [],
        "curv": [], "w_jit": [], "off_jit": [], "edge_jit": [],
        "pol_p": [], "rag_p": [], "rag_corr": [], "shape_mode": [],
        # New Material Fields
        "ref_index": [], "birefringence": [], "opacity": [],
        "tex_type": [], "surf_rough": [], "grain_size": [], "inclusions": []
    }

    if gs.enable and gs.fraction > 0 and n_total_main > 0:
        n_ghosts = max(0, int(round(n_total_main * float(gs.fraction))))
        
        if n_ghosts > 0:
            Lg = rand_uniform(n_ghosts, 12.0, 380.0, generator) # Approx max length
            arg = rand_uniform(n_ghosts, 0.01, 0.6, generator)
            Wg = torch.maximum(torch.tensor(2.0), Lg * arg)
            
            s_id, th_ratio, alpha, beta, gamma = gen_3d_params(n_ghosts, cfg, generator)
            Hg = Wg * th_ratio
            
            cxg = rand_uniform(n_ghosts, 0.05 * w, 0.95 * w, generator)
            cyg = rand_uniform(n_ghosts, 0.05 * h, 0.95 * h, generator)
            delta = rand_uniform(n_ghosts, gs.delta_range[0], gs.delta_range[1], generator)
            z = rand_uniform(n_ghosts, 0.5, 2.0, generator)
            seeds = torch.randint(0, 2**31-1, (n_ghosts,), generator=generator)
            g_ids = torch.full((n_ghosts,), -1, dtype=torch.long)
            
            results["cx"].append(cxg); results["cy"].append(cyg); results["z"].append(z)
            results["L"].append(Lg); results["W"].append(Wg); results["H"].append(Hg)
            results["alpha"].append(alpha); results["beta"].append(beta); results["gamma"].append(gamma)
            results["delta"].append(delta); results["seed"].append(seeds)
            results["group_id"].append(g_ids)
            results["req_label"].append(torch.zeros(n_ghosts, dtype=torch.bool))
            results["shape_id"].append(s_id)
            
            results["w_jit"].append(torch.full((n_ghosts,), gs.width_jit_amp))
            results["off_jit"].append(torch.full((n_ghosts,), gs.offset_jit_amp))
            results["edge_jit"].append(torch.full((n_ghosts,), gs.edge_jit_amp))
            results["curv"].append(rand_uniform(n_ghosts, gs.curve_kappa_range[0], gs.curve_kappa_range[1], generator))
            results["rag_p"].append(torch.full((n_ghosts,), gs.ragged_p))
            results["rag_corr"].append(torch.full((n_ghosts,), gs.ragged_corr))
            results["pol_p"].append(torch.zeros(n_ghosts))
            modes = torch.randint(0, 4, (n_ghosts,), generator=generator)
            results["shape_mode"].append(modes)
            
            # Material props for ghosts (Standard/Amorphous)
            # Use delta from config (legacy behavior for ghosts) but map to RI
            # RI = Medium + Delta
            med_ri = cfg.optics.medium_refractive_index
            ri = med_ri + delta
            results["ref_index"].append(ri)
            results["birefringence"].append(torch.zeros(n_ghosts))
            results["opacity"].append(torch.zeros(n_ghosts))
            results["tex_type"].append(torch.zeros(n_ghosts, dtype=torch.long)) # Smooth
            results["surf_rough"].append(torch.zeros(n_ghosts))
            results["grain_size"].append(torch.ones(n_ghosts))
            results["inclusions"].append(torch.zeros(n_ghosts))

    return results
