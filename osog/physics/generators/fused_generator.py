
import torch
import random
from ...config import SynthConfig
from .utils import rand_uniform

def generate_fused_clusters(cfg: SynthConfig, main_particles: dict, generator: torch.Generator, rng: random.Random):
    """
    Generates fused clusters by attaching children to existing 'parent' particles.
    """
    
    fs = cfg.physics.fused
    # Check if we have particles and if fusion is enabled
    if not fs.enable or fs.p1 <= 0 or not main_particles["cx"]:
        return {k: [] for k in main_particles.keys()} # Return empty lists
        
    results = {k: [] for k in main_particles.keys()}
    # Ensure new keys exist if main_particles has them (it should)
    for k in ["ref_index", "birefringence", "opacity", "tex_type", "surf_rough", "grain_size", "inclusions"]:
        if k not in results: results[k] = []
    
    # We iterate over the "batches" in main_particles (lists of tensors)
    # Typically main_particles["cx"] is a list of tensors [batch1_cx, batch2_cx, ...]
    
    num_batches = len(main_particles["cx"])
    
    for i in range(num_batches):
        n_curr = len(main_particles["cx"][i])
        if n_curr == 0: continue
        
        # Skip ghosts or invalid particles if needed
        # Assuming main_particles["req_label"][i] tells us if it's a real particle
        if not main_particles["req_label"][i][0]: continue
        
        # Prob check: p1 is probability of becoming a parent
        is_parent = torch.rand(n_curr, generator=generator) < fs.p1
        n_parents = is_parent.sum().item()
        
        if n_parents > 0:
            parent_indices = torch.nonzero(is_parent).squeeze(1)
            
            # Helper to get parent prop
            def get_p(key):
                return main_particles[key][i][parent_indices]
            
            p_cx = get_p("cx")
            p_cy = get_p("cy")
            p_z  = get_p("z")
            p_L  = get_p("L")
            p_W  = get_p("W")
            p_H  = get_p("H")
            p_alpha = get_p("alpha")
            p_beta  = get_p("beta")
            p_gamma = get_p("gamma")
            p_delta = get_p("delta")
            p_sid   = get_p("shape_id")
            
            p_pol   = get_p("pol_p")
            p_rag   = get_p("rag_p")
            p_rc    = get_p("rag_corr")
            p_mode  = get_p("shape_mode")
            
            # Material Props
            p_ri    = get_p("ref_index")
            p_bi    = get_p("birefringence")
            p_op    = get_p("opacity")
            p_tex   = get_p("tex_type")
            p_surf  = get_p("surf_rough")
            p_grain = get_p("grain_size")
            p_inc   = get_p("inclusions")
            
            # Generate Children
            n_arms = torch.randint(2, 6, (n_parents,), generator=generator)
            repeats = n_arms
            
            # Repeat parent props
            ex_cx = p_cx.repeat_interleave(repeats)
            ex_cy = p_cy.repeat_interleave(repeats)
            ex_z = p_z.repeat_interleave(repeats)
            ex_L = p_L.repeat_interleave(repeats)
            ex_W = p_W.repeat_interleave(repeats)
            ex_H = p_H.repeat_interleave(repeats)
            ex_alpha = p_alpha.repeat_interleave(repeats)
            ex_beta = p_beta.repeat_interleave(repeats)
            ex_gamma = p_gamma.repeat_interleave(repeats)
            ex_delta = p_delta.repeat_interleave(repeats)
            ex_sid = p_sid.repeat_interleave(repeats)
            
            ex_pol = p_pol.repeat_interleave(repeats)
            ex_rag = p_rag.repeat_interleave(repeats)
            ex_rc = p_rc.repeat_interleave(repeats)
            ex_mode = p_mode.repeat_interleave(repeats)
            
            # Material Repeats
            ex_ri = p_ri.repeat_interleave(repeats)
            ex_bi = p_bi.repeat_interleave(repeats)
            ex_op = p_op.repeat_interleave(repeats)
            ex_tex = p_tex.repeat_interleave(repeats)
            ex_surf = p_surf.repeat_interleave(repeats)
            ex_grain = p_grain.repeat_interleave(repeats)
            ex_inc = p_inc.repeat_interleave(repeats)
            
            total_children = len(ex_cx)
            
            # Perturb children
            spread = rand_uniform(n_parents, 10.0, 55.0, generator).repeat_interleave(repeats)
            d_alpha = torch.randn(total_children, generator=generator) * (spread * 0.25)
            c_alpha = ex_alpha + d_alpha
            
            # Size variation
            noise_L = torch.randn(total_children, generator=generator) * 0.25
            c_L = torch.maximum(torch.tensor(8.0), ex_L * (1.0 + noise_L))
            
            noise_W = torch.randn(total_children, generator=generator) * 0.25
            c_W = torch.maximum(torch.tensor(3.0), ex_W * (1.0 + noise_W))
            
            # Position offset
            r = rand_uniform(total_children, 0.2, 0.8, generator) * ex_W
            ang_offset = rand_uniform(total_children, 0.0, 360.0, generator)
            ang_rad = torch.deg2rad(ang_offset)
            
            c_cx = ex_cx + r * torch.cos(ang_rad)
            c_cy = ex_cy + r * torch.sin(ang_rad)
            
            c_seed = torch.randint(0, 2**31-1, (total_children,), generator=generator)
            
            # Add to results
            results["cx"].append(c_cx); results["cy"].append(c_cy); results["z"].append(ex_z)
            results["L"].append(c_L); results["W"].append(c_W); results["H"].append(ex_H)
            results["alpha"].append(c_alpha); results["beta"].append(ex_beta); results["gamma"].append(ex_gamma)
            results["delta"].append(ex_delta); results["seed"].append(c_seed)
            results["group_id"].append(torch.zeros(total_children, dtype=torch.long)) # Dummy
            results["req_label"].append(torch.ones(total_children, dtype=torch.bool))
            results["shape_id"].append(ex_sid)
            
            results["curv"].append(torch.zeros(total_children))
            results["w_jit"].append(torch.zeros(total_children))
            results["off_jit"].append(torch.zeros(total_children))
            results["edge_jit"].append(torch.zeros(total_children))
            results["pol_p"].append(ex_pol)
            results["rag_p"].append(ex_rag)
            results["rag_corr"].append(ex_rc)
            results["shape_mode"].append(ex_mode)
            
            # Material
            results["ref_index"].append(ex_ri)
            results["birefringence"].append(ex_bi)
            results["opacity"].append(ex_op)
            results["tex_type"].append(ex_tex)
            results["surf_rough"].append(ex_surf)
            results["grain_size"].append(ex_grain)
            results["inclusions"].append(ex_inc)
            
    return results
