
import torch
import random
from ...config import SynthConfig
from .utils import rand_uniform
from ...core.materials import get_material
from ...physics.particles import SHAPE_BUBBLE, SHAPE_DROPLET

def generate_attached_bubbles(cfg: SynthConfig, main_particles: dict, generator: torch.Generator, rng: random.Random):
    """
    Generates bubbles attached to existing particles (Phase 1: Stickiness).
    """
    bs = cfg.physics.bubble_specs
    # Check enablement
    if not bs.enable or bs.attach_prob <= 0 or not main_particles["cx"]:
        return {k: [] for k in main_particles.keys()}
        
    results = {k: [] for k in main_particles.keys()}
    # Ensure keys exist
    for k in ["ref_index", "birefringence", "opacity", "tex_type", "surf_rough", "grain_size", "inclusions", "turbidity", "reflectivity", "dispersion", "absorption_color", "anisotropy", "anisotropy_angle"]:
        if k not in results: results[k] = []
        
    # Material Props for Bubble (Air)
    mat = get_material(bs.material)
    med_ri = cfg.optics.medium_refractive_index
    dn = mat.refractive_index - med_ri
    
    tex_id = 0 # smooth
    
    num_batches = len(main_particles["cx"])
    
    for i in range(num_batches):
        n_curr = len(main_particles["cx"][i])
        if n_curr == 0: continue
        
        # Skip if not real
        if not main_particles["req_label"][i][0]: continue
        
        # Determine attachment
        is_attached = torch.rand(n_curr, generator=generator) < bs.attach_prob
        n_attach = is_attached.sum().item()
        
        if n_attach > 0:
            parent_indices = torch.nonzero(is_attached).squeeze(1)
            
            def get_p(key):
                return main_particles[key][i][parent_indices]
            
            p_cx = get_p("cx")
            p_cy = get_p("cy")
            p_z = get_p("z")
            p_L = get_p("L")
            p_W = get_p("W")
            p_ang = get_p("alpha")
            
            # Generate Bubble Props
            D = rand_uniform(n_attach, bs.diameter_range[0], bs.diameter_range[1], generator)
            
            # Position relative to parent
            # Attach to side (along width) or end? Usually side.
            # Random angle around the particle center?
            # Or specifically "touching" the edge?
            # Let's place them at distance W/2 + D/2 from center line, at random U along length.
            
            # Random U: -L/2 to L/2
            u_pos = (torch.rand(n_attach, generator=generator) - 0.5) * p_L
            
            # Side: +1 or -1
            side = torch.sign(torch.randn(n_attach, generator=generator))
            
            # Local V: W/2 + D/2 * overlap_factor
            # overlap < 1 means sinking in
            overlap = rand_uniform(n_attach, 0.6, 0.9, generator)
            v_pos = side * (p_W * 0.5 + D * 0.5 * overlap)
            
            # Rotate to global
            th = torch.deg2rad(p_ang)
            ct, st = torch.cos(th), torch.sin(th)
            
            # Rotated offset
            # u is along L (x'), v is along W (y')
            # x = x' cos - y' sin
            # y = x' sin + y' cos
            dx = u_pos * ct - v_pos * st
            dy = u_pos * st + v_pos * ct
            
            b_cx = p_cx + dx
            b_cy = p_cy + dy
            b_z = p_z + rand_uniform(n_attach, -0.05, 0.05, generator) # Slight Z offset
            
            # Add to results
            results["cx"].append(b_cx); results["cy"].append(b_cy); results["z"].append(b_z)
            results["L"].append(D); results["W"].append(D); results["H"].append(D)
            results["alpha"].append(torch.zeros(n_attach)); results["beta"].append(torch.zeros(n_attach)); results["gamma"].append(torch.zeros(n_attach))
            results["delta"].append(torch.full((n_attach,), dn))
            
            results["seed"].append(torch.randint(0, 2**31-1, (n_attach,), generator=generator))
            results["group_id"].append(torch.zeros(n_attach, dtype=torch.long))
            results["req_label"].append(torch.zeros(n_attach, dtype=torch.bool)) # Bubbles are usually artifacts (False)
            results["shape_id"].append(torch.full((n_attach,), SHAPE_BUBBLE, dtype=torch.long))
            
            results["curv"].append(torch.zeros(n_attach))
            results["w_jit"].append(torch.zeros(n_attach))
            results["off_jit"].append(torch.zeros(n_attach))
            results["edge_jit"].append(torch.zeros(n_attach))
            results["pol_p"].append(torch.zeros(n_attach))
            results["rag_p"].append(torch.zeros(n_attach))
            results["rag_corr"].append(torch.zeros(n_attach))
            results["shape_mode"].append(torch.zeros(n_attach, dtype=torch.long))
            results["corner_round"].append(torch.zeros(n_attach))
            results["corner_bend"].append(torch.zeros(n_attach))
            
            # Material
            results["ref_index"].append(torch.full((n_attach,), mat.refractive_index))
            results["birefringence"].append(torch.full((n_attach,), mat.birefringence))
            results["opacity"].append(torch.full((n_attach,), mat.opacity))
            results["tex_type"].append(torch.full((n_attach,), tex_id, dtype=torch.long))
            results["surf_rough"].append(torch.full((n_attach,), mat.roughness))
            results["grain_size"].append(torch.full((n_attach,), mat.grain_size))
            results["inclusions"].append(torch.full((n_attach,), mat.internal_inclusions))
            results["turbidity"].append(torch.full((n_attach,), mat.turbidity))
            
            # Phase 4.4.2.3.1
            results["anisotropy"].append(torch.zeros(n_attach))
            results["anisotropy_angle"].append(torch.zeros(n_attach))
            
            # Phase 4.3
            results["reflectivity"].append(torch.full((n_attach,), mat.reflectivity))
            results["dispersion"].append(torch.full((n_attach,), mat.dispersion))
            # RGB Color
            base_color = torch.tensor(mat.absorption_color, dtype=torch.float32)
            results["absorption_color"].append(base_color.unsqueeze(0).expand(n_attach, -1))
            
    return results

def generate_coalesced_droplets(cfg: SynthConfig, main_particles: dict, generator: torch.Generator, rng: random.Random):
    """
    Generates coalesced (merged) droplets.
    This logic mimics 'necking' by spawning droplets that overlap significantly with others,
    or by spawning 'dumbbell' shaped clusters using the Fused logic but for droplets.
    """
    ds = cfg.physics.droplet_specs
    # Check enablement. Droplets must be enabled.
    # We use a heuristic: if droplet count > some threshold, we force some to be coalesced.
    if not ds.enable or not ds.coalesce_enable:
        return {k: [] for k in main_particles.keys()}

    # Probability of a droplet being part of a doublet
    p_coalesce = ds.coalesce_prob 
    
    # We need to generate NEW droplets here, specifically doublets.
    # The standard generator handles single droplets.
    # Here we add EXTRA droplets that are doublets.
    
    # Let's say we want 1-2 doublets per image if droplets are enabled
    n_doublets = rng.randint(0, 2)
    if n_doublets == 0:
         return {k: [] for k in main_particles.keys()}
         
    results = {k: [] for k in main_particles.keys()}
    for k in ["ref_index", "birefringence", "opacity", "tex_type", "surf_rough", "grain_size", "inclusions", "turbidity", "reflectivity", "dispersion", "absorption_color", "anisotropy", "anisotropy_angle"]:
        if k not in results: results[k] = []

    mat = get_material(ds.material)
    med_ri = cfg.optics.medium_refractive_index
    dn = mat.refractive_index - med_ri
    tex_id = 0

    w, h = cfg.canvas.width, cfg.canvas.height
    
    for _ in range(n_doublets):
        # Center of the doublet
        cx = rand_uniform(1, 0, w, generator)
        cy = rand_uniform(1, 0, h, generator)
        
        # Size of first drop
        D1 = rand_uniform(1, ds.diameter_range[0], ds.diameter_range[1], generator)
        
        # Size of second drop
        D2 = rand_uniform(1, ds.diameter_range[0], ds.diameter_range[1], generator)
        
        # Overlap distance: sum of radii * overlap_factor
        # overlap < 1.0 means they merge
        overlap_factor = rand_uniform(1, 0.5, 0.8, generator) # Strong overlap for necking
        dist = (D1 + D2) * 0.5 * overlap_factor
        
        # Angle
        ang = rand_uniform(1, 0, 360, generator)
        th = torch.deg2rad(ang)
        
        dx = dist * torch.cos(th)
        dy = dist * torch.sin(th)
        
        # Drop 1
        cx1, cy1 = cx - dx/2, cy - dy/2
        # Drop 2
        cx2, cy2 = cx + dx/2, cy + dy/2
        
        # Add both
        for (tx, ty, tD) in [(cx1, cy1, D1), (cx2, cy2, D2)]:
            results["cx"].append(tx)
            results["cy"].append(ty)
            results["z"].append(torch.zeros(1))
            results["L"].append(tD); results["W"].append(tD); results["H"].append(tD)
            results["alpha"].append(torch.zeros(1)); results["beta"].append(torch.zeros(1)); results["gamma"].append(torch.zeros(1))
            results["delta"].append(torch.full((1,), dn))
            
            results["seed"].append(torch.randint(0, 2**31-1, (1,), generator=generator))
            results["group_id"].append(torch.zeros(1, dtype=torch.long))
            results["req_label"].append(torch.zeros(1, dtype=torch.bool))
            results["shape_id"].append(torch.full((1,), SHAPE_DROPLET, dtype=torch.long))
            
            # Zero out unused
            results["curv"].append(torch.zeros(1)); results["w_jit"].append(torch.zeros(1))
            results["off_jit"].append(torch.zeros(1)); results["edge_jit"].append(torch.zeros(1))
            results["pol_p"].append(torch.zeros(1)); results["rag_p"].append(torch.zeros(1))
            results["rag_corr"].append(torch.zeros(1)); results["shape_mode"].append(torch.zeros(1, dtype=torch.long))
            results["corner_round"].append(torch.zeros(1))
            results["corner_bend"].append(torch.zeros(1))
            
            # Material
            results["ref_index"].append(torch.full((1,), mat.refractive_index))
            results["birefringence"].append(torch.full((1,), mat.birefringence))
            results["opacity"].append(torch.full((1,), mat.opacity))
            results["tex_type"].append(torch.full((1,), tex_id, dtype=torch.long))
            results["surf_rough"].append(torch.full((1,), mat.roughness))
            results["grain_size"].append(torch.full((1,), mat.grain_size))
            results["inclusions"].append(torch.full((1,), mat.internal_inclusions))
            results["turbidity"].append(torch.full((1,), mat.turbidity))
            
            # Phase 4.4.2.3.1
            results["anisotropy"].append(torch.zeros(1))
            results["anisotropy_angle"].append(torch.zeros(1))
            
            # Phase 4.3
            results["reflectivity"].append(torch.full((1,), mat.reflectivity))
            results["dispersion"].append(torch.full((1,), mat.dispersion))
            # RGB Color
            base_color = torch.tensor(mat.absorption_color, dtype=torch.float32)
            results["absorption_color"].append(base_color.unsqueeze(0).expand(1, -1))

    return results

def generate_fused_clusters(cfg: SynthConfig, main_particles: dict, generator: torch.Generator, rng: random.Random):
    """
    Generates fused clusters (Agglomerates 2.0).
    Supports structured clustering (Stacking, Branching) based on parent shape.
    """
    
    fs = cfg.physics.fused
    # Check if we have particles and if fusion is enabled
    if not fs.enable or fs.p1 <= 0 or not main_particles["cx"]:
        return {k: [] for k in main_particles.keys()} # Return empty lists
        
    results = {k: [] for k in main_particles.keys()}
    # Ensure new keys exist if main_particles has them (it should)
    for k in ["ref_index", "birefringence", "opacity", "tex_type", "surf_rough", "grain_size", "inclusions", "turbidity", "reflectivity", "dispersion", "absorption_color", "anisotropy", "anisotropy_angle"]:
        if k not in results: results[k] = []
    
    # We iterate over the "batches" in main_particles (lists of tensors)
    num_batches = len(main_particles["cx"])
    
    from ...physics.particles import SHAPE_ROD, SHAPE_PLATE, SHAPE_CUBE
    
    for i in range(num_batches):
        n_curr = len(main_particles["cx"][i])
        if n_curr == 0: continue
        
        # Skip ghosts or invalid particles if needed
        if not main_particles["req_label"][i][0]: continue
        
        # Prob check: p1 is probability of becoming a parent (Nucleation site)
        effective_p1 = fs.p1 * 5.0 
        is_parent = torch.rand(n_curr, generator=generator) < effective_p1
        n_parents = is_parent.sum().item()
        
        if n_parents == 0: continue

        # Extract Initial Parents
        parent_indices = torch.nonzero(is_parent).squeeze(1)
        
        # Helper to extract parent props from batch i
        def extract_parents(indices):
            p_dict = {}
            for key in main_particles.keys():
                if key in ["seed", "group_id", "req_label"]: continue # Skip metadata
                if i < len(main_particles[key]):
                     p_dict[key] = main_particles[key][i][indices]
            return p_dict

        current_parents = extract_parents(parent_indices)
        
        # DLCA / Fractal Loop
        # If DLCA enabled, we iterate multiple times.
        # Limit total particles to avoid crash.
        max_depth = 3 if fs.dlca_enable else 1
        
        for depth in range(max_depth):
            if "cx" not in current_parents or len(current_parents["cx"]) == 0: break
            
            n_p = len(current_parents["cx"])
            
            # Decide number of children per parent
            # For DLCA, usually 1-2 per step to form chains/branches
            # For standard agglomerates, maybe more.
            if fs.dlca_enable:
                 n_arms = torch.randint(1, 3, (n_p,), generator=generator)
            else:
                 n_arms = torch.randint(2, 6, (n_p,), generator=generator)
                 
            repeats = n_arms
            total_children = repeats.sum().item()
            if total_children == 0: break
            
            # Repeat parent props
            ex = {}
            for k, v in current_parents.items():
                ex[k] = v.repeat_interleave(repeats, dim=0)
            
            # --- Agglomerates 2.0 / DLCA Logic ---
            weights = torch.tensor(fs.cluster_weights, device=generator.device if hasattr(generator, 'device') else 'cpu', dtype=torch.float)
            if len(weights) < 6:
                pad_size = 6 - len(weights)
                weights = torch.cat([weights, torch.zeros(pad_size, device=weights.device)])
            if weights.sum() == 0: weights = torch.ones(6)
            
            cluster_mode = torch.multinomial(weights, total_children, replacement=True, generator=generator)
            
            # Sintering Strength: Reduces distance
            sinter_factor = 1.0 - torch.clamp(torch.tensor(fs.sintering_strength), 0.0, 0.8)
            
            # Base Offsets
            # Reduce radius for DLCA to keep it tight?
            r = rand_uniform(total_children, 0.2, 0.8, generator) * ex["W"] * sinter_factor
            ang_offset = rand_uniform(total_children, 0.0, 360.0, generator)
            ang_rad = torch.deg2rad(ang_offset)
            
            dx = r * torch.cos(ang_rad)
            dy = r * torch.sin(ang_rad)
            dz = torch.zeros(total_children)
            d_alpha = torch.randn(total_children, generator=generator) * 10.0
            
            noise_L = torch.randn(total_children, generator=generator) * 0.2
            noise_W = torch.randn(total_children, generator=generator) * 0.2
            
            # Apply Modes (Stack, Chain, etc.) - same as before
            # Mode 1: Stacked
            is_stack = (cluster_mode == 1)
            if is_stack.any():
                dx = torch.where(is_stack, dx * 0.2, dx)
                dy = torch.where(is_stack, dy * 0.2, dy)
                z_sign = torch.sign(torch.randn(total_children, generator=generator))
                dz = torch.where(is_stack, z_sign * ex["H"] * 0.8 * sinter_factor, dz)

            # Mode 2: Chain
            is_chain = (cluster_mode == 2)
            if is_chain.any():
                th = torch.deg2rad(ex["alpha"])
                ct, st = torch.cos(th), torch.sin(th)
                dist = (ex["L"] * 0.5 + rand_uniform(total_children, -5.0, 5.0, generator)) * sinter_factor
                side = torch.sign(torch.randn(total_children, generator=generator))
                dx = torch.where(is_chain, side * dist * ct, dx)
                dy = torch.where(is_chain, side * dist * st, dy)
                d_alpha = torch.where(is_chain, torch.randn(total_children, generator=generator) * 5.0, d_alpha)

            # Mode 3: Cross
            is_cross = (cluster_mode == 3)
            if is_cross.any():
                dx = torch.where(is_cross, torch.zeros_like(dx), dx)
                dy = torch.where(is_cross, torch.zeros_like(dy), dy)
                d_alpha = torch.where(is_cross, torch.tensor(90.0) + torch.randn(total_children, generator=generator) * 5.0, d_alpha)

            # Mode 4: Snowflake
            is_snow = (cluster_mode == 4)
            if is_snow.any():
                hex_ang = (torch.randint(0, 6, (total_children,), generator=generator) * 60.0).float()
                d_alpha = torch.where(is_snow, hex_ang + torch.randn(total_children, generator=generator) * 2.0, d_alpha)
                th = torch.deg2rad(ex["alpha"] + d_alpha)
                dist = ex["L"] * 0.6 * sinter_factor
                dx = torch.where(is_snow, dist * torch.cos(th), dx)
                dy = torch.where(is_snow, dist * torch.sin(th), dy)
                noise_L = torch.where(is_snow, torch.tensor(-0.3), noise_L)

            # Mode 5: Spherulite
            is_sphere_agg = (cluster_mode == 5)
            if is_sphere_agg.any():
                rad_ang = rand_uniform(total_children, 0.0, 360.0, generator)
                rad_rad = torch.deg2rad(rad_ang)
                dist = rand_uniform(total_children, 0.0, 0.2, generator) * ex["L"] * sinter_factor
                dx = torch.where(is_sphere_agg, dist * torch.cos(rad_rad), dx)
                dy = torch.where(is_sphere_agg, dist * torch.sin(rad_rad), dy)
                d_alpha = torch.where(is_sphere_agg, rad_ang - ex["alpha"], d_alpha)
                noise_W = torch.where(is_sphere_agg, torch.tensor(-0.5), noise_W)

            # Calculate Final Properties
            c_cx = ex["cx"] + dx
            c_cy = ex["cy"] + dy
            c_z = ex["z"] + dz
            c_alpha = ex["alpha"] + d_alpha
            
            c_L = torch.maximum(torch.tensor(8.0), ex["L"] * (0.8 + noise_L))
            c_W = torch.maximum(torch.tensor(3.0), ex["W"] * (0.8 + noise_W))
            
            # Prepare Child Dict for next iteration
            child_dict = {
                "cx": c_cx, "cy": c_cy, "z": c_z,
                "L": c_L, "W": c_W, "H": ex["H"],
                "alpha": c_alpha, "beta": ex["beta"], "gamma": ex["gamma"],
                "delta": ex["delta"], "shape_id": ex["shape_id"],
                "pol_p": ex["pol_p"], "rag_p": ex["rag_p"], "rag_corr": ex["rag_corr"], "shape_mode": ex["shape_mode"],
                "corner_round": ex.get("corner_round", torch.zeros(total_children, device=ex["cx"].device)),
                "corner_bend": ex.get("corner_bend", torch.zeros(total_children, device=ex["cx"].device)),
                "ref_index": ex["ref_index"], "birefringence": ex["birefringence"], "opacity": ex["opacity"],
                "tex_type": ex["tex_type"], "surf_rough": ex["surf_rough"], "grain_size": ex["grain_size"], "inclusions": ex["inclusions"], "turbidity": ex["turbidity"],
                "anisotropy": ex["anisotropy"], "anisotropy_angle": ex["anisotropy_angle"],
                # Phase 4.3
                "reflectivity": ex["reflectivity"], "dispersion": ex["dispersion"], "absorption_color": ex["absorption_color"]
            }
            
            # Append to Results
            for k, v in child_dict.items():
                if k in results:
                    results[k].append(v)
            
            # Append missing metadata
            c_seed = torch.randint(0, 2**31-1, (total_children,), generator=generator)
            results["seed"].append(c_seed)
            results["group_id"].append(torch.zeros(total_children, dtype=torch.long))
            results["req_label"].append(torch.ones(total_children, dtype=torch.bool))
            
            # Append dummy jitters
            for k in ["curv", "w_jit", "off_jit", "edge_jit"]:
                results[k].append(torch.zeros(total_children))

            # Set current parents to children for next iteration
            current_parents = child_dict

    return results
