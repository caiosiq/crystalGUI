
import torch
import random
from ...config import SynthConfig
from ..constants import SHAPE_ROD, SHAPE_PLATE, SHAPE_CUBE, SHAPE_SPHERE, SHAPE_BUBBLE, SHAPE_DROPLET, SHAPE_MODE_MAP
from .utils import rand_uniform, gen_3d_params
from ...core.materials import get_material

TEXTURE_MAP = {
    "smooth": 0,
    "striated": 1,
    "pitted": 2,
    "granular": 3
}

def generate_main_particles(cfg: SynthConfig, w: int, h: int, generator: torch.Generator, rng: random.Random):
    """Generates main particles based on specific specs."""
    
    # Storage
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
        "tex_type": [], "surf_rough": [], "grain_size": [], "inclusions": [],
        # Phase 4.3: Technicolor
        "reflectivity": [], "dispersion": [], "absorption_color": []
    }
    
    med_ri = cfg.optics.medium_refractive_index

    def add_material_props(n, mat_name, res_dict, gen, override_inclusions=0.0):
        mat = get_material(mat_name)
        
        # Calculate Delta (RI - Medium)
        dn = mat.refractive_index - med_ri
        # Add some variation? Real materials have slight RI variation?
        # For now, constant.
        res_dict["delta"].append(torch.full((n,), dn))
        
        res_dict["ref_index"].append(torch.full((n,), mat.refractive_index))
        res_dict["birefringence"].append(torch.full((n,), mat.birefringence))
        res_dict["opacity"].append(torch.full((n,), mat.opacity))
        
        # Phase 4.3 Fields
        res_dict["reflectivity"].append(torch.full((n,), mat.reflectivity))
        res_dict["dispersion"].append(torch.full((n,), mat.dispersion))
        # RGB Color (N, 3)
        # Use unsqueeze/expand for safety
        base_color = torch.tensor(mat.absorption_color, dtype=torch.float32)
        col_tensor = base_color.unsqueeze(0).expand(n, -1) # (N, 3)
        res_dict["absorption_color"].append(col_tensor)
        
        tex_id = TEXTURE_MAP.get(mat.texture_type, 0)
        res_dict["tex_type"].append(torch.full((n,), tex_id, dtype=torch.long))
        
        res_dict["surf_rough"].append(torch.full((n,), mat.roughness))
        res_dict["grain_size"].append(torch.full((n,), mat.grain_size))
        
        # Inclusions: Use Max(Material, Override)
        inc = max(mat.internal_inclusions, override_inclusions)
        res_dict["inclusions"].append(torch.full((n,), inc))

    def get_aligned_alpha(n, gen):
        if cfg.physics.flow_enable:
            kappa = cfg.physics.flow_shear_rate * 20.0
            if kappa > 0.1:
                sigma = 180.0 / (1.0 + kappa)
                noise = torch.randn(n, generator=gen) * sigma
                return torch.full((n,), cfg.physics.flow_direction) + noise
            else:
                return rand_uniform(n, -90.0, 90.0, gen)
        else:
            return rand_uniform(n, -90.0, 90.0, gen)

    def get_sedimented_z(n, L, W, H, gen):
        if cfg.physics.sedimentation_enable:
             vol = L * W * H
             vol_n = vol / (vol.max() + 1e-6)
             strength = cfg.physics.sedimentation_strength
             
             if cfg.physics.size_segregation_enable:
                 # Brazil Nut Effect: Large particles rise to TOP (Z=1.0)
                 # Small particles sink or stay mixed.
                 # We bias large particles to 1.0.
                 range_width = 2.0 * (1.0 - strength * vol_n * 0.9)
                 # Start from 1.0 and go down
                 z_base = 1.0 - rand_uniform(n, 0.0, 1.0, gen) * range_width
                 return torch.clamp(z_base, -1.0, 1.0)
             else:
                 # Standard Sedimentation: Large particles sink to BOTTOM (Z=-1.0)
                 range_width = 2.0 * (1.0 - strength * vol_n * 0.9)
                 z_base = -1.0 + rand_uniform(n, 0.0, 1.0, gen) * range_width
                 return torch.clamp(z_base, -1.0, 1.0)
        else:
             return rand_uniform(n, -0.1, 0.1, gen)

    # 1. Rods
    rs = cfg.physics.rod_specs
    if rs.enable:
        n = rng.randint(rs.count_range[0], rs.count_range[1])
        if n > 0:
            L = rand_uniform(n, rs.length_range[0], rs.length_range[1], generator)
            asp = rand_uniform(n, rs.aspect_range[0], rs.aspect_range[1], generator)
            W = torch.maximum(torch.tensor(2.0), L * asp)
            H = W # Rods are cylindrical
            
            cx = rand_uniform(n, 0.05 * w, 0.95 * w, generator)
            cy = rand_uniform(n, 0.05 * h, 0.95 * h, generator)
            z = get_sedimented_z(n, L, W, H, generator)
            
            # Apply Material
            add_material_props(n, rs.material, results, generator)
            
            # Phase 4: Flow Alignment
            alpha = get_aligned_alpha(n, generator)

            beta = rand_uniform(n, -90.0, 90.0, generator) if cfg.physics.rods.enable_3d else torch.zeros(n)
            gamma = rand_uniform(n, -180.0, 180.0, generator) if cfg.physics.rods.enable_3d else torch.zeros(n)
            
            results["cx"].append(cx); results["cy"].append(cy); results["z"].append(z)
            results["L"].append(L); results["W"].append(W); results["H"].append(H)
            results["alpha"].append(alpha); results["beta"].append(beta); results["gamma"].append(gamma)
            # Delta handled by add_material_props
            results["seed"].append(torch.randint(0, 2**31-1, (n,), generator=generator))
            results["group_id"].append(torch.arange(n)) # Simple ID
            results["req_label"].append(torch.ones(n, dtype=torch.bool))
            results["shape_id"].append(torch.full((n,), SHAPE_ROD, dtype=torch.long))
            
            results["curv"].append(torch.zeros(n))
            results["w_jit"].append(torch.zeros(n))
            results["off_jit"].append(torch.zeros(n))
            results["edge_jit"].append(torch.zeros(n))
            results["pol_p"].append(torch.full((n,), rs.polarity_p))
            results["rag_p"].append(torch.full((n,), rs.ragged_p))
            results["rag_corr"].append(torch.full((n,), rs.ragged_corr))
            results["shape_mode"].append(torch.full((n,), SHAPE_MODE_MAP.get(rs.shape_mode, 0), dtype=torch.long))

    # 2. Spheres
    ss = cfg.physics.sphere_specs
    if ss.enable:
        n = rng.randint(ss.count_range[0], ss.count_range[1])
        if n > 0:
            D = rand_uniform(n, ss.diameter_range[0], ss.diameter_range[1], generator)
            L = D; W = D; H = D
            
            cx = rand_uniform(n, 0.05 * w, 0.95 * w, generator)
            cy = rand_uniform(n, 0.05 * h, 0.95 * h, generator)
            z = get_sedimented_z(n, L, W, H, generator)
            
            add_material_props(n, ss.material, results, generator, override_inclusions=0.0)
            
            alpha = rand_uniform(n, -90.0, 90.0, generator)
            beta = torch.zeros(n); gamma = torch.zeros(n)
            
            results["cx"].append(cx); results["cy"].append(cy); results["z"].append(z)
            results["L"].append(L); results["W"].append(W); results["H"].append(H)
            results["alpha"].append(alpha); results["beta"].append(beta); results["gamma"].append(gamma)
            # Delta handled by add_material_props
            results["seed"].append(torch.randint(0, 2**31-1, (n,), generator=generator))
            results["group_id"].append(torch.arange(n))
            results["req_label"].append(torch.ones(n, dtype=torch.bool))
            results["shape_id"].append(torch.full((n,), SHAPE_SPHERE, dtype=torch.long))
            
            results["curv"].append(torch.zeros(n))
            results["w_jit"].append(torch.zeros(n))
            results["off_jit"].append(torch.zeros(n))
            results["edge_jit"].append(torch.zeros(n))
            results["pol_p"].append(torch.zeros(n))
            results["rag_p"].append(torch.zeros(n))
            results["rag_corr"].append(torch.zeros(n))
            results["shape_mode"].append(torch.full((n,), 0, dtype=torch.long))

    # 3. Cubes
    cs = cfg.physics.cube_specs
    if cs.enable:
        n = rng.randint(cs.count_range[0], cs.count_range[1])
        if n > 0:
            S = rand_uniform(n, cs.size_range[0], cs.size_range[1], generator)
            L = S; W = S; H = S
            
            cx = rand_uniform(n, 0.05 * w, 0.95 * w, generator)
            cy = rand_uniform(n, 0.05 * h, 0.95 * h, generator)
            z = rand_uniform(n, -0.1, 0.1, generator)
            
            add_material_props(n, cs.material, results, generator)
            
            alpha = get_aligned_alpha(n, generator)
            # Phase 4.4: Full 3D Rotation for Cubes (Config Controlled)
            if cfg.physics.rods.enable_3d:
                beta = rand_uniform(n, -180.0, 180.0, generator) 
                gamma = rand_uniform(n, -180.0, 180.0, generator)
            else:
                beta = torch.zeros(n)
                gamma = torch.zeros(n)
            
            results["cx"].append(cx); results["cy"].append(cy); results["z"].append(z)
            results["L"].append(L); results["W"].append(W); results["H"].append(H)
            results["alpha"].append(alpha); results["beta"].append(beta); results["gamma"].append(gamma)
            # Delta handled by add_material_props
            results["seed"].append(torch.randint(0, 2**31-1, (n,), generator=generator))
            results["group_id"].append(torch.arange(n))
            results["req_label"].append(torch.ones(n, dtype=torch.bool))
            results["shape_id"].append(torch.full((n,), SHAPE_CUBE, dtype=torch.long))
            
            results["curv"].append(torch.zeros(n))
            results["w_jit"].append(torch.zeros(n))
            results["off_jit"].append(torch.zeros(n))
            results["edge_jit"].append(torch.zeros(n))
            results["pol_p"].append(torch.zeros(n))
            results["rag_p"].append(torch.zeros(n))
            results["rag_corr"].append(torch.zeros(n))
            results["shape_mode"].append(torch.full((n,), 0, dtype=torch.long))

    # 4. Plates
    ps = cfg.physics.plate_specs
    if ps.enable:
        n = rng.randint(ps.count_range[0], ps.count_range[1])
        if n > 0:
            L = rand_uniform(n, ps.size_range[0], ps.size_range[1], generator)
            asp = rand_uniform(n, ps.aspect_range[0], ps.aspect_range[1], generator)
            W = torch.maximum(torch.tensor(2.0), L * asp)
            thick = rand_uniform(n, ps.thickness_range[0], ps.thickness_range[1], generator)
            H = W * thick
            
            cx = rand_uniform(n, 0.05 * w, 0.95 * w, generator)
            cy = rand_uniform(n, 0.05 * h, 0.95 * h, generator)
            z = rand_uniform(n, -0.1, 0.1, generator)
            
            add_material_props(n, ps.material, results, generator)
            
            alpha = rand_uniform(n, -90.0, 90.0, generator)
            # Phase 4.4: Full 3D Rotation for Plates (Config Controlled)
            if cfg.physics.rods.enable_3d:
                beta = rand_uniform(n, -180.0, 180.0, generator)
                gamma = rand_uniform(n, -180.0, 180.0, generator)
            else:
                beta = torch.zeros(n)
                gamma = torch.zeros(n)
            
            results["cx"].append(cx); results["cy"].append(cy); results["z"].append(z)
            results["L"].append(L); results["W"].append(W); results["H"].append(H)
            results["alpha"].append(alpha); results["beta"].append(beta); results["gamma"].append(gamma)
            # Delta handled by add_material_props
            results["seed"].append(torch.randint(0, 2**31-1, (n,), generator=generator))
            results["group_id"].append(torch.arange(n))
            results["req_label"].append(torch.ones(n, dtype=torch.bool))
            results["shape_id"].append(torch.full((n,), SHAPE_PLATE, dtype=torch.long))
            
            results["curv"].append(torch.zeros(n))
            results["w_jit"].append(torch.zeros(n))
            results["off_jit"].append(torch.zeros(n))
            results["edge_jit"].append(torch.zeros(n))
            results["pol_p"].append((torch.rand(n, generator=generator) < ps.polarity_p).float())
            results["rag_p"].append(torch.full((n,), ps.ragged_p))
            results["rag_corr"].append(torch.full((n,), ps.ragged_corr))
            results["shape_mode"].append(torch.full((n,), SHAPE_MODE_MAP.get(ps.shape_mode, 0), dtype=torch.long))

    # 5. Bubbles
    bs = cfg.physics.bubble_specs
    if bs.enable:
        n = rng.randint(bs.count_range[0], bs.count_range[1])
        if n > 0:
            D = rand_uniform(n, bs.diameter_range[0], bs.diameter_range[1], generator)
            L = D; W = D; H = D
            
            cx = rand_uniform(n, 0.05 * w, 0.95 * w, generator)
            cy = rand_uniform(n, 0.05 * h, 0.95 * h, generator)
            z = rand_uniform(n, -0.1, 0.1, generator)
            
            add_material_props(n, bs.material, results, generator)
            
            alpha = rand_uniform(n, -90.0, 90.0, generator)
            beta = torch.zeros(n); gamma = torch.zeros(n)
            
            results["cx"].append(cx); results["cy"].append(cy); results["z"].append(z)
            results["L"].append(L); results["W"].append(W); results["H"].append(H)
            results["alpha"].append(alpha); results["beta"].append(beta); results["gamma"].append(gamma)
            # Delta handled by add_material_props
            results["seed"].append(torch.randint(0, 2**31-1, (n,), generator=generator))
            results["group_id"].append(torch.arange(n))
            results["req_label"].append(torch.ones(n, dtype=torch.bool))
            results["shape_id"].append(torch.full((n,), SHAPE_BUBBLE, dtype=torch.long))
            
            results["curv"].append(torch.zeros(n))
            results["w_jit"].append(torch.zeros(n))
            results["off_jit"].append(torch.zeros(n))
            results["edge_jit"].append(torch.zeros(n))
            results["pol_p"].append(torch.zeros(n))
            results["rag_p"].append(torch.zeros(n))
            results["rag_corr"].append(torch.zeros(n))
            results["shape_mode"].append(torch.full((n,), 0, dtype=torch.long))

    # 6. Droplets
    ds = cfg.physics.droplet_specs
    if ds.enable:
        n = rng.randint(ds.count_range[0], ds.count_range[1])
        if n > 0:
            D = rand_uniform(n, ds.diameter_range[0], ds.diameter_range[1], generator)
            L = D; W = D; H = D
            
            cx = rand_uniform(n, 0.05 * w, 0.95 * w, generator)
            cy = rand_uniform(n, 0.05 * h, 0.95 * h, generator)
            z = rand_uniform(n, -0.1, 0.1, generator)
            
            add_material_props(n, ds.material, results, generator)
            
            alpha = rand_uniform(n, -90.0, 90.0, generator)
            beta = torch.zeros(n); gamma = torch.zeros(n)
            
            results["cx"].append(cx); results["cy"].append(cy); results["z"].append(z)
            results["L"].append(L); results["W"].append(W); results["H"].append(H)
            results["alpha"].append(alpha); results["beta"].append(beta); results["gamma"].append(gamma)
            # Delta handled by add_material_props
            results["seed"].append(torch.randint(0, 2**31-1, (n,), generator=generator))
            results["group_id"].append(torch.arange(n))
            results["req_label"].append(torch.ones(n, dtype=torch.bool))
            results["shape_id"].append(torch.full((n,), SHAPE_DROPLET, dtype=torch.long))
            
            results["curv"].append(torch.zeros(n))
            results["w_jit"].append(torch.zeros(n))
            results["off_jit"].append(torch.zeros(n))
            results["edge_jit"].append(torch.zeros(n))
            results["pol_p"].append(torch.zeros(n))
            results["rag_p"].append(torch.zeros(n))
            results["rag_corr"].append(torch.zeros(n))
            results["shape_mode"].append(torch.full((n,), 0, dtype=torch.long))
            
    return results
