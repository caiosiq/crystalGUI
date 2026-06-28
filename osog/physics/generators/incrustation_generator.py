
import torch
import random
from ...config import SynthConfig
from .utils import rand_uniform
from ...core.materials import get_material
from ...physics.particles import SHAPE_POLYHEDRA, SHAPE_CUBE, SHAPE_ROD, SHAPE_SPHERE, SHAPE_BUBBLE, SHAPE_DROPLET, SHAPE_PLATE

def generate_incrustations(cfg: SynthConfig, main_particles: dict, generator: torch.Generator, rng: random.Random):
    """
    Generates tiny surface incrustations (micro-crystals) attached to main particles.
    Phase 4.4.2.3.1: Surface Incrustation
    """
    iss = cfg.physics.incrustation_specs
    
    # Check enablement
    if not iss.enable or iss.fraction <= 0 or not main_particles["cx"]:
        return {k: [] for k in main_particles.keys()}
        
    results = {k: [] for k in main_particles.keys()}
    # Ensure keys exist
    for k in ["ref_index", "birefringence", "opacity", "tex_type", "surf_rough", "grain_size", "inclusions", "turbidity", "reflectivity", "dispersion", "absorption_color", "anisotropy", "anisotropy_angle"]:
        if k not in results: results[k] = []
        
    # Material Props for Incrustations
    mat = get_material(iss.material)
    med_ri = cfg.optics.medium_refractive_index
    dn = mat.refractive_index - med_ri
    
    # Incrustations are usually rough/polycrystalline
    tex_id = 1 # granular/pitted
    
    num_batches = len(main_particles["cx"])
    
    for i in range(num_batches):
        n_curr = len(main_particles["cx"][i])
        if n_curr == 0: continue
        
        # Skip ghosts or invalid particles
        if not main_particles["req_label"][i][0]: continue
        
        # Determine which particles get incrusted
        is_incrusted = torch.rand(n_curr, generator=generator) < iss.fraction
        n_hosts = is_incrusted.sum().item()
        
        if n_hosts > 0:
            parent_indices = torch.nonzero(is_incrusted).squeeze(1)
            
            def get_p(key):
                return main_particles[key][i][parent_indices]
            
            p_cx = get_p("cx")
            p_cy = get_p("cy")
            p_z = get_p("z")
            p_L = get_p("L")
            p_W = get_p("W")
            p_H = get_p("H")
            p_ang = get_p("alpha")
            p_shape = get_p("shape_id")
            
            # For each host, generate N tiny crystals
            # We vectorize this by repeating host props
            counts = torch.randint(iss.count_range[0], iss.count_range[1], (n_hosts,), generator=generator)
            total_inc = counts.sum().item()
            
            # Repeat host props
            r_cx = p_cx.repeat_interleave(counts)
            r_cy = p_cy.repeat_interleave(counts)
            r_z = p_z.repeat_interleave(counts)
            r_L = p_L.repeat_interleave(counts)
            r_W = p_W.repeat_interleave(counts)
            r_H = p_H.repeat_interleave(counts)
            r_ang = p_ang.repeat_interleave(counts)
            r_shape = p_shape.repeat_interleave(counts)
            
            # Generate Incrustation Props
            # Size is tiny
            size = rand_uniform(total_inc, iss.size_range[0], iss.size_range[1], generator)
            
            # Position Generation based on Shape
            local_x = torch.zeros(total_inc)
            local_y = torch.zeros(total_inc)
            local_z = torch.zeros(total_inc)
            
            # --- 1. SPHERES ---
            is_sphere = (r_shape == SHAPE_SPHERE) | (r_shape == SHAPE_BUBBLE) | (r_shape == SHAPE_DROPLET)
            
            if is_sphere.any():
                # Uniform sphere sampling
                # R = L/2 (Diameter)
                R = r_L * 0.5
                
                # Sample z from -R to R
                z_sph = (torch.rand(total_inc, generator=generator) * 2.0 - 1.0) * R
                # Sample phi from 0 to 2pi
                phi_sph = torch.rand(total_inc, generator=generator) * 2.0 * 3.14159265
                
                # Compute radius at z
                # Clamp to avoid sqrt(negative) due to float errors
                r_xy = torch.sqrt(torch.clamp(R**2 - z_sph**2, min=0.0))
                
                x_sph = r_xy * torch.cos(phi_sph)
                y_sph = r_xy * torch.sin(phi_sph)
                
                local_x = torch.where(is_sphere, x_sph, local_x)
                local_y = torch.where(is_sphere, y_sph, local_y)
                local_z = torch.where(is_sphere, z_sph, local_z)
                
            # --- 2. RODS (Cylinders) ---
            is_rod = (r_shape == SHAPE_ROD)
            if is_rod.any():
                # Cylinder surface sampling (Side only for now)
                # Length L along X-axis
                # Radius R = W/2
                
                x_rod = (torch.rand(total_inc, generator=generator) - 0.5) * r_L
                theta_rod = torch.rand(total_inc, generator=generator) * 2.0 * 3.14159265
                R_rod = r_W * 0.5
                
                # Cylinder aligned along X
                y_rod = R_rod * torch.cos(theta_rod)
                z_rod = R_rod * torch.sin(theta_rod)
                
                local_x = torch.where(is_rod, x_rod, local_x)
                local_y = torch.where(is_rod, y_rod, local_y)
                local_z = torch.where(is_rod, z_rod, local_z)

            # --- 3. BOXES (Cube, Plate, Polyhedra) ---
            # Default fallback
            is_box = ~(is_sphere | is_rod)
            
            if is_box.any():
                # Box sampling
                face_id = torch.randint(0, 6, (total_inc,), generator=generator)
                
                u = (torch.rand(total_inc, generator=generator) - 0.5)
                v = (torch.rand(total_inc, generator=generator) - 0.5)
                
                # Face 0: +X (Right) -> x=L/2, y=u*W, z=v*H
                # Face 1: -X (Left)  -> x=-L/2, y=u*W, z=v*H
                # Face 2: +Y (Back)  -> x=u*L, y=W/2, z=v*H
                # Face 3: -Y (Front) -> x=u*L, y=-W/2, z=v*H
                # Face 4: +Z (Top)   -> x=u*L, y=v*W, z=H/2
                # Face 5: -Z (Bot)   -> x=u*L, y=v*W, z=-H/2
                
                # X-Faces
                is_x = (face_id == 0) | (face_id == 1)
                lx_box = torch.where(face_id == 0, r_L * 0.5, -r_L * 0.5)
                ly_box = u * r_W
                lz_box = v * r_H
                
                # Y-Faces
                is_y = (face_id == 2) | (face_id == 3)
                lx_y = u * r_L
                ly_y = torch.where(face_id == 2, r_W * 0.5, -r_W * 0.5)
                lz_y = v * r_H
                
                lx_box = torch.where(is_y, lx_y, lx_box)
                ly_box = torch.where(is_y, ly_y, ly_box)
                lz_box = torch.where(is_y, lz_y, lz_box)
                
                # Z-Faces
                is_z = (face_id == 4) | (face_id == 5)
                lx_z = u * r_L
                ly_z = v * r_W
                lz_z = torch.where(face_id == 4, r_H * 0.5, -r_H * 0.5)
                
                lx_box = torch.where(is_z, lx_z, lx_box)
                ly_box = torch.where(is_z, ly_z, ly_box)
                lz_box = torch.where(is_z, lz_z, lz_box)
                
                local_x = torch.where(is_box, lx_box, local_x)
                local_y = torch.where(is_box, ly_box, local_y)
                local_z = torch.where(is_box, lz_box, local_z)
            
            # Add random jitter to "sink" them slightly or stick out
            z_jitter = rand_uniform(total_inc, -0.5, 0.5, generator) * size
            local_z += z_jitter
            
            # Rotate to global
            th = torch.deg2rad(r_ang)
            ct, st = torch.cos(th), torch.sin(th)
            
            dx = local_x * ct - local_y * st
            dy = local_x * st + local_y * ct
            
            b_cx = r_cx + dx
            b_cy = r_cy + dy
            b_z = r_z + local_z
            
            # Add to results
            results["cx"].append(b_cx); results["cy"].append(b_cy); results["z"].append(b_z)
            results["L"].append(size); results["W"].append(size); results["H"].append(size)
            
            # Random orientation for the tiny crystals
            results["alpha"].append(rand_uniform(total_inc, 0, 360, generator))
            results["beta"].append(rand_uniform(total_inc, 0, 360, generator))
            results["gamma"].append(rand_uniform(total_inc, 0, 360, generator))
            
            results["delta"].append(torch.full((total_inc,), dn))
            
            results["seed"].append(torch.randint(0, 2**31-1, (total_inc,), generator=generator))
            results["group_id"].append(torch.zeros(total_inc, dtype=torch.long))
            results["req_label"].append(torch.zeros(total_inc, dtype=torch.bool)) # Incrustations are artifacts
            
            # Use small cubes or polyhedra
            results["shape_id"].append(torch.full((total_inc,), SHAPE_CUBE, dtype=torch.long))
            
            results["curv"].append(torch.zeros(total_inc))
            results["w_jit"].append(torch.zeros(total_inc))
            results["off_jit"].append(torch.zeros(total_inc))
            results["edge_jit"].append(torch.zeros(total_inc))
            results["pol_p"].append(torch.zeros(total_inc))
            results["rag_p"].append(torch.zeros(total_inc))
            results["rag_corr"].append(torch.zeros(total_inc))
            results["shape_mode"].append(torch.zeros(total_inc, dtype=torch.long))
            results["corner_round"].append(torch.zeros(total_inc))
            results["corner_bend"].append(torch.zeros(total_inc))
            
            # Material
            results["ref_index"].append(torch.full((total_inc,), mat.refractive_index))
            results["birefringence"].append(torch.full((total_inc,), mat.birefringence))
            results["opacity"].append(torch.full((total_inc,), mat.opacity))
            results["tex_type"].append(torch.full((total_inc,), tex_id, dtype=torch.long))
            results["surf_rough"].append(torch.full((total_inc,), mat.roughness))
            results["grain_size"].append(torch.full((total_inc,), mat.grain_size))
            results["inclusions"].append(torch.full((total_inc,), mat.internal_inclusions))
            results["turbidity"].append(torch.full((total_inc,), mat.turbidity))
            
            # Phase 4.4.2.3.1
            results["anisotropy"].append(torch.zeros(total_inc)) # Incrustations are usually isotropic/granular
            results["anisotropy_angle"].append(torch.zeros(total_inc))
            
            # Phase 4.3
            results["reflectivity"].append(torch.full((total_inc,), mat.reflectivity))
            results["dispersion"].append(torch.full((total_inc,), mat.dispersion))
            # RGB Color
            base_color = torch.tensor(mat.absorption_color, dtype=torch.float32)
            results["absorption_color"].append(base_color.unsqueeze(0).expand(total_inc, -1))
            
    return results
