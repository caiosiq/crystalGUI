# Procedural Polyhedra Implementation in OSOG

This document details the implementation of **Procedural Polyhedra (Euhedral Crystals)** in the OSOG engine. This feature moves beyond simple geometric primitives (rods, spheres, cubes) to generate complex, faceted mineral shapes like Garnet or Quartz by "sculpting" them from empty space using random cutting planes.

## 1. The Concept: Half-Space Intersection

Instead of rasterizing a mesh, we define a convex polyhedron as the **intersection of multiple half-spaces**.
*   Start with an infinite volume (or a large bounding box).
*   Generate $N$ random planes defined by $Ax + By + Cz + D = 0$.
*   Each plane divides space into "Inside" and "Outside".
*   The crystal exists only where a point is "Inside" **all** planes simultaneously.
*   We calculate the thickness at every pixel by finding the interval $[Z_{enter}, Z_{exit}]$ along the view ray that remains inside all planes.

---

## 2. Configuration (`osog/config.py`)

We added a specific configuration class `PolyhedraSpecs` to control the generation parameters.

```python
@dataclass
class PolyhedraSpecs:
    """Specific configuration for Procedural Polyhedra (Euhedral Crystals)"""
    enable: bool = False
    count_range: Tuple[int, int] = (5, 20)
    size_range: Tuple[float, float] = (30.0, 100.0) # Overall scale
    num_planes_range: Tuple[int, int] = (6, 12) # Complexity (Number of cutting planes)
    irregularity: float = 0.2 # 0.0 = Symmetric, 1.0 = Highly random distances
    material: str = "standard"
    
    # Phase 4.4.1.5 Texture Pass
    texture_type: str = "none"
    surf_roughness: float = 0.0
    internal_inclusions: float = 0.0
    polarity_flip_p: float = 0.0
```

---

## 3. Generator Logic (`osog/physics/generators/main_generator.py`)

In the main generator, we instantiate `SHAPE_POLYHEDRA` particles. Crucially, we repurpose existing fields in `ParticleBatch` to store the procedural parameters (Number of Planes and Irregularity) to avoid breaking the GPU schema.

*   **Number of Planes** is stored in `curvature`.
*   **Irregularity** is stored in `ragged_p`.
*   **Seed** is crucial as it drives the deterministic random generation of planes in the shader.

```python
    # 7. Polyhedra (Euhedral Crystals)
    pys = cfg.physics.polyhedra_specs
    if pys.enable:
        # ... (Random Position/Size Logic) ...
        
        # Polyhedra always use full 3D rotation
        alpha = get_aligned_alpha(n, generator)
        beta = rand_uniform(n, -180.0, 180.0, generator)
        gamma = rand_uniform(n, -180.0, 180.0, generator)
        
        # ... (Standard Fields) ...

        results["shape_id"].append(torch.full((n,), SHAPE_POLYHEDRA, dtype=torch.long))
        
        # Pack Polyhedra specific params into unused fields:
        
        num_planes = torch.randint(pys.num_planes_range[0], pys.num_planes_range[1] + 1, (n,), generator=generator).float()
        results["curv"].append(num_planes) # Hack: Store num_planes in curvature
        
        results["rag_p"].append(torch.full((n,), pys.irregularity)) # Hack: Store irregularity in rag_p
```

---

## 4. The Geometry Shader (`osog/optics/shaders/geometry.py`)

This is the core of the implementation. The `render_batch` function now includes a "Procedural Plane Sculpting" block.

### 4.1. Setup
We initialize the "Floor" ($Z_{min}$) and "Ceiling" ($Z_{max}$) of the valid volume.

```python
            if is_poly.any():
                # Procedural Plane Sculpting (Phase 4.4.1.7)
                
                # ... (Setup) ...
                
                # Initialize ceilings and floors
                big_val = 1e5
                z_ceil = torch.full_like(X_rot, big_val)
                z_floor = torch.full_like(X_rot, -big_val)
```

### 4.2. Plane Generation Loop
We iterate `MAX_PLANES` times. For each iteration, we generate a pseudo-random plane based on the particle's `seed`.

```python
                MAX_PLANES = 12
                
                # ...
                
                for i in range(MAX_PLANES):
                    # Pseudo-random generation
                    # distinct seed for each plane iteration
                    p_seed = batch.seed.view(N, 1, 1) + i * 1337
                    
                    # ... (Spherical Coordinate Hashing for Normal Vector nx, ny, nz) ...
                    
                    # Random Distance D from center
                    # Dist varies from 0.3*Size to 0.5*Size based on irregularity
                    dist_val = (min_dim * 0.4) + (min_dim * 0.4) * h3 * (0.5 + 0.5 * irregularity)
```

### 4.3. Ray-Plane Intersection
We solve for the intersection distance $t$ along the view ray.

```python
                    # Plane Equation: nx*X + ny*Y + nz*Z + D = 0
                    # Ray: P = O + t*D_ray
                    # t = - (N dot O + dist) / (N dot D_ray)
                    
                    denom = nx * dx_loc + ny * dy_local + nz * dz_local
                    numer = -(nx * ox_loc + ny * oy_loc + nz * oz_loc + dist_val)
                    
                    t_plane = numer / denom_safe
```

### 4.4. Half-Space Logic (Sculpting)
This is the critical logic that was recently fixed. We determine if the intersection is an "Entry" (Floor) or "Exit" (Ceiling) based on the angle between the ray and the plane normal.

*   If the ray travels **against** the normal (`denom < 0`), it is entering the volume (Floor).
*   If the ray travels **with** the normal (`denom > 0`), it is exiting the volume (Ceiling).

```python
                    is_exit = (denom < 0)
                    is_enter = (denom > 0)
                    
                    # Update bounds
                    # z_ceil takes the MINIMUM of all Exits (must be below all ceilings)
                    z_ceil = torch.where(mask_plane & is_exit, torch.minimum(z_ceil, t_plane), z_ceil)
                    
                    # z_floor takes the MAXIMUM of all Enters (must be above all floors)
                    z_floor = torch.where(mask_plane & is_enter, torch.maximum(z_floor, t_plane), z_floor)
```

### 4.5. Final Thickness
The final thickness is simply the distance between the highest floor and lowest ceiling. If `floor > ceil`, the ray missed the object (thickness is 0).

```python
                # Final thickness
                thickness_poly = torch.clamp(z_ceil - z_floor, min=0.0)
```
