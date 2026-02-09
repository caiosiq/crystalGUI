# Implementation Plan: Phase 4.4.1.7 - Procedural Polyhedra

## Goal
Implement "Procedural Polyhedra" (Euhedral Crystals) as a new particle type. These particles are defined by the intersection of multiple flat planes (Half-Space Intersection), allowing for complex, realistic mineral shapes (e.g., Garnet, Quartz) unlike the perfect geometric primitives (Cube, Sphere).

## 1. Architecture & Data Structures

### 1.1. Constants (`osog/physics/constants.py`)
*   Add a new shape ID:
    ```python
    SHAPE_POLYHEDRA = 6
    ```

### 1.2. Configuration (`osog/config.py`)
*   Create a new dataclass `PolyhedraSpecs`:
    *   `enable`: bool
    *   `count_range`: Tuple[int, int]
    *   `size_range`: Tuple[float, float] (Overall scale)
    *   `num_planes_range`: Tuple[int, int] (Complexity of the shape, e.g., 6-12 planes)
    *   `irregularity`: float (0.0 = symmetric, 1.0 = highly random plane distances)
    *   `material`: str
    *   `texture_type`, `roughness`, etc. (Standard material props)
*   Update `PhysicsConfig` to include `polyhedra_specs`.
*   Update `SynthConfig` parsing logic (`from_dict`, `from_flat_dict`) to handle the new specs.

### 1.3. Particle Batch (`osog/physics/particles.py`)
*   No new fields required. We will leverage the existing `seed` field.
*   The `GeometryShader` will use the `seed` to deterministically generate the unique set of planes for each polyhedra particle.

## 2. Generator Logic (`osog/physics/generators/main_generator.py`)

*   Add a new section for `PolyhedraSpecs` in `generate_main_particles`.
*   Logic:
    *   Sample count `N`.
    *   Sample position (`cx`, `cy`, `z`) and size (`L`, `W`, `H`).
    *   **Rotation**: Polyhedra are 3D objects, so `beta` (tilt) and `gamma` (roll) are critical. We will enable full 3D rotation by default or respect the `enable_3d` flag.
    *   **Shape ID**: Set to `SHAPE_POLYHEDRA`.
    *   **Seed**: Ensure high-entropy seeds are generated, as they dictate the shape geometry.

## 3. Shader Implementation (`osog/optics/shaders/geometry.py`)

This is the core of the implementation.

### 3.1. `_render_polyhedra`
*   **Input**: Batch of particles, UV coordinates.
*   **Logic (Per Particle)**:
    1.  **Coordinate Transformation**: Transform UVs into the particle's local 3D frame (Rotated & Scaled).
    2.  **Plane Generation**:
        *   Use `seed` to generate `M` random plane normals $(A, B, C)$ and distances $D$.
        *   Normals should be distributed on a sphere (or biased towards specific crystal habits if we get fancy later).
        *   $D$ determines the distance from center.
    3.  **Intersection (The "Sculpting")**:
        *   For each pixel $(u, v)$:
            *   Initialize `z_ceil = +infinity`, `z_floor = -infinity`.
            *   Iterate through all `M` planes:
                *   Calculate $z_{plane} = -(Ax + By + D) / C$.
                *   If $C > 0$ (Up-facing): `z_ceil = min(z_ceil, z_{plane})`
                *   If $C < 0$ (Down-facing): `z_floor = max(z_floor, z_{plane})`
            *   `Thickness = max(0, z_ceil - z_floor)`
            *   `Height = (z_ceil + z_floor) / 2.0` (Relative to center)
    4.  **Optimization**:
        *   Since we can't loop easily in vectorized PyTorch code without performance hits, we might define a fixed maximum number of planes (e.g., 12) and vectorize the plane calculation.
        *   Alternatively, pre-compute the planes in the generator and pass them (though this changes `ParticleBatch`).
        *   **Decision**: We will implement a "Random Polygon" approach or a fixed set of random planes (e.g., 8-12 random planes) generated on-the-fly using vectorized hashing of the seed.

## 4. UI Updates

### 4.1. HTML (`playground.html`)
*   Add a new "Polyhedra" tab or section in the particle controls.
*   Inputs: Count, Size, Material, Texture, maybe "Complexity" (Num Planes).

### 4.2. JS (`playground.js`)
*   Collect the new inputs.
*   Update the JSON payload sent to `/render`.

## 5. Execution Steps

1.  **Define Constants & Configs**: Update `constants.py` and `config.py`.
2.  **Update Generator**: Modify `main_generator.py` to produce Polyhedra particles.
3.  **Implement Shader Logic**: Add the math to `geometry.py`.
4.  **Update UI**: Add controls to Playground.
5.  **Verify**: Use a debug script to visualize a single Polyhedron and ensure it rotates and intersects correctly.
