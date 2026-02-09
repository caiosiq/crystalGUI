# OSOG Paper Goals: From Simulator to Publication

This document outlines the roadmap specifically designed to reach the milestones required for the upcoming OSOG paper.

## Phase 1: The Geometry Engine (The "Form")
*Goal: Stop generating "blobs" and start generating "minerals."*

- [x] **Polyhedron Generator (Half-Space)**
    *   **Description**: We must implement this first. It allows us to generate the "shattered cube" look of KCl and the "gemstone" look of other minerals.
- [x] **Aggregates/Agglomeration**
    *   **Context**: KCl and many industrial crystals clump together. We need a specific logic to spawn particles that intersect or "grow" off each other.
    *   **Task**: `ParticleBatch` needs a `clump_id` to fuse geometries before rendering.

## Phase 2: The Optical Core (The "Light")
*Goal: Match the specific look of Blaze/FBRM/PVM probes.*

- [ ] **Fresnel Edge Darkening**
    *   **Context**: In brightfield, transparent crystals look like dark outlines because light hitting the edge refracts away from the camera.
    *   **Task**: Modify `shader.py` to correctly map transmission $T \approx 1 - Fresnel(\theta)$.
- [ ] **Internal Scattering (Cloudiness)**
    *   **Context**: Real industrial crystals aren't perfect glass; they are milky.
    *   **Task**: Add a `turbidity` parameter to the ray-marcher (Volumetric Fog inside the crystal).
- [ ] **Blaze/Directional Lighting**
    *   **Context**: Blaze probes often use specific illumination angles to highlight edges.
    *   **Task**: Instead of a fixed "Headlamp" light, allow the user to define a `LightSource(direction, intensity, spread)`. This allows mimicking different probe brands.

## Phase 3: The "Crystallization-Ready" Features
*Goal: Ensure the engine can simulate the process, not just the particle.*

- [ ] **Polydispersity Handling**
    *   **Context**: Real reactors have 1 huge crystal and 1,000 tiny fines.
    *   **Task**: The batch generator must support "Power Law" size distributions (generating many tiny dots and few large rocks in one frame).
- [ ] **Depth of Field (DoF)**
    *   **Context**: In a dense suspension, particles slightly out of focus look like vague shadows. This is crucial for segmentation models (to teach them what to ignore).
    *   **Task**: Implement a Z-dependent blur layer in the compositor.
- [ ] **Motion Blur**
    *   **Context**: In-situ probes often capture fast-moving particles.
    *   **Task**: A simple directional blur vector applied to the final particle patch.

## Phase 4: The Paper Experiments (The Proof)
*Goal: Prove OSOG is ready for publication.*

- [ ] **The "Probe Match" (Qualitative)**
    *   **KCl Challenge**: Take a real image of KCl. Tune OSOG to generate "Shattered Cubes" with "Rounded Edges" (simulation of attrition). Show them side-by-side.
    *   **Blaze Challenge**: Mimic the high-contrast lighting of the Blaze probe.
- [ ] **Inverse Rendering (The Quantitative Proof)**
    *   **Experiment**: Take a single real image of a KCl crystal.
    *   **Method**: Use Gradient Descent to optimize the OSOG parameters (Shape, Refractive Index, Roughness) until the render matches the photo.
    *   **Conclusion**: Demonstrate that "OSOG is differentiable, allowing us to 'fit' physics parameters to visual data."
