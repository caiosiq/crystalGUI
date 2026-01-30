# Implementation Plan: OSOG Phase 4.25 & 4.5

This document outlines the detailed implementation strategy for the "Multi-Modal Sensor Heads" (Phase 4.25) and "High-Fidelity Optical Features" (Phase 4.5) updates.

## Phase 4.25: The "G-Buffer" Architecture (The "Reflectance" Update)

**Goal**: Move from a monolithic "God Class" architecture to a modern "G-Buffer" rendering pipeline, enabling multiple sensor modalities (DIC, Brightfield, PVM) to be simulated from a single geometric pass.

### 1. Architectural Refactoring (The "Monolith" Breakup)

We will separate the **Geometry Pass** (Particle Physics) from the **Optical Pass** (Light Simulation).

*   **Renaming Strategy**:
    *   `DICModulatorTorch` -> `OpticalEngine` (The Manager)
    *   `ParticleShader` -> `GeometryShader` (The Geometry Generator)

*   **Step A: The Geometry Pass (G-Buffer)**
    *   Modify `GeometryShader.render_batch` to STOP producing an image directly.
    *   Instead, it will produce a **G-Buffer Tensor** `(N, 4, H_patch, W_patch)`.
        *   **Channel 0 (Height)**: Physical thickness (microns).
        *   **Channel 1 (Mask)**: Binary presence mask (0 or 1).
        *   **Channel 2 (Refractive Index)**: Material property.
        *   **Channel 3 (Orientation)**: Angle $\theta$ (for polarization/anisotropy).

*   **Step B: The Optical Head (The "Sensor")**
    *   Implement modular "Optical Functions" in `OpticalEngine` that consume the G-Buffer:
        *   `sim_dic(g_buffer, shear_angle)`: Computes Gradient of Ch0 -> Image.
        *   `sim_brightfield(g_buffer, aperture)`: Computes Laplacian of Ch0 (Becke Line) -> Image.
        *   `sim_polarization(g_buffer, polarizer_angle)`: Computes $sin^2(Ch0 \times Ch3)$ -> Image.
        *   `sim_pvm(g_buffer, laser_angle)`: Computes Normals from Ch0 -> Reflectance Image.

### 2. Multi-Head Pipeline
*   **Update `Pipeline.generate`**:
    *   Call `GeometryShader` ONCE to get the G-Buffer for all particles.
    *   Loop through requested modalities (e.g., `['dic', 'pvm']`).
    *   Apply the corresponding Optical Function to the *same* G-Buffer patches.
    *   Stamp the results onto separate Canvases.
    *   Output: `Dict[str, np.ndarray]` (e.g., `{'dic': img1, 'pvm': img2}`).

### 3. The Reflectance Shader (PVM Mode)
*   **New Function**: `optics.sim_pvm(g_buffer, laser_vec)`
*   **Physics**: Surface Normal Scattering (BRDF).
    *   Compute Normals $\vec{N}$ from Height Map (Ch0).
    *   Calculate Diffuse: $\vec{N} \cdot \vec{L}$.
    *   Calculate Specular (Sparkle): High-frequency noise mask $\times$ Specular Highlight.

---

## Phase 4.5: High-Fidelity Optical Features (The "Realism" Update)

**Goal**: Add specific optical artifacts that bridge the gap between simulation and real-world microscopy, utilizing the new G-Buffer architecture.

### 1. Pseudo-Phase Brightfield (The "Becke Line")
*   **New Logic in `sim_brightfield`**:
    *   Input: G-Buffer Height (Ch0).
    *   Operation: `laplacian = filters.laplacian(height)`.
    *   Compositing: `intensity = background - absorption + k * laplacian`.
    *   Result: A realistic bright halo at the edges of transparent particles.

### 2. Geometric Crystal Habits (The "Adipic" Shader)
*   **Update `GeometryShader`**:
    *   Implement `stepped_terrace_mask`.
    *   Apply recursive rectangular masking to the Height Map (Ch0) during generation.
    *   This ensures all optical modes (DIC, PVM) see the same realistic "stepped" crystal geometry.

### 3. Deep-Z Bokeh (The "Slurry" Shader)
*   **Update Composition**:
    *   Implement a Depth-Dependent Blur *after* the Optical Pass but *before* Stamping.
    *   Use the particle's `z` coordinate to select a blur kernel size.
    *   Apply Disk Kernel convolution for "Bokeh" effect on out-of-focus particles.

### 4. Contact Shadows (Ambient Occlusion)
*   **Update Composition**:
    *   Compute Screen-Space Ambient Occlusion (SSAO) on the full-frame Height Map (stamped Ch0).
    *   Darken pixels where height gradients suggest deep crevices between touching particles.

## Execution Order

1.  **Refactor & Rename** (Phase 4.25 Foundation)
    *   Split `ParticleShader` into `GeometryShader` (G-Buffer producer).
    *   Rename `DICModulatorTorch` to `OpticalEngine`.
2.  **Implement Optical Functions** (Phase 4.25 Core)
    *   Port existing DIC/Brightfield logic to consume G-Buffer.
    *   Implement new PVM (Reflectance) logic.
3.  **Pipeline Integration** (Phase 4.25 Delivery)
    *   Update `Pipeline` to handle multi-modal output.
4.  **Realism Shaders** (Phase 4.5)
    *   Implement Becke Line, Terracing, and Bokeh using the new modular structure.
