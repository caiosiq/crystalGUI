# OSOG Roadmap: From Simulator to Digital Twin

This roadmap outlines the path to transforming OSOG (Optical Synthetic Object Generator) from a high-speed data generator into a scientifically rigorous "Digital Twin" for optical microscopy.

## Current Status (v2.0 - "The 2.5D Vectorized Engine")
*Last Updated: 2026-01-14*

**Recently Completed:**
*   **Fully Vectorized GPU Pipeline**: Migrated from object-based CPU rendering to a PyTorch-native tensor pipeline.
    *   Achieved massive speedup by eliminating Python loops ("Kernel Launch Overhead") via `index_put_` batch stamping.
    *   Zero-Copy architecture: Data stays on VRAM from generation to final sensor simulation.
*   **2.5D Physics Architecture**:
    *   Unified `ParticleBatch` supporting **Rods, Plates, Cubes, and Spheres**.
    *   Added 3D state: `z` (depth), `beta` (tumble), `gamma` (roll), `height` (thickness).
    *   Implemented **Analytic Thickness Projection**: Efficiently renders 3D habits by calculating their 2D projected footprints and optical path lengths without mesh rasterization.
*   **Multi-Modal Optics Foundation**:
    *   **DIC**: Upgraded to use gradient of 3D height maps.
    *   **Polarization**: Added "Maltese Cross" simulation logic ($I \propto \sin^2(2\theta)\sin^2(\delta/2)$).
    *   **Shadowgraphy**: Added silhouette mode for back-lit probes.
*   **Sensor Head Optimization**:
    *   Migrated Blur, Noise, and Background generation to GPU (`SensorHeadTorch`).
    *   Optimized large-scale illumination variations using upsampling instead of heavy convolutions.
*   **Architecture Refactoring**:
    *   **Modular Shader System**: Refactored monolithic `DICModulatorTorch` into specialized shaders (`ParticleShader`, `DebrisShader`).
    *   **Utils Separation**: Extracted shared utilities to `osog/optics/utils.py`.

---

## Phase 1: The "Enemies of the Probe" (Negative Classes)
*Goal: Generate the specific artifacts that cause False Positives in real-world detection systems.*

- [x] **Advanced Air Bubbles (The "Donut" Artifact)**
    - [x] **Implementation**: Added `SHAPE_BUBBLE` (ID 4) and specialized "Donut" shader in `dic_torch.py`.
    - [x] **Brightfield**: Simulates dark outer ring (refraction limit) and bright central spot (lensing).
    - [x] **DIC**: Uses steep gradient modulation to create 3D bubble appearance.
    - [ ] **Interaction**: Bubbles should "stick" to crystals (flotation).
- [x] **Immiscible Droplets ("Oiling Out")**
    - [x] **Implementation**: Added `SHAPE_DROPLET` (ID 5).
    - [x] **Transparency**: Droplets rendered with subtle rim effects to simulate transparency with refractive edges.
    - [ ] **Coalescence**: Logic to merge touching droplets into a peanut shape (Necking).
- [x] **Fouling & Smudges**
    - [x] **Lens Fouling**: Implemented "Lens Dirt" simulation in `SensorHeadTorch` (static, out-of-focus blobs).
    - [ ] **Biofilm/Residue**: Low-frequency, semi-transparent textures overlaying the background.

## Phase 2: 2.5D Crystal Habits (Positive Classes)
*Goal: Move beyond "Rods" to the full spectrum of crystal morphology.*

- [x] **Basic 3D Support (Completed)**: `ParticleBatch` now supports Shape IDs (Rod/Plate/Cube/Sphere) and 3D rotation angles.
- [x] **Refined Plate Simulation**
    - [x] **Edge-On vs Face-On**: Accurate intensity modulation based on optical path length (tumble-dependent path).
    - [x] **Tumbling**: Smooth transition from "Ghost" (face) to "Needle" (edge) via `tumble_strength` height modulation.
- [x] **Cubic & Blocky Crystals (NaCl-like)**
    - [x] **Internal Edges**: Implemented fake "X" pattern for transparent cubes using pyramidal height modulation based on tumble angle.
    - [ ] **Corner Glint**: Fresnel reflection peaks at sharp corners.
- [ ] **Agglomerates 2.0**
    - [ ] **3D Clustering**: Instead of 2D overlap, simulate 3D stacking where depth of field blurs parts of the cluster.
    - [ ] **Grain Boundaries**: Dark lines where crystals intersect/fuse.
- [ ] **Dendrites & Spherulites**
    - [ ] **Procedural Generation**: Use fractal growth (DLA) or branching L-systems to generate snowflake-like structures.

## Phase 3: Multi-Modal Optics
*Goal: Simulate expensive hardware using software.*

- [x] **Polarized Light Microscopy (PLM) - Foundation**
    - [x] **Maltese Cross**: Basic logic implemented in `DICModulatorTorch`.
    - [ ] **Polychromatic Polarization**: Simulate "Michel-Levy Chart" colors (interference colors) for thick crystals.
- [ ] **Fluorescence & Confocal**
    - [ ] **Glow Shader**: Additive blending mode (light on dark).
    - [ ] **Halo/Bloom**: Gaussian spread of light beyond object boundaries.
    - [ ] **Bleaching**: Simulation of intensity decay over time (if video support is added).
- [ ] **Shadowgraphy (Backlight)**
    - [x] **Basic Silhouette**: Implemented.
    - [ ] **Bokeh Engine**: Accurate depth-of-field blur (Circle of Confusion) for out-of-focus particles.
    - [ ] **Diffraction Fringes**: Airy disks at the edges of small particles.

## Phase 4: Physics & Dynamics
*Goal: Crystals should behave like physical objects, not just static images.*

- [ ] **Flow Alignment**
    - [ ] Simulate shear flow: Long rods should align with the flow direction.
    - [ ] Tumbling in shear: Jefferey orbits for particles in fluid.
- [ ] **Sedimentation**
    - [ ] Large particles sink (Z-depth increases/decreases).
    - [ ] Size segregation: Big rocks at the bottom, fines floating.
- [ ] **Defects & Inclusions**
    - [ ] **Solvent Inclusions**: Liquid pockets inside a crystal (visible as a bubble inside a square).
    - [ ] **Cracks/Fractures**: Sharp lines breaking the geometry.

## Phase 5: Engineering & Validation
*Goal: Ensure the tool is fast, usable, and trusted.*

- [ ] **Real-World Validation Metric**
    - [ ] **FID (Fréchet Inception Distance)**: Compare statistics of OSOG images vs. Real Microscope images.
    - [ ] **Turing Test**: A blind test app for chemists ("Real or Synthetic?").
- [x] **The "OSOG Lab" GUI**
    - [x] A dedicated "Playground" window (Synthesis Tab controls).
    - [x] Sliders for optical parameter (Polarization Angle).
    - [ ] Live preview of 3D rotations (in progress).
- [ ] **Neural Style Refiner (GAN/Diffusion)**
    - [ ] Train a lightweight Pix2Pix or CycleGAN to "texture" the output of OSOG.
    - [ ] Use OSOG for geometry/labels (perfect truth) and GAN for texture (perfect realism).

## Creative / Experimental Ideas

*   **"Virtual Dyes"**: Simulate the effect of adding Methylene Blue (absorption) or Rhodamine (fluorescence) to specific polymorphs to test if computer vision can detect them.
*   **"The Dirty Probe" Scenario**: A difficulty slider that progressively degrades image quality (fouling, scratches, bad lighting) to stress-test AI models.
*   **Auto-Calibration**: Input a real image, and use an optimizer (Gradient Descent) to tune OSOG parameters (`config.yaml`) until the synthetic image matches the real one (Inverse Rendering).
