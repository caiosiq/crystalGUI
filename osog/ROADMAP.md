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
    - [x] **Interaction**: Bubbles should "stick" to crystals (flotation).
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
    - [x] **Corner Glint**: Fresnel reflection peaks at sharp corners.
- [x] **Advanced Material Shader**
    - [x] **Opacity/Translucency**: Support for semi-transparent or opaque particles (e.g., slurries).
    - [x] **Internal Inclusions**: Noise texture *inside* the particle body (cloudiness).
    - [x] **Metallic Materials**: High reflectivity (glints) and opacity for steel shavings.
    - [x] **Birefringence**: High birefringence support for colorful crystals.
- [x] **Agglomerates 2.0**
    - [x] **Structured Clusters**: Support for Stacking, Chaining (End-to-End), and Cross (90°) formations.
    - [x] **Z-Stacking**: Particles can now sit on top of each other.
    - [x] **Procedural Generation**: Added "Dendrite/Snowflake" mode using hexagonal branching.
    - [x] **Spherulites**: Added "Spherulite" mode (Radial Starburst).
    - [x] **3D Clustering**: Enhanced Z-offsets for stacking to maximize Depth of Field blur.
    - [x] **Grain Boundaries**: Shader now darkens crevices (positive curvature regions) to simulate contact lines.

## Phase 3: Advanced Optics Laboratory (Completed)
*Goal: Simulate expensive hardware using software.*

- [x] **Polarized Light Microscopy (PLM)**
    - [x] **Maltese Cross**: Basic logic implemented in `DICModulatorTorch`.
    - [x] **Polychromatic Polarization**: Simulate "Michel-Levy Chart" colors (interference colors) with improved spectral mapping and high-order retardation.
- [x] **Fluorescence & Confocal**
    - [x] **Glow Shader**: Additive blending mode.
    - [x] **Confocal Sectioning**: Depth-dependent weighting.
- [x] **Shadowgraphy (Backlight)**
    - [x] **Basic Silhouette**: Implemented.
    - [x] **Bokeh Engine**: Accurate depth-of-field blur (Airy Disks and defocus-dependent diffraction) implemented in `shadowgraphy` mode.
- [x] **Diffraction Fringes**: Airy disks at the edges of small particles (Shader-based).
- [x] **Chromatic Aberration**: Lateral color fringing (Sensor-based).

## Phase 4: Physics & Dynamics (In Progress)
*Goal: Move from "Static Picture" to "Dynamic Environment".*

- [x] **Flow Alignment**
    - [x] **Shear Flow**: Rods align with flow direction (Von Mises distribution).
    - [x] **Tumbling**: Randomized orientation for non-aligned particles.
- [x] **Sedimentation**
    - [x] **Z-Depth Bias**: Large particles sink to bottom (Z=-1), small particles float.
    - [x] **Size Segregation**: "Brazil Nut Effect" (Large particles rise to top when enabled).
- [x] **Defects & Inclusions**
    - [x] **Solvent Inclusions**: Liquid pockets trapped inside crystals (Noise-based).
    - [x] **Cracks/Fractures**: Stress fractures and cleavage planes (Lightning pattern shader).
- [x] **Aggregation Physics**
    - [x] **DLCA**: Diffusion-Limited Cluster Aggregation (Recursive fractal generation).
    - [x] **Sintering**: Neck formation simulated by overlap strength.
## Phase 4.25: Multi-Modal Sensor Heads (The "Reflectance" Update)
*Goal: Expand OSOG from a "Microscope Simulator" to a "Process Analytical Technology (PAT) Simulator" by supporting in-situ laser backscatter probes (e.g., PVM, FBRM, Blaze).*

- [ ] **Multi-Head Architecture**
    * **Concept**: Decouple "Physical Reality" (The Particle Batch) from "Observation" (The Sensor).
    * **Capability**: Render the *exact same* particle distribution twice simultaneously: once via Transmission (Wave Optics) and once via Reflectance (Laser Scattering).
    * **Value**: Enables generation of perfect **"Paired Datasets"** (Sensor Fusion) which are impossible to capture physically.

- [ ] **Reflectance Shader (PVM Mode)**
    * **Physics**: Replace Optical Path Difference (OPD) with **Surface Normal Scattering** (Bidirectional Reflectance Distribution Function - BRDF).
    * **Implementation**:
        * **Flash (Lambertian)**: Intensity $\propto \vec{N} \cdot \vec{L}$ (Dot product of surface normal and laser angle).
        * **Sparkle (Specular)**: High-frequency noise multiplication to simulate rough crystalline facets catching the light.
        * **Bloom**: Apply heavy Gaussian blur to saturated pixels to simulate sensor blooming (over-exposure).

- [ ] **Surface Micro-Texture**
    * **Problem**: Smooth geometric primitives look like plastic in reflectance mode.
    * **Solution**: Apply a "Micro-Roughness" normal map to all particles before rendering. This ensures even flat crystal faces have realistic granular scattering textures.
## Phase 4.5: High-Fidelity Optical Features (The "Realism" Update)
*Goal: Bridge the "Uncanny Valley" by simulating specific diffraction artifacts seen in real flow-cells.*
- [ ] Pseudo-Phase Brightfield (The "Halo" Shader)
    Problem: Current transparent objects look like flat, dark silhouettes. Real needles show a "Becke Line" (bright halo) due to refraction.Solution: Implement a Laplacian-of-Gaussian (LoG) edge term in the Brightfield shader.Math: $I_{out} = I_{bg} - k_{abs} \cdot \alpha + k_{diff} \cdot \nabla^2 \phi$ (where $\nabla^2 \phi$ adds the glow at the boundaries).
- [ ] Geometric Crystal Habits (The "Adipic" Shader)
    Problem: Roughness (Perlin noise) looks organic/wavy. Real crystals have sharp, stepped growth terraces (internal straight lines).
    Solution: Create a "Terrace Texture" generator.Implementation: Use a distance_transform or recursive rectangular masking on the particle surface to create "step-down" height maps, simulating 3D prism layers.
- [ ] Deep-Z Bokeh (The "Slurry" Shader)
    Problem: Current "Ghost" blur is a simple Gaussian. Real out-of-focus highlights form bright disks (Airy patterns) that preserve intensity.
    Solution: Implement a Disk Kernel Blur for the Ghosts module. Bright spots in the background should expand into translucent circles (Bokeh), not just fade away.
- [ ] Contact Shadows (Ambient Occlusion)Problem: In dense clusters, particles blend together.
    Solution: Implement Screen-Space Ambient Occlusion (SSAO) logic in the composition step. Darken the pixels where two z-buffer values are close together to simulate light trapped in the crevice between touching crystals.
- [ ] Motion Blur Vectorization
    Problem: Flow images often have directional smear if the particle is moving fast.
    Solution: Add a velocity vector to the ParticleBatch. Apply a Directional Blur kernel oriented along the flow vector $(u, v)$ for each particle individually before stamping.
- [ ] **Flexible Filament Engine (The "Insulin" Shader)**
    * **Primitive**: Switch from rigid `Cylinders` to `Cubic Bezier Tubes` to model flexible amyloid fibrils.
    * **Physics**: "Spaghetti Dynamics"—filaments can bend, twist, and form "bird's nest" clusters without passing through each other.
    * **Optics**: Sub-pixel intensity preservation to ensure fibrils <1px wide remain visible (conservative rasterization).

- [ ] **Polyhedral Geometry (The "Glycine" Shader)**
    * **Mesh Generation**: Procedural generation of habit based on **Miller Indices** (e.g., `{101}` planes for bipyramids) rather than simple scale scaling.
    * **Internal Faceting**: Calculate optical path lengths through *angled* faces to replicate the characteristic "X" refraction lines and internal steps seen in bipyramidal crystals.

- [ ] **Amorphous & Liquid Shader (The "Indomethacin" Shader)**
    * **Metaballs**: Implement "Smooth Union" ($min(d1, d2) - k$) SDF blending to simulate "Oiling Out" (liquid droplets merging).
    * **Surface Texture**: Apply "Stucco" or "Brain-like" normal maps to simulate the non-crystalline, rough surface of amorphous agglomerates.

- [ ] **Pseudo-Phase Brightfield (The "Becke Line" Update)**
    * **Problem**: Transparent objects currently look like dark silhouettes.
    * **Solution**: Implement a **Laplacian-based Edge Shader**.
    * **Effect**: Adds the characteristic bright white halo *outside* or *inside* the particle edge (depending on refractive index difference), critical for detecting transparent needles in brightfield.

- [ ] **Multi-Scale Noise (The "Dirty Water" Effect)**
    * **Micro-Fines**: Add a layer of "sub-visible" background particulate matter (1-2px speckles) that drift independently.
    * **Fluid Inhomogeneity**: Simulate "Schlieren" lines caused by mixing fluids of different densities/temperatures.

- [ ] **Geometric Crystal Habits (The "Adipic" Shader)**
    * **Terracing**: Replace random Perlin roughness with a "Stepped" texture generator to simulate growth terraces (concentric geometric lines) on large plate faces.
## Phase 5: Engineering & Validation
*Goal: Ensure the tool is fast, usable, and trusted.*

- [ ] **Real-World Validation Metric**
    - [ ] **FID (Fréchet Inception Distance)**: Compare statistics of OSOG images vs. Real Microscope images.
    - [ ] **Turing Test**: A blind test app for chemists ("Real or Synthetic?").
- [x] **The "OSOG Lab" GUI**
    - [x] A dedicated "Playground" window (Synthesis Tab controls).
    - [x] Sliders for optical parameter (Polarization Angle).
    - [x] Live preview of 3D rotations.
- [ ] **Neural Style Refiner (GAN/Diffusion)**
    - [ ] Train a lightweight Pix2Pix or CycleGAN to "texture" the output of OSOG.
    - [ ] Use OSOG for geometry/labels (perfect truth) and GAN for texture (perfect realism).

## Phase 6: Next-Gen Capabilities
*Goal: Expand beyond static images and standard training.*

- [ ] **Video & Time-Series Generation**
    - [ ] **Flow Dynamics**: Simulate particles moving, rotating, and interacting in a fluid stream (Brownian motion + Shear flow).
    - [ ] **Z-Stack Simulation**: Generate a sequence of images at different focal depths (essential for training autofocus algorithms).
- [ ] **Domain Randomization (Dr)**
    - [ ] **Automatic Parameter Sweeping**: Randomize lighting, noise, and texture parameters within defined bounds to create robust training sets.
    - [ ] **Texture Synthesis**: Procedurally generate diverse backgrounds (biofilm, scratches, dust) without relying on GANs.
- [ ] **Standardized Data Export**
    - [ ] **COCO/YOLO Format**: Direct export of bounding boxes and segmentation masks in industry-standard formats.
    - [ ] **Instance Segmentation**: Pixel-perfect masks for every individual crystal, handling overlaps correctly (already supported by engine, needs export pipeline).

## Phase 7: Differentiable Microscopy (The "Inverse" Engine)
*Goal: Leverage PyTorch gradients to solve inverse problems.*

- [ ] **Inverse Rendering (Auto-Calibration)**
    - [ ] **Gradient-Based Optimization**: Feed a real image, freeze the weights, and optimize the *input parameters* (size, refractive index, light angle) to minimize the loss between Real and Synthetic.
    - [ ] **Material Property Extraction**: Infer the birefringence or thickness of a real crystal by fitting the simulation to the image.
- [ ] **End-to-End Optics Design**
    - [ ] **Learned Apertures**: Optimize the shape of the condenser aperture (mask) to maximize contrast for a specific detection task (classification/segmentation).
    - [ ] **PSF Engineering**: Design optimal Point Spread Functions for 3D localization.
- [ ] **Holographic Reconstruction**
    - [ ] **Digital Holography**: Simulate the propagation of complex fields to reconstruct 3D volume from 2D holograms.
    - [ ] **Phase Retrieval**: Recover the phase information (thickness/refractive index) from intensity-only images.

## Creative / Experimental Ideas

*   **"Virtual Dyes"**: Simulate the effect of adding Methylene Blue (absorption) or Rhodamine (fluorescence) to specific polymorphs to test if computer vision can detect them.
*   **"The Dirty Probe" Scenario**: A difficulty slider that progressively degrades image quality (fouling, scratches, bad lighting) to stress-test AI models.
*   **Auto-Calibration**: Input a real image, and use an optimizer (Gradient Descent) to tune OSOG parameters (`config.yaml`) until the synthetic image matches the real one (Inverse Rendering).
