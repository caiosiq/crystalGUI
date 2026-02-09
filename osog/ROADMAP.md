# OSOG Roadmap: From Simulator to Digital Twin

This roadmap outlines the path to transforming OSOG (Optical Synthetic Object Generator) from a high-speed data generator into a scientifically rigorous "Digital Twin" for optical microscopy.

## Current Status (v2.1 - "The Procedural Geometry Engine")
*Last Updated: 2026-02-08*

**Recently Completed:**
*   **Procedural Polyhedra (Euhedral Crystals)**:
    *   Implemented "Half-Space Intersection" engine to sculpt complex mineral shapes (Garnet, Quartz) from random planes.
    *   Full 3D rotation and physically accurate thickness calculation for arbitrary convex polyhedra.
*   **Texture System Upgrade**:
    *   Implemented Surface Roughness Maps (Striated, Pitted, Stepped) decoupled from geometry.
    *   Fixed "Smooth" texture baseline and "Hallucinations" in roughness logic.
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

- [x] **Multi-Head Architecture**
    * **Concept**: Decouple "Physical Reality" (The Particle Batch) from "Observation" (The Sensor).
    * **Capability**: Render the *exact same* particle distribution twice simultaneously: once via Transmission (Wave Optics) and once via Reflectance (Laser Scattering).
    * **Value**: Enables generation of perfect **"Paired Datasets"** (Sensor Fusion) which are impossible to capture physically.

- [x] **Reflectance Shader (PVM Mode)**
    * **Physics**: Replace Optical Path Difference (OPD) with **Surface Normal Scattering** (Bidirectional Reflectance Distribution Function - BRDF).
    * **Implementation**:
        * **Flash (Lambertian)**: Intensity $\propto \vec{N} \cdot \vec{L}$ (Dot product of surface normal and laser angle).
        * **Sparkle (Specular)**: High-frequency noise multiplication to simulate rough crystalline facets catching the light.
        * **Bloom**: Apply heavy Gaussian blur to saturated pixels to simulate sensor blooming (over-exposure).

- [x] **Surface Micro-Texture**
    * **Problem**: Smooth geometric primitives look like plastic in reflectance mode.
    * **Solution**: Apply a "Micro-Roughness" normal map to all particles before rendering. This ensures even flat crystal faces have realistic granular scattering textures.
## Phase 4.3: The "Technicolor" Update (Physics 2.1)
*Goal: Fix the "Grey Goo" problem. Real microscopy is often full of color due to interference (Polarization), dispersion (Christiansen Effect), or laser interaction.*

- [x] **Birefringence Support (Polarization)**
    * **Issue**: Current polarization mode is black because particles lack intrinsic optical anisotropy properties.
    * **Fix**: Add `birefringence` parameter to all particle configs.
    * **Effect**: Real "Michel-Levy" interference colors (Gold/Blue/Pink) based on crystal thickness and orientation.

- [x] **Thin-Film Interference (Iridescence)**
    * **Issue**: PVM images of thin plates should shimmer with color (like oil on water).
    * **Fix**: Implement `sim_thin_film_interference()` in the PVM shader.
    * **Math**: $I(\lambda) = \cos^2(2\pi n d / \lambda)$.

- [x] **Chromatic Dispersion (Rainbow Edges)**
    * **Issue**: High-index crystals act like prisms.
    * **Fix**: Split the refractive index into $n_R, n_G, n_B$ and render 3 passes with slightly different focus/refraction.

- [x] **Laser Interaction Colors (PVM)**
    * **Issue**: PVM lasers are monochromatic (e.g., 660nm Red or 405nm Blue), but our simulation is grayscale.
    * **Fix**: Add `laser_wavelength` to OpticsConfig.
    * **Effect**: Render PVM in the correct laser color (e.g., deep red), then apply Bayer filter artifacts if simulating a color camera.

## Phase 4.4: 3D Rotated Objects (The "Geometry" Update)
*Goal: Implement true 3D rotation for all particle shapes to enable realistic multi-facet rendering in brightfield.*
- [x] **Full Euler Angles**:
    * **Update**: Modify `main_generator` to sample full `alpha`, `beta`, `gamma` angles for Cubes and Plates (currently zero/fixed).
- [x] **3D Rasterization**:
    * **Update**: Modify `GeometryShader` to support rotated 3D primitives. Current implementation uses 2.5D analytical height projection which assumes face-up orientation for non-rods.
    * **Approach**: Implement SDF (Signed Distance Field) ray-marching or exact 3D mesh rasterization for Cubes/Plates to capture side faces and correct slopes.
- [x] **Fresnel Accuracy**:
    * **Impact**: Enabling side faces will allow the new `sim_brightfield` Fresnel logic to correctly render dark edges and depth cues for tumbling crystals.

## Phase 4.4.1: Spectral Brightfield (The "Prism" Engine)
*Goal: Move beyond monochrome simulation to full spectral rendering.*
- [x] **Wavelength-Dependent Refraction (Dispersion)**:
    * **Physics**: Blue light bends more than Red light ($n_{blue} > n_{red}$).
    * **Implementation**: Split render loop into 3 passes (R, G, B) with varying refractive indices.
        * Red Pass: Low RI, soft bending.
        * Green Pass: Medium RI.
        * Blue Pass: High RI, sharp bending.
- [x] **Internal Caustics (Hotspots)**:
    * **Physics**: Crystals act as lenses focusing light internally.
    * **Implementation**: Use Curvature (2nd Derivative of Height) to inject additive brightness ("Hotspots") inside shadows.
- [x] **Fresnel Rim Lighting**:
    *   **Physics**: 100% reflection at glancing angles.
    *   **Implementation**: Calculate `1.0 - (View dot Normal)` and boost brightness at the perimeter.

## Phase 4.4.1.5: The "Texture Pass" Architecture (Physics 2.5)
*Goal: Decouple physical complexity from optical rendering by introducing a Texture System.*
- [x] **Pipeline Upgrade**:
    *   **Current**: Geometry -> Texture (Texture/Physics) -> Optics.
- [x] **Roughness Map (Surface Physics)**:
    *   **Concept**: A texture that modifies the physical Height Map.
    *   **Effect**: Changes reflection/refraction angles (normal map perturbation).
    *   **Visual**: "Lumpy" spheres, growth steps on cubes, scratches.
- [x] **Transmission Map (Volume Physics)**:
    *   **Concept**: A texture defining local transparency (0.0 to 1.0).
    *   **Effect**: Blocks light passing through the object (independent of thickness).
    *   **Visual**: Internal "cloudiness", inclusions, "dirty" centers.

## Phase 4.4.1.7: Procedural Polyhedra (Euhedral Crystals)
*Goal: Achieve "many faces and many reflections" typical of minerals like Garnet or Quartz, moving away from synthetic-looking perfect cylinders and cubes.*

- [x] **Procedural Plane Sculpting**:
    - [x] **Concept**: Define a crystal by slicing empty space with random planes ($Ax + By + Cz + D = 0$) instead of using geometric primitives.
    - [x] **Ceilings**: Define planes where normal faces UP ($C > 0$). Pixel height $z \le -(Ax+By+D)/C$.
    - [x] **Floors**: Define planes where normal faces DOWN ($C < 0$). Pixel height $z \ge -(Ax+By+D)/C$.
    - [x] **Thickness**: Implement efficient vectorized min/max logic: $Thickness = \max(0, \min(Ceilings) - \max(Floors))$.
- [x] **Euhedral Habit Generator**:
    - [x] **Randomization**: Logic to generate sets of planes that form closed, convex shapes (Bipyramids, Prisms, Dodecahedra).
    - [x] **Integration**: Hook into `GeometryShader` to replace the Box/Rod analytic intersection logic for this new shape type.

## Phase 4.4.2: The Virtual Microscope (The "Lens" Engine)
*Goal: Simulate mechanical and optical limitations of the camera system.*

### Phase 4.4.2.1: The Physics of Light (Fundamental)
*Goal: Implement the core wave-optics behaviors that define how light interacts with matter.*

- [ ] **Refractive Index Matching (Index-Matched Background)**
    *   **Physics**: When crystal RI matches solvent RI, reflection/refraction vanishes ("Invisible Crystals").
    *   **Implementation**:
        *   Add `solvent_ri` (Refractive Index of Medium) to `OpticsConfig`.
        *   Add `solvent_color` to `SensorConfig` to tint background based on chemical context (e.g., yellowish oil).
        *   Scale Fresnel intensity by $\Delta n = |n_{crystal} - n_{solvent}|$.
        *   UI: Add "Solvent/Background" section to Playground.
- [ ] **Fresnel Edge Darkening**
    *   **Physics**: In brightfield, transparent crystals look like dark outlines because light hitting the edge refracts away from the camera.
    *   **Implementation**: Modify shader to correctly map transmission $T \approx 1 - Fresnel(\theta)$.
- [ ] **Internal Scattering (Cloudiness)**
    *   **Physics**: Real industrial crystals aren't perfect glass; they are milky.
    *   **Implementation**: Add a `turbidity` parameter to the ray-marcher (Volumetric Fog inside the crystal).

### Phase 4.4.2.2: The Physics of the Probe (Instrumental)
*Goal: Simulate the specific hardware configurations of industrial probes (Blaze/FBRM).*

- [ ] **Blaze/Directional Lighting**
    *   **Physics**: Blaze probes often use specific illumination angles to highlight edges.
    *   **Implementation**: Instead of a fixed "Headlamp" light, allow the user to define a `LightSource(direction, intensity, spread)`. This allows mimicking different probe brands.

### Phase 4.4.2.3: The Physics of the Sensor (Artifacts)
*Goal: Simulate the imperfections of the imaging system.*

- [ ] **Depth of Field (DoF) & Bokeh**:
    * **Physics**: High NA objectives have thin focal planes. Out-of-focus highlights form Airy disks.
    * **Implementation**: Use HeightMap/Z-Buffer. Apply blur proportional to distance. For "Ghosts" (Deep-Z), implement Disk Kernel Blur to create translucent circles instead of Gaussian fog.
- [ ] **Becke Lines (Diffraction Halos)**:
    * **Physics**: Diffraction halos at refractive boundaries that move with focus.
    * **Implementation**: Apply "Unsharp Mask" (High-Pass Filter) to intensity image.
- [ ] **Sensor Noise, Bloom & Motion**:
    * **Physics**: Shot noise, electron spillover, and motion smear.
    * **Implementation**: Add Poisson noise. Apply "Glare" to saturated pixels. Implement Directional Blur aligned with particle flow velocity.
- [ ] **Multi-Scale Contaminants**:
    * **Physics**: Dirty water/solvent has micro-fines and density schlieren.
    * **Implementation**: Layer of sub-visible speckles and low-frequency distortion noise.

## Phase 4.5: Advanced Geometry & Habits (The "Morphology" Update)
*Goal: Support complex, non-primitive crystal shapes and biological forms.*
- [ ] **Geometric Crystal Habits (The "Adipic" Shader)**:
    * **Problem**: Perlin noise looks organic. Real crystals have sharp growth steps.
    * **Solution**: Implement "Terrace Texture" generator (recursive rectangular masking) to simulate prism layers.
- [ ] **Polyhedral Geometry (The "Glycine" Shader)**:
    * **Problem**: Current shapes are just stretched cubes/plates.
    * **Solution**: Procedural generation based on **Miller Indices** (e.g., bipyramids) with internal refraction logic for angled faces.
- [ ] **Flexible Filament Engine (The "Insulin" Shader)**:
    * **Problem**: Rigid cylinders cannot model amyloid fibrils.
    * **Solution**: Switch to **Cubic Bezier Tubes** that can bend and twist ("Spaghetti Dynamics").
- [ ] **Amorphous & Liquid Shader**:
    * **Problem**: Need to simulate "Oiling Out" and non-crystalline agglomerates.
    * **Solution**: Metaball rendering (Smooth Union SDF) and "Stucco" normal maps.

## Phase 4.6: Advanced Lighting & Physics
- [ ] **Contact Shadows (SSAO)**:
    * **Problem**: Dense clusters lose definition.
    * **Solution**: Screen-Space Ambient Occlusion to darken crevices between touching particles.
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

## Phase 8: Active Domain Adaptation (The "Teacher" Engine)
*Goal: Close the loop between the human expert in the Playground and the AI model.*

- [ ] **Interactive "Hard Mining"**
    - [ ] **Failure Case Annotation**: Allow users to mark regions in real images where the model fails.
    - [ ] **Inverse Parameter Search**: OSOG automatically attempts to find synthetic parameters that generate similar features to the failure case.
- [ ] **Synthetic Fine-Tuning Loop**
    - [ ] **One-Click "Fix This Error"**: Generate a mini-batch of variations based on the failure case.
    - [ ] **Local Adaptation**: Fine-tune the model locally on this targeted synthetic data and re-evaluate.
- [ ] **Style Transfer (Sim2Real)**
    - [ ] **CycleGAN Integration**: Implement a post-processing layer to bridge the texture gap between synthetic and real images while preserving label integrity.

## Creative / Experimental Ideas

*   **"Virtual Dyes"**: Simulate the effect of adding Methylene Blue (absorption) or Rhodamine (fluorescence) to specific polymorphs to test if computer vision can detect them.
*   **"The Dirty Probe" Scenario**: A difficulty slider that progressively degrades image quality (fouling, scratches, bad lighting) to stress-test AI models.
*   **Auto-Calibration**: Input a real image, and use an optimizer (Gradient Descent) to tune OSOG parameters (`config.yaml`) until the synthetic image matches the real one (Inverse Rendering).
