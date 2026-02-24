# OSOG Roadmap: From Simulator to Digital Twin

This roadmap outlines the path to transforming OSOG (Optical Synthetic Object Generator) from a high-speed data generator into a scientifically rigorous "Digital Twin" for optical microscopy.

## Current Status (v2.1 - "The Procedural Geometry Engine")
*Last Updated: 2026-02-23*

**Recently Completed:**
*   **Advanced Physics (Phase 4.4.2)**:
    *   **Solvent Displacement**: Implemented Beer-Lambert differential absorption (`exp(-h * (mu_particle - mu_solvent))`) to correctly simulate "invisible" or "bright" crystals in colored solvents.
    *   **Refractive Index Matching**: Full Fresnel implementation ($T \approx (1 - R)^2$) where reflection depends on $\Delta n$.
    *   **Volumetric Turbidity**: Added internal scattering (fog) parameter for milky crystals.
    *   **Dynamic Lighting**: Added configurable `light_direction` vector to simulate oblique illumination (Blaze probes) and shadow casting.

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
*   **The "Soup" Update (Phase 4.4.2.4)**:
    *   **Procedural Backgrounds**: Replaced heavy particle-based debris with performant, infinite procedural noise.
    *   **Anisotropy**: Added "Stretch" control to simulate directional flow blur in the background.
    *   **Lens Fouling**: Implemented realistic lens dirt and biofilm occlusion layers that correctly overlay the scene.
    *   **Shallow Depth of Field**: Implemented physically-based Z-blur (Circle of Confusion) for foreground particles.

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

- [x] **Reflectance Shader (PVM Mode) [LEGACY/REMOVED]**
    * **Note**: PVM mode has been deprecated and removed from the active simulator in favor of Blaze/Darkfield.
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

- [x] **Refractive Index Matching (Index-Matched Background)**
    *   **Physics**: When crystal RI matches solvent RI, reflection/refraction vanishes ("Invisible Crystals").
    *   **Implementation**:
        *   Added `solvent_ri` (Refractive Index of Medium) to `OpticsConfig`.
        *   Added `solvent_color` to `SensorConfig` to tint background based on chemical context (e.g., yellowish oil).
        *   Scale Fresnel intensity by $\Delta n = |n_{crystal} - n_{solvent}|$.
        *   UI: Added "Solvent/Background" section to Playground.
- [x] **Fresnel Edge Darkening**
    *   **Physics**: In brightfield, transparent crystals look like dark outlines because light hitting the edge refracts away from the camera.
    *   **Implementation**: Modified shader to correctly map transmission $T \approx 1 - Fresnel(\theta)$.
- [x] **Internal Scattering (Cloudiness)**
    *   **Physics**: Real industrial crystals aren't perfect glass; they are milky.
    *   **Implementation**: Added a `turbidity` parameter to the ray-marcher (Volumetric Fog inside the crystal).

### Phase 4.4.2.2: The Physics of the Probe (Reflectance/Darkfield)
*Goal: Simulate the specific optical physics of backscatter probes, where light comes from the camera side (Episcopic) rather than behind the object (Diascopic).*

- [x] **Episcopic Lighting Model (Ring Light)**
    *   **Physics**: Light originates from the probe tip (around the lens). The Light Vector ($L$) is roughly parallel to the View Vector ($V$).
    *   **Implementation**: Create `sim_blaze` shader where $L \approx [0, 0, 1]$. Background is set to black (0.0) or dark solvent color.
    *   **Update**: Implemented multi-sample Ring Light (8 points) to simulate cone illumination.
- [x] **Specular Reflection (The "Flash")**
    *   **Physics**: Facets perpendicular to the camera reflect light directly back. This creates the characteristic bright white "flashes" seen in Blaze images.
    *   **Implementation**: Calculate Blinn-Phong Specular intensity weighted by Fresnel Reflection ($R$). High $\Delta n$ = Brighter Glints.
- [x] **Roughness-Induced Edge Scattering**
    *   **Physics**: Smooth faces reflect light away (dark), but rough edges scatter light everywhere (bright). This makes crystal edges glow in Darkfield.
    *   **Implementation**: Use the `roughness_map` to modulate a diffuse lighting term. High roughness + High slope = Bright Pixel.
- [x] **Internal Backscatter (The "Milky Glow")**
    *   **Physics**: Light enters the crystal, hits internal defects/turbidity, and bounces back to the camera.
    *   **Implementation**: Add a volumetric term `turbidity * thickness` that adds brightness (instead of subtracting it like in Brightfield).
- [x] **Dispersion Sparkles (Chromatic Aberration in Reflection)**
    *   **Physics**: High-refractive-index crystals split white light into colored sparkles at the edges of specular highlights.
    *   **Implementation**: Shift the Specular Highlight calculation slightly for R, G, and B channels based on the dispersion parameter.
- [x] **High-Gain Sensor Noise**
    *   **Physics**: Reflectance probes often operate in dark environments with high gain, leading to significant shot noise.
    *   **Implementation**: Add a post-process Gaussian + Poisson noise layer specific to the Blaze mode.

### Phase 4.4.2.3: The Blaze Complexity Gap (Texture & Volumetrics)
*Goal: Bridge the "Uncanny Valley" between geometric renders and organic reality in Darkfield/Reflectance mode.*

#### Phase 4.4.2.3.1: Advanced Surface Physics (The "Skin" Update)
*Focus: Realistic light interaction with the crystal surface boundary.*
- [x] **Anisotropic Surface Texture (Striations)**
    *   **Problem**: Current roughness is uniform (isotropic), looking like sandblasted plastic. Real crystals have directional growth lines.
    *   **Solution**: Implement `TextureShader` with **Anisotropic Noise** (stretched Perlin noise).
    *   **Physics**: These micro-grooves act as diffraction gratings, catching light only at specific angles ($\vec{N} \cdot \vec{L}$ is high only when groove aligns).
- [x] **Micro-Topography & Incrustations**
    *   **Problem**: Crystal silhouettes are too perfect. Real crystals have jagged edges (anhedral growth) and surface debris.
    *   **Solution**:
        *   [x] **Stochastic Injection Map**: Use a noise map to gate where light enters the crystal "waveguide" (non-uniform injection).
        *   [x] **Surface Incrustation**: Add a new `IncrustationBatch` to stamp tiny high-frequency geometry *onto* the surface of main rods.
        *   [x] **SDF Jitter**: Add a "Perturbation Pass" to the `GeometryShader` that jitters the Signed Distance Field (SDF) boundary to create jagged edges.

#### Phase 4.4.2.3.2: Volumetric Complexity (The "Guts" Update)
*Focus: Internal heterogeneity and light transport inside the crystal.*
- [x] **Fractal Volumetric Inclusions**
    *   **Problem**: Current `turbidity` is a constant scalar. Real crystals have "milky" clusters and gradients.
    *   **Solution**: Replace constant turbidity with a **3D Fractal Noise Volume** (Cloud Map).
    *   **Physics**: Simulates Mie scattering from heterogeneous micro-defects deep inside the crystal body.
- [x] **Polycrystalline Aggregates (Grain Boundaries)**
    *   **Problem**: "Static" bodies look like single crystals. Real aggregates are messy.
    *   **Solution**: Use a **Cellular Noise (Voronoi)** map to modulate reflectivity across the face, simulating internal grain boundaries where light gets lost.

#### Phase 4.4.2.3.3: Optical Realism (The "Lens" Update)
*Focus: How the scattered light is processed by the imaging system.*
- [x] **Diffractive Bloom & PSF**
    *   **Problem**: Gaussian blur looks too soft. Real laser probes have distinct "Star" or "Glare" patterns.
    *   **Solution**: Replace Gaussian kernel with a **Lorentzian Point Spread Function (PSF)** (Star Filter).
    *   **Effect**: Creates "Long Tail" halos and diffraction spikes around bright specular highlights.
- [x] **Spectral Dispersion (Chromatic Aberration)**
    *   **Problem**: Edges look too sharp and monochrome.
    *   **Solution**: Slightly offset R, G, and B channels based on the surface gradient to replicate faint "rainbow" fringing.

### Phase 4.4.2.4: The Physics of the Sensor (Artifacts)
*Goal: Simulate the imperfections of the imaging system and train ML models to ignore what they shouldn't measure.*

- [x] **Depth of Field (DoF) & Bokeh (The "Soup" Layer)**:
    * **Physics**: High NA objectives have extremely thin focal planes. Thousands of crystals sit just behind the focal plane, forming a dense, glowing "soup" of overlapping Airy disks/Bokeh.
    * **ML Strategy**: **Distractor Backgrounds**. These out-of-focus blobs must NOT be labeled. The model must learn to ignore them.
    * **Implementation**:
        *   **Procedural "Soup"**: Replaced heavy particle generation with a lightweight procedural noise generator (Anisotropic 2.5D Noise) directly on the tensor.
        *   **Smart Compositing**:
            *   **Darkfield/Blaze**: Additive blend (Glow).
            *   **Brightfield**: Multiplicative/Subtractive blend (Shadows).
        *   **Foreground Embedding**: Render sharp, labeled crystals on top of this glowing/shadowy background.

- [x] **Shallow Depth of Field (Zone 2)**:
    *   **Physics**: Apply Z-dependent blur (Circle of Confusion) to foreground particles.
    *   **Implementation**: Used "Layered Blending" (Sharp/Medium/High blur layers blended by CoC) for fast, realistic Bokeh.
    *   **ML Safety**: Blurred particles are still labeled, preventing false negatives.

- [x] **Window Fouling (Lens Dirt)**:
    *   **Physics**: Dirt, smudges, and biofilm on the lens/window occlude the view and scatter light.
    *   **Implementation**: Added `apply_fouling` pass in `SensorHeadTorch`.
        *   **Static Blobs**: Discrete dirt particles on the lens surface (defocused).
        *   **Biofilm**: Low-frequency noise overlay simulating organic residue.
    *   **Differentiation**: Unlike the background "soup" which moves with flow, fouling is static relative to the camera.

- [ ] **Sensor Noise, Bloom & Motion**:
    * **Physics**: Shot noise, electron spillover, and motion smear.
    * **Implementation**: Add Poisson noise. Apply "Glare" to saturated pixels. Implement Directional Blur aligned with particle flow velocity.

- [ ] **Becke Lines (Diffraction Halos)**:
    * **Physics**: Diffraction halos at refractive boundaries that move with focus.
    * **Implementation**: Apply "Unsharp Mask" (High-Pass Filter) to intensity image.

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
- [ ] **Advanced Optical Modalities (Beyond Microscopy)**
    - [ ] **Raman Chemical Mapping**: Simulate hyperspectral cubes where pixel intensity corresponds to specific chemical bond vibrations (e.g., differentiating Polymorph A vs B based on spectra, not just shape).
    - [ ] **Schlieren/Phase Contrast**: Visualize fluid density gradients (mixing, dissolution boundary layers) around dissolving crystals.
    - [ ] **Light Scattering (MALS/DLS)**: Simulate the far-field diffraction pattern (Fourier transform of the image) used by laser diffraction sizers.

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
