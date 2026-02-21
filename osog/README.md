# OSOG: Optical Synthetic Object Generator
## A Differentiable Engine for Wave-Propagation Microscopy

OSOG is a high-performance, GPU-accelerated synthetic data generation engine designed to simulate optical microscopy images. Unlike traditional renderers (like Blender) that rely on **ray tracing** (geometric optics), OSOG is built on **wave optics** principles to accurately replicate diffraction, interference, and phase effects that dominate at the microscopic scale.

Built entirely in **PyTorch**, OSOG is fully **differentiable**, opening the door to inverse rendering, auto-calibration, and end-to-end optimization of optical systems.

## Simulation Capabilities

OSOG provides a vast array of configurable parameters, all available through the interactive **OSOG Playground**.

### 1. Canvas & Compute
*   **Resolution**: Adjustable canvas size (e.g., 1024x1024).
*   **GPU Acceleration**: Fully vectorized pipeline for real-time generation on CUDA-enabled devices.

### 2. Particles & Physics
Define the physical properties of the sample being imaged.
*   **3D Rotation**: Enable full 3D orientation for particles.
*   **Particle Types**:
    *   **Rods**: Main crystalline structures. Configurable Count, Length, Aspect Ratio.
    *   **Spheres**: Beads or droplets. Configurable Count, Diameter.
    *   **Cubes**: Cubic crystals. Configurable Count, Size.
    *   **Plates**: Flat crystalline plates. Configurable Count, Size, Aspect Ratio, Thickness.
    *   **Polyhedra (Euhedral Crystals)**: Procedurally sculpted mineral shapes (Garnet, Quartz) using half-space intersection.
*   **Materials**: Simulate different refractive indices and optical properties.
    *   *Standard (Plastic)*
    *   *Fibrous Crystal (Needles)*
    *   *Smooth Crystal (Salts)*
    *   *High Birefringence (Urea)*
    *   *Amorphous (Aggregates)*
    *   *Glass (Beads)*
    *   *Metal (Shavings)*
    *   *Air (Bubbles)*
    *   *Oil (Droplets)*

### 3. Morphology & Interactions
Control the fine-grained details of particle formation and interaction.
*   **Surface Morphology**:
    *   **Texture System**: Decoupled surface roughness maps (Striated, Pitted, Granular).
    *   **Roughness**: Simulate surface imperfections.
    *   **Polarity Flip**: Create "dark" crystal variations.
    *   **Inclusions**: Simulate cracks or internal defects.
    *   **Shape Mode**: Straight, Wavy, Kinked, or Noisy particle shapes.
*   **Agglomeration & Growth**:
    *   **Agglomeration**: Control the tendency of particles to clump together.
    *   **DLCA**: Diffusion-Limited Cluster Aggregation for fractal-like growth.
    *   **Sintering**: Simulate neck growth between fused particles.
    *   **Cluster Types**: Random, Stacking (3D), Chain, Cross (90°), Dendrite (Snowflake), Spherulite.
*   **Dynamics**:
    *   **Flow Alignment**: Simulate fluid flow aligning particles (Direction, Shear Rate).
    *   **Sedimentation**: Simulate gravity effects and settling strength.
    *   **Size Segregation**: Brazil-nut effect (larger particles rising).
    *   **Debris**: Add background particulate noise.
    *   **Ghosts**: Out-of-focus background particles.

### 4. Optics & Microscopy
Simulate the microscope itself.
*   **Imaging Modes**:
    *   **DIC (Differential Interference Contrast)**: Standard pseudo-3D phase imaging.
    *   **Brightfield**: Standard absorption imaging.
    *   **Polarization (Crossed)**: Birefringence visualization (dark background, bright crystals).
    *   **Polarization (RGB)**: Michel-Levy interference colors.
    *   **Fluorescence**: Widefield fluorescence simulation.
    *   **Confocal**: Optical sectioning.
    *   **Shadowgraphy**: Projection imaging.
    *   **PVM (Reflectance)**: Laser backscatter simulation (Flash/Sparkle/Bloom) for in-situ probes.
*   **Parameters**:
    *   **Polarizer Angle**: Rotate the polarizer for birefringence effects.
    *   **Shadow Gain**: Adjust the contrast strength of phase gradients.
    *   **Focus Plane (Z)**: Move the focal plane through the 3D sample.
    *   **Laser Wavelength**: Control PVM laser color (e.g., 660nm Red).

### 5. Sensor & Artifacts
Simulate the camera and environmental imperfections.
*   **Sensor Noise**: Gaussian/Poisson shot noise.
*   **Blur**: Gaussian blur (simulating lens quality or motion).
*   **Vignette**: Darkening at the image corners.
*   **Chromatic Aberration**: Color fringing at high-contrast edges.
*   **Background**:
    *   **Tilt Gradient**: Uneven illumination.
    *   **Relief Texture**: Non-uniform background substrate.
*   **Imaging Artifacts**:
    *   **Bubbles**: Air bubbles trapped in the medium (with adhesion probability).
    *   **Droplets**: Oil/liquid droplets.
    *   **Lens Fouling**: Dust or smudges on the lens optics.

## Why OSOG? (The "Blender Problem")

Standard 3D engines (Blender, Unity, Unreal) are excellent for macroscopic scenes but fail at the micron scale.
*   **Ray Tracing**: Simulates light as particles traveling in straight lines. Great for shadows and reflections, but it cannot naturally handle the wave nature of light.
*   **Microscopy**: At this scale, light behaves as a wave. Effects like **diffraction limits (Airy disks)**, **interference (DIC/Phase Contrast)**, **birefringence**, and **depth-dependent point spread functions (PSF)** are fundamental.

OSOG solves this by simulating the **optical path difference (OPD)** and **phase shifts** of light passing through objects, then propagating this wavefront through a virtual microscope (Objective -> Tube Lens -> Sensor).

## 1. Architecture Overview

OSOG operates on a **fully vectorized, GPU-native pipeline**. Unlike traditional object-oriented simulators that process objects sequentially (Loop-over-Objects), OSOG treats the entire scene as a set of tensors. This allows it to leverage the massive parallelism of modern GPUs (e.g., A100), rendering thousands of objects in milliseconds.

### Core Philosophy
*   **Vectorization**: All physics parameters (positions, angles, sizes) are generated as tensors `(N,)`.
*   **Batch Rendering**: Objects are rendered in batches `(N, C, H, W)` rather than individually.
*   **Zero-Copy**: Data stays on the VRAM (GPU memory) from generation to final sensor simulation. The only CPU transfer happens at the very end when exporting the final image.
*   **Differentiability**: Every operation (from rotation to rendering) is a differentiable PyTorch operation, allowing gradients to flow backward from the image pixel to the physical parameter.

## 2. Module Structure

```
osog/
├── config.py           # Centralized configuration management (Hydra-like dicts)
├── core/
│   ├── pipeline.py     # The main orchestrator. Manages the generation loop.
│   └── canvas.py       # Wrapper for the image buffer.
├── physics/
│   ├── distribution.py # The "Reactor". Generates random distributions of particles.
│   ├── particles.py    # Data structures (ParticleBatch, DebrisBatch) holding SoA tensors.
│   └── generators/     # Specialized generators (main_generator.py).
├── optics/
│   ├── optical_engine.py # Coordinate system for multi-stage rendering.
│   ├── shaders/          # Modular shader system.
│   │   ├── geometry.py   # Geometry Pass (SDF/Height Map).
│   │   ├── texture.py    # Texture Pass (Roughness/Transmission).
│   │   └── dic_torch.py  # Optical Pass (DIC/Brightfield/Polarization).
│   └── sensor_torch.py   # GPU Sensor simulation (Noise, Blur, Background).
└── utils/
    └── math_torch.py   # Vectorized math helpers (interpolation, noise generation).
```

## 3. How It Works: The Pipeline

The generation process (`Pipeline.generate`) follows a strict linear flow designed to minimize kernel launches and memory movement.

### Step 1: Physics Generation ("The Reactor")
*   **Input**: Configuration (density, size distribution) + Random Seed.
*   **Process**:
    *   `generate_distribution` calculates random positions, angles, and dimensions for $N$ particles and debris.
    *   Instead of creating $N$ Python objects, it populates a `ParticleBatch` (Structure-of-Arrays).
    *   Example: `batch.cx` is a CUDA tensor of shape `(2000,)`.
*   **Output**: `ParticleBatch`, `DebrisBatch` on GPU.

### Step 2: Background Generation ("The Sensor")
*   **Process**: `SensorHeadTorch` generates the background canvas directly on the GPU.
*   **Features**:
    *   Perlin-like low-frequency noise (illumination unevenness).
    *   Directional gradients (tilt).
    *   Sensor noise (Gaussian/Poisson).
*   **Optimization**: Large illumination blurs are approximated via upsampling small noise tensors to avoid massive convolutions.

### Step 3: Geometry & Texture Pass ("The Sculptor")
*   **Process**: `GeometryShader` computes the physical shape of each particle.
    *   **Analytic Projection**: Calculates exact 2D bounds of 3D rotated boxes/rods.
    *   **Ray-Slab Intersection**: Computes exact thickness at every pixel for 3D shapes.
    *   **Procedural Sculpting**: Carves complex Polyhedra from random planes.
    *   **Texture Mapping**: `TextureShader` applies roughness and internal inclusion maps.
*   **Output**: G-Buffer `(N, 4, H_patch, W_patch)` containing Height, Mask, Delta, and Orientation.

### Step 4: Optical Rendering ("The Modulator")
*   **Process**: `OpticalEngine` transforms the G-Buffer into an optical intensity image.
    *   **DIC**: Computes gradient of optical path length.
    *   **Polarization**: Applies Maltese Cross pattern based on orientation and birefringence.
    *   **PVM**: Simulates laser scattering off surface normals.
*   **Output**: A tensor of rendered patches `(N, C, H_patch, W_patch)`.

### Step 5: Composition ("The Stamper")
*   **Problem**: Adding 2000 small images to a large canvas sequentially in Python causes massive "Kernel Launch Overhead" (CPU waiting for GPU).
*   **Solution**: `_stamp_tensor_batch` uses `torch.index_put_` (Scatter Add).
    *   It calculates global indices for every pixel of every patch in one go.
    *   It executes a single atomic add kernel to paste all objects onto the canvas simultaneously, handling overlaps correctly.

### Step 6: Sensor Artifacts & Export
*   **Process**:
    *   Global blur (simulating optics quality) is applied to the full canvas on GPU.
    *   The final tensor is downloaded to CPU only at this stage.
    *   Scalebars and text overlays are drawn using CPU libraries (Pillow/OpenCV) on the final NumPy array.

## 4. Key Technical Concepts

*   **Structure-of-Arrays (SoA)**: We use `ParticleBatch` (storing columns of data) instead of `List[Particle]`. This is cache-friendly and GPU-native.
*   **Procedural Plane Sculpting**: Complex minerals are generated by defining them as the intersection of random half-spaces, allowing for infinite variation in crystal habit.
*   **Multi-Head Rendering**: The engine can render the *exact same* physical batch in multiple optical modes (e.g., DIC + Fluorescence + Depth Map) simultaneously, generating perfect paired datasets for sensor fusion.

## 5. Usage

```python
from crystalGUI.osog.config import SynthConfig
from crystalGUI.osog.core.pipeline import Pipeline

# 1. Load Config (Object-based or Dict-based)
config = SynthConfig()
config.canvas.width = 1024
config.canvas.height = 1024
config.canvas.use_gpu = True
config.physics.rods.n_rods_rng_lo_hi = (1000, 1500)

# 2. Initialize Pipeline
# Pipeline accepts the config object or its dictionary representation
pipe = Pipeline(config.to_dict())

# 3. Generate
# Returns a numpy array (H, W, 3) ready for saving or training
image, labels = pipe.generate(t=0.0, return_obbs=True)
```
