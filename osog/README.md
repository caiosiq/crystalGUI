# OSOG: Optical Synthetic Object Generator

OSOG is a high-performance, GPU-accelerated synthetic data generation engine designed to simulate optical microscopy images of rod-like particles (crystals) and debris. It is built to generate large-scale datasets for training machine learning models, focusing on physical realism (DIC effects, depth of field, diffraction) and computational efficiency.

## 1. Architecture Overview

OSOG operates on a **fully vectorized, GPU-native pipeline**. Unlike traditional object-oriented simulators that process objects sequentially (Loop-over-Objects), OSOG treats the entire scene as a set of tensors. This allows it to leverage the massive parallelism of modern GPUs (e.g., A100), rendering thousands of objects in milliseconds.

### Core Philosophy
*   **Vectorization**: All physics parameters (positions, angles, sizes) are generated as tensors `(N,)`.
*   **Batch Rendering**: Objects are rendered in batches `(N, C, H, W)` rather than individually.
*   **Zero-Copy**: Data stays on the VRAM (GPU memory) from generation to final sensor simulation. The only CPU transfer happens at the very end when exporting the final image.

## 2. Module Structure

```
osog/
├── config.py           # Centralized configuration management (Hydra-like dicts)
├── core/
│   ├── pipeline.py     # The main orchestrator. Manages the generation loop.
│   └── canvas.py       # Wrapper for the image buffer.
├── physics/
│   ├── distribution.py # The "Reactor". Generates random distributions of particles.
│   ├── particles.py    # Data structures (RodBatch, DebrisBatch) holding SoA tensors.
│   └── ghosts.py       # Logic for out-of-focus "ghost" particles.
├── optics/
│   ├── dic_torch.py    # The GPU Renderer. Simulates Differential Interference Contrast.
│   └── sensor_torch.py # GPU Sensor simulation (Noise, Blur, Background, Scalebars).
└── utils/
    └── math_torch.py   # Vectorized math helpers (interpolation, noise generation).
```

## 3. How It Works: The Pipeline

The generation process (`Pipeline.generate`) follows a strict linear flow designed to minimize kernel launches and memory movement.

### Step 1: Physics Generation ("The Reactor")
*   **Input**: Configuration (density, size distribution) + Random Seed.
*   **Process**:
    *   `generate_distribution` calculates random positions, angles, and dimensions for $N$ rods and debris.
    *   Instead of creating $N$ Python objects, it populates a `RodBatch` (Structure-of-Arrays).
    *   Example: `rods.cx` is a CUDA tensor of shape `(2000,)`.
*   **Output**: `RodBatch`, `DebrisBatch` on GPU.

### Step 2: Background Generation ("The Sensor")
*   **Process**: `SensorHeadTorch` generates the background canvas directly on the GPU.
*   **Features**:
    *   Perlin-like low-frequency noise (illumination unevenness).
    *   Directional gradients (tilt).
    *   Sensor noise (Gaussian/Poisson).
*   **Optimization**: Large illumination blurs are approximated via upsampling small noise tensors to avoid massive convolutions.

### Step 3: Optical Rendering ("The Modulator")
*   **Process**: `DICModulatorTorch` takes the batches and renders them into small patches `(N, 3, H_patch, W_patch)`.
*   **Vectorized Math**:
    *   Computes optical path differences (OPD) for all pixels of all rods in parallel.
    *   Applies warping (bending, kinking) using vectorized grid sampling.
    *   Simulates DIC shear and shadow effects using tensor operations.
*   **Output**: A tensor of rendered patches and their top-left coordinates `(x_min, y_min)`.

### Step 4: Composition ("The Stamper")
*   **Problem**: Adding 2000 small images to a large canvas sequentially in Python causes massive "Kernel Launch Overhead" (CPU waiting for GPU).
*   **Solution**: `_stamp_tensor_batch` uses `torch.index_put_` (Scatter Add).
    *   It calculates global indices for every pixel of every patch in one go.
    *   It executes a single atomic add kernel to paste all objects onto the canvas simultaneously, handling overlaps correctly.

### Step 5: Sensor Artifacts & Export
*   **Process**:
    *   Global blur (simulating optics quality) is applied to the full canvas on GPU.
    *   The final tensor is downloaded to CPU only at this stage.
    *   Scalebars and text overlays are drawn using CPU libraries (Pillow/OpenCV) on the final NumPy array.

## 4. Key Technical Concepts

*   **Structure-of-Arrays (SoA)**: We use `RodBatch` (storing columns of data) instead of `List[Rod]`. This is cache-friendly and GPU-native.
*   **DIC Simulation**: Differential Interference Contrast is simulated by computing the gradient of the optical path length in the shear direction, creating the characteristic pseudo-3D shadow effect.
*   **Depth of Field**: Objects have a `z` coordinate. A depth-dependent Gaussian blur is applied efficiently using separable 1D convolutions on the GPU.

## 5. Usage

```python
from osog.config import SynthConfig
from osog.core.pipeline import Pipeline

# 1. Load Config
config = {
    "canvas": {"width": 1024, "height": 1024, "use_gpu": True},
    "physics": {"rods": {"n_rods_rng_lo_hi": (1000, 1500)}}
}

# 2. Initialize Pipeline
pipe = Pipeline(config)

# 3. Generate
# Returns a numpy array (H, W, 3) ready for saving or training
image, labels = pipe.generate(t=0.0, return_obbs=True)
```
