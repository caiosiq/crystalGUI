# DiffOSOG: Differentiable Calibration Engine

**DiffOSOG** (Differentiable Optical Simulation of Organic Growth) is an advanced framework for **automatic parameter calibration** of the OSOG rendering engine. It bridges the gap between synthetic data generation and real-world microscope images by using differentiable rendering and gradient descent.

Instead of manually tuning simulation parameters (like focus depth, noise levels, and optical blur) to match an experimental dataset, DiffOSOG formulates this as an "inverse rendering" problem. It iteratively refines the simulation parameters by minimizing a composite loss function until the synthetic image structurally and texturally matches a target reference image.

## Core Concepts

### 1. Automatic Curriculum Learning
Optimizing highly non-linear rendering parameters simultaneously often leads to local minima. DiffOSOG solves this via an **Auto-Scheduler** that breaks the optimization into distinct, logical stages:
*   **Stage 1: Geometry & Optics**: Focuses on macroscopic structural alignment (e.g., optical blur, defocus, particle sizing). It uses a *locked random seed* to ensure particle positions remain static, preventing chaotic gradients.
*   **Stage 2: Texture & Noise**: Freezes geometry parameters and focuses entirely on high-frequency details (e.g., sensor noise, vignette, distractor opacity).
*   **Stage 3: Fine-Tuning**: Unfreezes all parameters for a final, low-learning-rate joint optimization pass to ensure harmony between geometry and texture.

### 2. Optimization Rules (`optimization_rules.json`)
The behavior of the curriculum is strictly governed by `optimization_rules.json`. This acts as the "brain" of the calibration engine, defining:
*   **Bounds & Scaling**: Min/max physical limits and scaling logic (linear vs. log) for each parameter.
*   **Stage Assignment**: Which curriculum stage (Geometry vs. Texture) a parameter belongs to.
*   **Loss Weights**: Stage-specific weights dictating which loss functions to prioritize.
*   **Learning Rate Multipliers**: Per-parameter sensitivity tuning.

### 3. Semantic Loss Routing (`src/loss/`)
A simple pixel-wise difference (MSE/L1) fails spectacularly for synthetic-to-real matching due to inherent misalignments. DiffOSOG employs a suite of semantic losses:
*   **Perceptual Loss (`perceptual.py`)**: Uses a pre-trained VGG-16 network to extract feature maps. By comparing deep features, it matches the overall "look and feel" and structural layout regardless of exact pixel alignment. Highly weighted in the **Geometry** stage.
*   **Spectral Loss (`spectral.py`)**: Uses Fast Fourier Transforms (FFT) to compare the frequency domain. Excellent for matching the overall sharpness, blur characteristics, and global optical properties.
*   **Texture / Gram Loss (`texture.py`)**: Computes the Gram matrix of VGG features to capture style and texture independent of spatial structure. Heavily weighted in the **Texture** stage to match background noise, grain, and lighting artifacts.
*   **Histogram Loss**: Ensures the overall brightness and contrast distribution matches the target.

## Directory Structure

*   `src/`: Core implementation files.
    *   `diff_wrapper.py`: Differentiable PyTorch wrapper for the OSOG rendering pipeline.
    *   `calibration_engine.py`: Orchestrates the optimization loop, backward passes, and optimizer stepping.
    *   `parameter_manager.py`: Handles bidirectional mapping between physical config values and bounded/normalized latent tensors.
    *   `auto_scheduler.py`: Manages the Curriculum Learning (Geometry -> Texture -> Fine-Tune).
    *   `loss_manager.py`: Aggregates the various loss functions according to stage-specific weights.
    *   `convergence_guard.py`: Monitors loss plateaus to trigger early stopping or stage advancement.
    *   `robustness.py`: Ensures stability via gradient clipping and NaN detection.
*   `src/loss/`: Custom loss function implementations (VGG Perceptual, Spectral FFT, Texture Gram).
*   `tests/`: Validation and diagnostic scripts.
    *   `validate_engine.py`: Full end-to-end academic validation, producing parameter trajectory plots and loss curves.
*   `optimization_rules.json`: The central configuration dictating the calibration curriculum.

## Usage

### Running Validation
To test the full calibration cycle against a synthetic ground truth (ideal for academic reporting):

```bash
python3 diff_calibration/tests/validate_engine.py
```

This script will:
1.  Generate a "Ground Truth" image with known parameters.
2.  Initialize the engine with deliberately poor parameters.
3.  Execute the multi-stage curriculum optimization to recover the GT parameters.
4.  Output high-quality, publication-ready plots (Loss Convergence, Parameter Trajectory) to `diff_calibration/validation_output/engine_test/`.

### Debugging Gradients
If optimization stalls or behaves erratically, use the debug script to check gradient flow for specific parameters:

```bash
python3 diff_calibration/tests/debug_grads_v2.py
```

## Configuration

The optimization behavior is controlled by `optimization_rules.json`. You can adjust:
*   **Bounds:** Min/max values for each parameter.
*   **Stage:** Which curriculum stage a parameter belongs to (`geometry` or `texture`).
*   **Loss Weights:** Importance of VGG vs Spectral vs Gram loss for each parameter.
*   **Learning Rate Multiplier:** Per-parameter learning rate scaling.

## Current Status (Phase 3.7)
The engine is functional and capable of recovering key parameters like `blur_sigma` and `bg_noise_std`. We are currently tuning the loss weights to improve the stability of `focus_z` optimization and handling the ambiguity between sensor blur and optical defocus.
