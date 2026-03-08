# DiffOSOG: Differentiable Calibration Engine

**DiffOSOG** (Differentiable Optical Simulation of Organic Growth) is an advanced framework for **automatic parameter calibration** of the OSOG rendering engine. It uses differentiable rendering and gradient descent to "inverse render" microscope images, finding the optimal simulation parameters (focus, blur, noise, lighting) that match a target real-world image.

## Key Features

*   **Differentiable Pipeline:** Fully differentiable PyTorch implementation of the OSOG rendering pipeline (Geometry -> Optics -> Sensor).
*   **Automatic Curriculum Learning:** Solves the optimization problem in stages (Geometry -> Texture -> Fine-Tune) to avoid local minima.
*   **Semantic Loss Routing:** Uses a combination of Perceptual (VGG), Spectral (FFT), and Texture (Gram Matrix) losses to match human perception.
*   **Robust Optimization:** Handles bounded parameters, gradient clipping, and convergence detection.
*   **Parameter Manager:** Maps physical parameters (e.g., focus Z position) to normalized latent space for stable optimization.

## Directory Structure

*   `src/`: Core implementation files.
    *   `diff_wrapper.py`: Differentiable wrapper for OSOG.
    *   `calibration_engine.py`: Main orchestration logic.
    *   `parameter_manager.py`: Handles parameter injection and normalization.
    *   `auto_scheduler.py`: Manages the multi-stage optimization plan.
    *   `loss_manager.py`: Handles loss functions and weighting.
    *   `convergence_guard.py`: Detects when optimization has stalled.
    *   `robustness.py`: Handles gradient clipping and NaN checks.
*   `src/loss/`: Custom loss function implementations.
    *   `perceptual.py`: VGG-based perceptual loss.
    *   `spectral.py`: Fourier Transform based loss for sharpness matching.
    *   `texture.py`: Gram Matrix and Histogram losses for texture matching.
*   `tests/`: Validation and debug scripts.
    *   `validate_engine.py`: Full end-to-end test of the calibration loop.
    *   `debug_grads_v2.py`: Diagnostic script for checking gradient flow.
*   `optimization_rules.json`: Configuration file defining bounds, stages, and loss weights for each parameter.

## Usage

### Running Validation
To test the full calibration cycle against a synthetic ground truth:

```bash
python3 diff_calibration/tests/validate_engine.py
```

This will:
1.  Generate a "Ground Truth" image with known parameters.
2.  Initialize the engine with "bad" parameters.
3.  Run the optimization loop to recover the GT parameters.
4.  Output results (images, plots, CSV history) to `diff_calibration/validation_output/engine_test/`.

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
