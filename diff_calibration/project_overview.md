# Project Overview: OSOG & DiffOSOG

This document summarizes the core components of the project we are developing: the **OSOG (Optical Simulation of Organic Growth)** engine and its differentiable counterpart, **DiffOSOG**.

## 1. OSOG (Optical Simulation of Organic Growth)
**OSOG** is a high-performance, physics-based rendering engine designed to simulate microscope images of crystal growth processes. It generates synthetic training data that is visually indistinguishable from real-world microscopy.

### Key Features
*   **Procedural Geometry:** Generates complex 3D crystal structures (rods, plates, spheres) with realistic distributions (length, width, orientation).
*   **Optical Physics:** Simulates light interaction with microscopic objects, including:
    *   **DIC (Differential Interference Contrast):** Simulates the "3D relief" effect using shear, gradients, and polarization.
    *   **Brightfield/Darkfield:** Simulates absorption, refraction (Fresnel equations), and scattering.
    *   **Depth of Field (DoF):** Simulates focal planes and bokeh using physically accurate Circle of Confusion (CoC) models.
*   **Sensor Simulation:** Adds camera artifacts like Gaussian noise, shot noise, chromatic aberration, spectral dispersion, and motion blur.
*   **Performance:** Optimized PyTorch implementation capable of generating high-resolution batches on GPU.

## 2. DiffOSOG (Differentiable OSOG)
**DiffOSOG** is a wrapper around OSOG that makes the entire rendering pipeline **differentiable**. This means we can compute gradients of the output image with respect to the input parameters (e.g., "How does changing the focus z-position change the pixel intensity?").

### Why is this powerful?
Instead of manually tuning hundreds of parameters to match a real microscope image (Calibration), we can use **Gradient Descent** to automatically find the parameters that generated a given target image. This is "Inverse Rendering."

### Key Features
*   **Differentiable Pipeline:** Every step (Geometry -> Optics -> Sensor) preserves gradients. Non-differentiable operations (like discrete sorting or random sampling) are handled via "Reparameterization Tricks" or "Straight-Through Estimators."
*   **Parameter Manager:** Maps physical parameters (e.g., `focus_z: -100 to 100`) to a normalized, unbounded "Latent Space" suitable for optimizers like Adam.
*   **Auto-Curriculum Scheduler:** Solves the optimization problem in stages to avoid local minima:
    1.  **Geometry Stage:** Locks seed, optimizes macro shapes (blur, focus, lighting).
    2.  **Texture Stage:** Randomizes seed (Monte Carlo), optimizes statistical textures (noise, grain).
    3.  **Fine-Tuning:** Polishes everything together.
*   **Loss Engineering:** Uses a combination of Perceptual Loss (VGG), Spectral Loss (FFT), and Texture Loss (Gram Matrices) to match human perception rather than just pixel values.

## Current Status (Phase 3.7)
We are currently debugging the **Auto-Calibration Engine**.
*   **Goal:** Input a real image -> Output the OSOG config that recreates it.
*   **Current Challenge:** The optimizer sometimes "cheats" by finding non-physical solutions (e.g., extreme focus values) that statistically minimize loss but look wrong. We are tuning the loss weights and regularization to enforce physical realism.
