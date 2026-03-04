# Differentiable Domain Calibration - Roadmap

## Phase 1: The Differentiable Core (OSOG Refactoring)
*Goal: Ensure gradients can flow through the rendering pipeline without breaking.*

- [x] **1.1. Parameter Registry**
    *   Create `DiffOSOG(nn.Module)` wrapper.
    *   Implement `register_active_params()` to convert selected config values into `nn.Parameter`.
    *   **Test:** Verified `model.parameters()` returns only the selected subset and gradients flow through `blur_sigma` and `distractor_opacity`.
    *   **Note:** Patched `SensorHeadTorch` and `Pipeline` to support differentiable rendering (removed `no_grad` and fixed tensor casting).

- [x] **1.2. Soft Mask Implementation (Landmine A Fix)**
    *   Modify `GeometryShader` to support a `soft_edge_mode=True` flag.
    *   Replace hard boolean masks (`dist < radius`) with `torch.sigmoid((radius - dist) * temperature)`.
    *   **Test:** `diff_calibration/tests/test_gradient_flow_geometry.py` - Confirmed gradients flow for `L`, `W`, `CX` (Rod) and `D` (Sphere).
    *   **Note:** Implemented `soft_clamp_zero` and `soft_clamp_unit` helpers using `softplus` and `sigmoid`. Forced Profile-based approximation for Rods/Boxes in soft mode to avoid zero-gradient issues with ray tracing parallel to view.

- [x] **1.3. Reparameterization of Distributions**
    *   Rewrite `generate_distribution()` to use the Reparameterization Trick ($L = \mu + \sigma \cdot \epsilon$).
    *   Ensure `L_mean`, `L_std`, `W_mean` are connected to the computational graph.
    *   **Test:** `diff_calibration/tests/test_gradient_flow_distributions.py` - Confirmed `rod_length_max.grad` is ~8.7, proving gradients flow from image loss back to distribution bounds.
    *   **Note:** Updated `DiffOSOG` to support nested attributes (`physics.rod_specs.length_range`) and verified `rand_uniform` supports differentiability.

- [x] **1.4. Soft Clamping (Gradient Preservation)**
    *   *Problem:* `torch.clamp(0, 255)` kills gradients for over-exposed pixels.
    *   *Fix:* Implement `soft_clamp(x, min, max)` using Tanh or leaky ReLU logic near boundaries.
    *   *Action:* Replace hard clamps in `optical_engine.py` with soft alternatives during calibration mode.
    *   **Test:** `diff_calibration/tests/test_soft_clamp.py` - Verified non-zero gradients for out-of-bound values using Tanh soft clamp.

---

## Phase 2: The Loss Engine
*Goal: Build robust metrics that ignore position but capture style.*

- [x] **2.1. VGG-19 Perceptual Loss**
    *   Implement `PerceptualLoss(nn.Module)`.
    *   Load pre-trained VGG-19 (ImageNet).
    *   Extract features from layers `relu1_1`, `relu2_1`, `relu3_1`, `relu4_1`.
    *   Compute Gram Matrices ($G = F \cdot F^T$).
    *   **Test:** Compare two random noise images vs. two shifted versions of the same image.

- [x] **2.2. Fourier / Spectral Loss (Landmine C Fix)**
    *   Implement `SpectralLoss(nn.Module)`.
    *   Use `torch.fft.fft2` and `torch.abs()` to get amplitude spectrums.
    *   Compute Log-MSE of spectrums.
    *   **Test:** Optimize a blurred image to match a sharp target (should recover `blur_sigma`).

- [x] **2.3. Patch-Based Sampling (Landmine B Fix)**
    *   Implement `RandomCropLoss` wrapper.
    *   Takes full images $(1024^2)$, crops $N \times 256^2$ patches, computes loss on patches.
    *   **Test:** Verify VRAM usage stays flat regardless of canvas size.

- [x] **2.4. Dynamic Loss Weighting**
    *   *Problem:* VGG Loss (~0.05) vs Spectral Loss (~1500.0). Optimizer ignores VGG.
    *   *Fix:* Implement automatic scaling or explicit $\lambda$ weights (e.g., `1.0` vs `1e-4`).
    *   *Action:* Add `LossBalancer` class that normalizes gradients or scales losses to be O(1).


---

## Phase 3: The Calibration Loop (Standalone)
*Goal: A working script that tunes parameters.*

- [x] **3.1. The Optimizer Script (`calibrate.py`)**
    *   Load Target Image.
    *   Initialize `DiffOSOG`.
    *   **Parameter Groups:** Assign distinct Learning Rates for Geometry (High LR, e.g., 1.0) vs. Optics (Low LR, e.g., 0.01).
    *   Loop: Forward -> Loss -> Backward -> Step.
    *   Log loss curves and parameter values.
    *   Save `calibrated_config.yaml`.
    *   **Feature:** Implemented `LossBalancer` to handle VGG/Spectral scale diffs.
    *   **Feature:** Added `RandomCropLoss` for low-VRAM optimization.

- [x] **3.2. Validation & Convergence Tests**
    *   **Test A (Sanity):** "Twin Study" - Generate a target image with known params (e.g., `blur=2.0`), then initialize optimizer with `blur=5.0`. Does it converge back to 2.0?
    *   **Test B (Realism):** Tune `noise_scale` and `shadow_gain` against a real brightfield image.
    *   **Implemented:** `diff_calibration/tests/validate_convergence.py` generates GT target, initializes optimizer far away, and verifies convergence of `blur` and `noise`.

---

## Phase 4: Integration & UI (The "Auto-Tune" Experience)
*Goal: A real-time, interactive calibration tool directly in the Playground.*

- [ ] **4.1. Backend Streaming API**
    *   Create `/api/calibrate_stream` endpoint (WebSocket or Server-Sent Events).
    *   **Inputs:** Reference Image (Base64), `active_params` list, `max_steps`.
    *   **Outputs:** Stream of JSON events containing:
        *   `current_image`: Base64 preview of the optimization state (every N steps).
        *   `loss`: Current loss value.
        *   `params`: Current values of the tuning parameters.

- [ ] **4.2. Frontend "Calibration Mode"**
    *   **Side-by-Side View:** Split screen showing "Real Target" vs. "Optimizing Simulation".
    *   **Parameter Selection:** Add checkboxes next to relevant sliders (e.g., "Tune Focus", "Tune Noise").
    *   **Controls:** "Upload Target", "Start Optimization", "Stop".
    *   **Live Updates:** The simulation view should update in real-time as the backend optimizer runs.

- [ ] **4.3. User Workflow Polish**
    *   **Preset Profiles:** One-click setups (e.g., "Tune Focus Only", "Tune Material Texture").
    *   **Safety Rails:** Warning if user selects too many parameters or incompatible ones (e.g., integer counts).
    *   **Result Application:** "Apply Calibrated Parameters" button to save the result to the main simulation config.
