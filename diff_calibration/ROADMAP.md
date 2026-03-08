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

## Phase 3.5: The Parameter Optimization Engine (The "Brain")
*Goal: A robust, automated "Manager" that abstracts optimization complexity from the user. The user just selects parameters; the Engine handles the strategy.*

- [x] **3.5.1. The Grand Unification (DiffOSOG V2)**
    *   *Goal: Map every single optimizeable parameter in OSOG to the Differentiable Wrapper.*
    *   **Audit:** Check `osog/config.py` vs `diff_wrapper.py`.
    *   **Cleanup:** Remove phantom parameters like `optics.blur_sigma` (it doesn't exist in `OpticsConfig`; only `sensor.blur_sigma` does).
    *   **Clarification:** Explicitly support both `optics.noise_scale` (Shot Noise, object-only) and `sensor.bg_noise_std` (Read Noise, global).
    *   **Expansion:** Add support for:
        *   **Physics:** `birefringence`, `rod_len_px_lo_hi` (float ranges), `rod_aspect_lo_hi` (float ranges).
        *   **Optics:** `lighting_angle`, `light_direction` (3D vector), `focus_z`, `aperture` (DoF).
        *   **Sensor:** `vignette`, `fouling_opacity`, `chromatic_aberration`, `diffraction_spikes`.
    *   **Note on Discrete Parameters:** Integers (`n_rods`, `count_range`) cannot be optimized via Gradient Descent. We will only optimize *Continuous* (Float) parameters.
    *   **Note on Ranges:** Parameters like `rod_len` are defined as `(min, max)` tuples. The optimizer will optimize the *mean* or *bounds* of these distributions (e.g., shifting the entire range).

- [x] **3.5.2. The "Parameter Rules" Knowledge Base (`optimization_rules.json`)**
    *   *Goal: Externalize the "Optimization Intelligence" into a config file.*
    *   Define a JSON schema that describes how to treat each parameter.
    *   **Structure per Parameter:**
        *   `bounds`: Physical Min/Max (e.g., `[0.0, 1.0]`).
        *   `scale`: `linear` or `log` (for parameters spanning orders of magnitude like `noise`).
        *   `stage`: `geometry`, `texture`, or `fine_tune`.
        *   `loss_weights`: `{"vgg": 1.0, "spectral": 0.0}`.
        *   `lr_mult`: Learning rate multiplier (e.g., `0.1` for sensitive params).
    *   **Benefit:** Allows tweaking the optimizer's strategy without touching code.

- [x] **3.5.3. The "Parameter Manager" (Bounded Latent Space)**
    *   **Concept:** A dedicated class (`ParameterManager`) that decouples the Optimizer from the Physical Model.
    *   **Latent Mapping:** Optimizer works in an unbounded latent space (Gaussian).
    *   **Sigmoid Bounding:** Latent values are mapped to [0, 1] via Sigmoid, then linearly scaled to strict Physical Min/Max bounds.
    *   **Universal Normalization:** All gradients effectively operate in the [0, 1] normalized space, allowing a single robust Learning Rate (e.g., 0.1) to work for both `refractive_index` (range 1.3-1.6) and `n_rods` (range 10-1000).

- [x] **3.5.4. Auto-Curriculum Scheduler (Hidden Complexity)**
    *   *Problem:* Optimizing Noise and Blur simultaneously causes crosstalk (local minima).
    *   *Solution:* An automated scheduler that generates a multi-stage plan based on the *user-selected* parameters.
    *   **Dependency Graph:** Define rules like "Texture relies on Geometry".
    *   **Execution:**
        *   *Stage 1 (Geometry):* Activate params tagged `stage="geometry"`. Freeze others.
        *   *Stage 2 (Texture):* Activate params tagged `stage="texture"`.
        *   *Stage 3 (Fine-tuning):* Unfreeze all.
    *   **User Experience:** User sees a single progress bar; the engine handles the freezing/unfreezing internally.

- [x] **3.5.5. Semantic Loss Routing**
    *   *Concept:* Different parameters respond to different loss functions.
    *   **Loss Router:** Automatically adjusts loss weights based on active parameters.
    *   **Rules:**
        *   If optimizing `Geometry` (rods, shape) -> Boost `VGG_Content` and `Spectral_Phase`.
        *   If optimizing `Texture` (noise, blur) -> Boost `GramMatrix` (Style) and `HistogramLoss`.
    *   **Implementation:** A `LossManager` that updates weights dynamically per stage.

- [x] **3.5.6. Robustness Features (Gradient & Bounds)**
    *   **Gradient Clipping:** `torch.nn.utils.clip_grad_norm_` to prevent explosion during stage transitions.
    *   **NaN Protection:** Automatic rollback if a step produces NaNs.
    *   **History Tracking:** Keep a "Best Known State" snapshot to restore if the optimizer diverges.

- [x] **3.5.7. The "Seed Strategy" Config**
    *   *Concept:* Different parameters require different randomness behaviors.
    *   **Rules:**
        *   `stage="geometry"` (Blur/Focus): **Lock Seed**. Optimizer must see deterministic structure to calculate gradients.
        *   `stage="texture"` (Noise/Style): **Unlock Seed**. Use Gradient Accumulation over $N$ different rod layouts to learn the global statistical profile.
    *   **Implementation:** Added `seed_mode` ("locked" or "random") to the `optimization_rules.json`. The AutoScheduler will enforce this by checking the active parameters' requirements.

- [x] **3.5.8. Parameter-Space Velocity Trigger**
    *   *Problem:* Loss curves in stochastic rendering are too noisy to trust for convergence detection (high noise floor).
    *   *Solution:* Monitor the **Latent Parameters** directly, not the loss.
    *   **Metric:** Track the sliding window variance (or velocity) of parameters in the normalized [0,1] space.
    *   **Trigger:** If parameters are "vibrating in place" (high loss variance but low parameter displacement) or velocity drops near zero -> **Stage Complete**.
    *   **Benefit:** Ignores rendering noise; detects true convergence of the variables.

- [x] **3.5.9. The "Texture" Loss Functions**
    *   *Goal: Enable optimization of stochastic noise/texture.*
    *   **Gram Matrix Loss:** Extract style features from VGG (ignore position, match correlation).
    *   **Histogram Loss:** Match pixel intensity distribution (for noise levels).
    *   **Implementation:** Add `diff_calibration/loss/texture.py`.

- [x] **3.6. The Unified Engine Integration (`calibration_engine.py`)**
    *   *Goal: Tie everything together.*
    *   **Orchestration:** Loop through Stages (Scheduler).
    *   **Optimization:** Initialize params (Manager), calculate Loss (Router + Losses), Step (Optimizer + Guard).
    *   **Logging:** Track progress via callback.
    *   **Output:** Return optimized config values.

- [x] **3.7. Refactoring & Validation**
    *   **Refactor:** Moved `diff_calibration` code into `src` structure. Moved losses to `src/loss`.
    *   **Validate:** Created `tests/validate_engine.py` to test the full engine against GT parameters.

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
