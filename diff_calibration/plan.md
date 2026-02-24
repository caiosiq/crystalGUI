# Differentiable Domain Calibration Plan

## 1. The Concept
**Goal:** Automatically tune selected OSOG simulation parameters ($\theta$) so that the generated synthetic images ($I_{syn}$) statistically match a target real microscope image ($I_{real}$).

**Core Idea:** Instead of "Inverse Rendering" (recreating a specific scene geometry), we perform **"Domain Calibration"** (matching the *style* and *statistics* of the microscope).

**Mathematical Formulation:**
$$ \theta^* = \arg\min_{\theta \in \Theta_{active}} \mathcal{L}( \Phi(OSOG(\theta)), \Phi(I_{real}) ) $$
Where $\Phi$ is a feature extractor (e.g., VGG-19) that captures texture/style, $\mathcal{L}$ is a distribution-matching loss, and $\Theta_{active}$ is the **user-selected subset** of parameters to optimize.

---

## 2. Feasibility Analysis

### A. The Good (Differentiable Components)
OSOG is built on PyTorch, meaning most operations are naturally differentiable:
*   **Tensor Operations:** Blur, Noise injection, Color shifts, Tensor stamping (atomic adds).
*   **Physics Math:** Fresnel equations, Beer-Lambert law, and Geometry projections (sin/cos/matmul) allow gradients to flow from pixels back to parameters like `refractive_index` or `roughness`.

### B. The Challenge (Non-Differentiable Steps)
Standard Monte-Carlo rendering has steps that break the gradient chain (non-differentiable):
1.  **Discrete Sampling:** Deciding *how many* particles to spawn (Integer).
2.  **Hard Decisions:** `if shape == 'rod': ...` (Categorical).
3.  **Stochastic Sampling:** `random.uniform(min, max)` creates a number, but the gradient doesn't flow back to `min` or `max` unless reparameterized.

### C. The Solution: Reparameterization Trick
To make distribution ranges learnable, we must rewrite sampling logic:
*   **Old (Blocking):** `L = random.uniform(L_min, L_max)`
*   **New (Flowing):** 
    $$ \epsilon \sim \mathcal{U}(0, 1) \quad (\text{fixed noise}) $$
    $$ L = L_{min} + (L_{max} - L_{min}) \cdot \epsilon $$
    Now PyTorch can calculate $\frac{\partial L}{\partial L_{min}}$ and $\frac{\partial L}{\partial L_{max}}$.

---

## 3. Target Parameters for Optimization

We categorize parameters by optimization strategy. **We strictly exclude integer/discrete parameters from Gradient Descent.**

### Group 1: The "Low Hanging Fruit" (Sensor & Post-Process)
*Continuous tensor ops. High impact, easy gradients.*
*   **Optics:** `blur_sigma` (Focus), `chromatic_aberration_strength`.
*   **Sensor:** `noise_sigma` (Gain), `fouling_opacity` (Lens Dirt), `background_soup_density`.
*   **Lighting:** `shadow_gain`, `light_direction` (for Brightfield/Blaze).

### Group 2: Material Properties (Physics)
*Requires backprop through the `render_batch` loop.*
*   **Refractive Index:** `delta` (Contrast).
*   **Surface:** `roughness_amplitude` (Texture).
*   **Absorption:** `absorption_scale` (Darkness).

### Group 3: Geometry Distributions (Reparameterized)
*Continuous distribution parameters.*
*   **Size:** `L_mean`, `L_std`, `W_mean`.
*   **Shape:** `aspect_ratio_min`, `aspect_ratio_max`.

### Group 4: Manual / Fixed Parameters (NO Gradient Descent)
*These must be set manually by the user or optimized via Grid Search if absolutely necessary.*
*   **Counts:** `particle_count`, `debris_count`.
*   **Modes:** `shape_id` (Rod vs Cube), `optics_mode` (DIC vs Brightfield).
*   **Resolution:** `canvas_width`, `canvas_height`.

---

## 4. Loss Functions (The "Trap" vs "Solution")

### The Trap: Pixel-Wise MSE
Comparison of raw pixels ($|I_{syn} - I_{real}|^2$) fails because synthetic particles will never align perfectly with real particles. Gradients will be chaotic.

### The Solution: Distributional / Style Loss
We need a loss that says "These two images have the same *texture* and *blob size*", ignoring position.

1.  **Gram Matrix Loss (Style Loss):**
    *   Pass both images through VGG-19.
    *   Compute Gram Matrix $G = F \cdot F^T$ at multiple layers.
    *   Minimize MSE between Gram matrices.
    *   *Pros:* Excellent for capturing "Is it noisy?", "Is it sharp?", "Is the lighting directional?".
    
2.  **Sliced Wasserstein Distance (SWD):**
    *   Project pixel distributions onto random vectors.
    *   Compare 1D histograms.
    *   *Pros:* Good for matching the distribution of colors and intensities.

3.  **Fourier / Spectral Loss (Frequency Domain):**
    *   Transform both images via `torch.fft.fft2`.
    *   Compare Amplitude Spectrums ($|FFT(I)|$).
    *   *Pros:* Dirt-cheap computationally. Forces optimizer to match Blur and Noise Scale perfectly. Crucial for optical realism.

---

## 5. Integration Plan

### Phase 1: The "Selective Differentiable Wrapper"
Create a `nn.Module` wrapper that supports **Parameter Masking**:
```python
class DiffOSOG(nn.Module):
    def __init__(self, config, active_params: List[str]):
        super().__init__()
        # 1. Register all potential params
        self._register_param('blur_sigma', config.blur_sigma)
        self._register_param('noise_scale', config.noise_scale)
        
        # 2. Freeze non-selected parameters
        for name, param in self.named_parameters():
            if name in active_params:
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    def forward(self):
        # Run pipeline using self.params
        return generated_image
```

### Phase 2: The Calibration Loop
A standalone script (`calibrate.py`) that:
1.  Loads a target `real_image.png`.
2.  Initializes `DiffOSOG` with a specific list of `active_params`.
3.  **Patch-Based Training:**
    *   Instead of optimizing full 1024x1024 images (VRAM Explosion), take $N$ random $256 \times 256$ crops.
    *   Calculate Loss on these crops.
    *   Significantly reduces memory usage and improves local texture matching.
4.  Runs Adam Optimizer only on active parameters.
5.  Outputs the calibrated values.

### Phase 3: UI Integration (Real-Time Side-by-Side Tuning)
*   **Split-Screen Interface:** 
    *   Left Panel: "Real Target Image" (User Upload).
    *   Right Panel: "Optimizing Simulation" (Live Updates).
*   **Parameter Selection:** 
    *   Add "Optimize this?" checkboxes next to each slider in the Playground.
    *   User selects a subset (e.g., Blur, Noise, Contrast).
*   **Streaming Optimization:**
    *   User clicks "Start Auto-Tune".
    *   Backend runs the optimizer and streams intermediate images (every ~10 steps) back to the UI via WebSocket/SSE.
    *   User sees the synthetic image slowly morphing to match the real one in real-time.

---

## 6. Technical Challenges & Solutions

1.  **Landmine A: The "Hard Edge" Gradient Problem**
    *   *Problem:* Hard masks (0 or 1) have zero gradient everywhere except the edge, where it's infinite (Dirac Delta). Optimizer learns nothing.
    *   *Solution:* **Soft Masks**. Use `torch.sigmoid((radius - dist) * temperature)` in the Geometry Shader. This creates a smooth slope at edges, allowing gradients to flow back to size parameters ($L$, $W$).

2.  **Landmine B: VRAM Explosion**
    *   *Problem:* Backpropagating 1024x1024 images through VGG-19 causes OOM.
    *   *Solution:* **Patch-Based Training**. Crop small random patches (e.g., 256x256) during the forward pass and compute loss on those. Reduces VRAM usage by ~90%.

3.  **Landmine C: Missing Frequency Information**
    *   *Problem:* VGG captures texture but misses exact blur radii or noise frequencies.
    *   *Solution:* **Fourier Loss**. Add an explicit FFT-based loss term to force matching of the power spectrum.

4.  **The "Soup" Paradox**
    *   Optimizing the background soup vs. foreground particles. The loss function needs to balance them.
