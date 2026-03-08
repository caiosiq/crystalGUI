# Analysis: Why Some Gradients Are Tiny (Or Zero)

You have successfully debugged the "Disconnected Graph" error (where gradients were `None`), but now you are facing the "Vanishing Gradient" problem (where gradients are numerically zero or extremely small).

This document explains exactly why `optics.focus_z` and `sensor.bg_noise_std` appear to be doing nothing, based on a deep dive into the `osog` source code.

## 1. The Case of the Missing Focus (`optics.focus_z`)

**Symptom:** The gradient for `focus_z` is exactly zero (or just the dummy gradient value), and changing it has no effect on the image.

**The Cause: The "Pinhole Camera" Default**
In `osog/optics/sensor_torch.py`, the Depth of Field (DoF) effect is calculated using a "Circle of Confusion" (CoC):

```python
# osog/optics/sensor_torch.py

def apply_dof(self, img, depth_map, focus_z, aperture):
    # ...
    # CoC depends on the distance from focus plane AND the aperture
    coc = torch.abs(depth_map - focus_z) * aperture * 0.1 
    
    # ...
    # If CoC is small, we blend 0% blur.
    t1 = torch.clamp((coc - 0.5) / (3.0 - 0.5), 0.0, 1.0)
    out = img * (1.0 - t1) + img_med * t1 * ...
```

Crucially, check the default configuration in `osog/config.py`:

```python
@dataclass
class OpticsConfig:
    focus_z: float = 0.0
    aperture: float = 0.0  # <--- THE CULPRIT
```

The default `aperture` is `0.0`. This simulates a Pinhole Camera, which has **infinite depth of field**. Everything is perfectly sharp, regardless of the `focus_z` distance.

Mathematically:
1. `aperture = 0.0`
2. `coc = |depth - z| * 0.0 = 0.0`
3. `t1 = clamp((0.0 - 0.5)/2.5, 0, 1) = 0.0`
4. `out = img * 1.0 + blurred * 0.0 = img`

Since `out` is identical to `img` regardless of `focus_z`, the derivative `d(Loss)/d(focus_z)` is **exactly zero**.

**The Fix:**
You must ensure `optics.aperture` is non-zero during optimization.
1.  **Option A (Static):** Set a default aperture (e.g., `0.1`) in your initial config if you want to optimize focus.
2.  **Option B (Dynamic):** Add `optics.aperture` to your optimization plan (it is in `optimization_rules.json`, but check if it's actually enabled/active in the `AutoScheduler`).

---

## 2. The Case of the Invisible Noise (`sensor.bg_noise_std`)

**Symptom:** The gradient exists but is tiny (`~0.0004`), and the loss barely moves.

**The Cause: Scale Mismatch**
In `osog/optics/sensor_torch.py`:

```python
noise = bg_noise * torch.randn(1, h, w, ...)
img = img + noise
```

The image `img` has values in the range `[0, 255]` (floats).
Your optimizer initialized `bg_noise_std` to `0.5`.

*   **Visual Impact:** A standard deviation of `0.5` on a `0-255` scale is less than 1/500th of the dynamic range. It is virtually invisible to the human eye and barely registers in the loss function.
*   **Loss Function:** If you are using L1 Loss (`|pred - target|`), adding zero-mean noise averages out over the image. The loss only increases by the *magnitude* of the noise (`mean(|noise|) * weight`).
    *   For Gaussian noise with `std=0.5`, the mean absolute deviation is `sqrt(2/pi) * 0.5 ≈ 0.4`.
    *   If your image is `1024x768`, the total loss sum increases, but if you take the mean, it's a small constant shift.
*   **Gradient:** The gradient tries to push `bg_noise_std` to match the noise level of the target. If the target is clean (`std=0`), the gradient points down. If the target is noisy, it points up. But the *magnitude* of this gradient is proportional to the residual error, which is dominated by the massive geometric differences (rods being in the wrong place).

**The Fix:**
1.  **Boost Initialization:** Start `bg_noise_std` at a higher value (e.g., `5.0` or `10.0`) so it's "visible" to the optimizer.
2.  **Curriculum:** This confirms why your **Geometry -> Texture** separation is correct. The geometry loss (rods) is massive compared to the texture loss (noise). You *must* freeze geometry while tuning texture, or the noise gradient will be drowned out.
3.  **Loss Weight:** You already have a `loss_weights` entry for this. Ensure the weight for `histogram` or `gram` loss is high enough (e.g., `1000.0`) to make `0.5` noise matter.

---

## Summary of Action Plan

1.  **For Focus:** We need to modify `DiffOSOG` or the initialization logic to ensure `aperture >= 0.05` whenever `focus_z` is being optimized.
2.  **For Noise:** We should verify the initialization rules in `optimization_rules.json` and potentially increase the initial value or the loss weight.
