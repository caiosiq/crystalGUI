# Plan: Fixing the "Runaway Focus" & Visual Mismatch

## The Problem
1.  **Focus Divergence:** `optics.focus_z` converges to `-31.14` instead of `0.0`.
2.  **Visual Discrepancy:** The optimized image is "massively different" from the Ground Truth (GT).
    *   **GT:** Clear rods, defined edges, higher contrast.
    *   **Result:** Faint, low-contrast, washed-out features.
3.  **Interpretation:** The optimizer has found a "cheating" solution. It minimizes the loss (likely VGG/Spectral) by producing a generic, blurry, low-contrast image that statistically resembles the target but lacks physical correctness. The "Loss Blindness" is real.

## Why is this happening?
1.  **The "Pinhole Mismatch" (Confirmed):**
    *   GT was likely generated with `aperture=0.0` (Infinite DoF).
    *   Optimizer is forced to use `aperture=0.2`.
    *   Result: The optimizer moves `focus_z` to a depth where *nothing* exists (e.g., -31) so that *everything* is blurred, trying to match the `blur_sigma` of the GT (or lack thereof).
2.  **The "Gray Goo" Local Minimum:**
    *   If the optimizer can't match the sharp edges of the rods (due to seed mismatch or geometric misalignment), the safest way to minimize L1/L2/VGG loss is to **reduce contrast**.
    *   A faint, gray image has a lower pixel error against a sharp black-and-white image than a sharp image with rods in the wrong places (double penalty).
    *   Our `histogram` loss weight might be too low to prevent this contrast collapse.

## The Solution: Phase 3.7.1 - Physical Alignment & Contrast Enforcement

### 1. Fix Ground Truth Physics (Immediate Action)
We must ensure the GT is generated with the **exact same physical constraints** as the optimizer.
*   **Action:** In `validate_engine.py`, explicitly set `optics.aperture = 0.2` (and `blur_sigma = 0.0` initially) for the GT generation.
*   **Goal:** Make `focus_z` physically meaningful in the GT. If the GT has depth blur, the optimizer can find the focus plane.

### 2. Boost Contrast & Texture Enforcement
We need to force the optimizer to match the *intensity distribution* of the GT, preventing the "faint gray" solution.
*   **Action:** Drastically increase `histogram` loss weight in `optimization_rules.json`.
    *   Current: `10.0`
    *   Proposed: `100.0` or `500.0`.
*   **Why:** This forces the pixel value histogram (brightness/contrast) to match, even if the rods aren't perfectly aligned.

### 3. Diagnostic "Loss Monitor"
We need to see which loss component is failing.
*   **Action:** Update `CalibrationEngine` to log all individual loss terms (L1, VGG, Spectral, Gram, Histogram) separately.
*   **Hypothesis:** We will see `histogram` loss staying high while `vgg` drops, confirming the "Gray Goo" theory.

### 4. Regularization (The "Leash")
If `focus_z` still wanders, we add a soft constraint.
*   **Action:** Add a `regularization` term to the total loss.
    *   `Loss += lambda * (focus_z - 0.0)^2`
    *   This keeps focus near 0 unless the image data *strongly* pulls it away.

## Execution Steps

1.  **Modify `validate_engine.py`:** Set `GT_CONFIG.optics.aperture = 0.2` and `GT_CONFIG.sensor.blur_sigma = 0.5`.
2.  **Update `CalibrationEngine.py`:** Add detailed loss monitoring/logging.
3.  **Run Validation:** Check if `focus_z` behaves better.
4.  **If visual mismatch persists:** Tune `optimization_rules.json` (boost Histogram loss).
