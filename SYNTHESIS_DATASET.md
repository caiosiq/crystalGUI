# Synthetic Crystallization Dataset: Methodology and Effects

## Motivation
We synthesize rod-like crystals under DIC-style imaging, modeling growth stages and artifacts observed in real datasets to train and evaluate detection models.

## Image Formation Model
- Base luminance: `bg_gray_range` with optional tilt gradient and vignette.
- DIC relief field: Low-frequency random field `H` blurred by `relief_field_sigma_px`, directional gradient via Sobel with a preferred angle `relief_field_dir_deg`, scaled by `relief_field_gain`.
- Noise: Additive Gaussian background noise (`bg_noise_std`) and optional illumination field (`illum_*`).

## Rod Rendering
- Oriented rods parameterized by length `L`, width `W`, aspect `A = W/L`, and angle `θ`.
- Taper envelope along `u` (long axis) reduces cross-width near tips (`taper_strength`, `taper_power`, `min_width_ratio`).
- Phase-contrast edge pair: Bright/dark lobes across `v` (short axis) with `shadow_gain`, `shadow_width_mult`, `shadow_bias`, `shadow_offset_px`.
- Cross-width softness: `cross_soft_sigma` controls Gaussian falloff across `v`.
- Halo: A faint outer glow via blurred support (`rod_halo_sigma`, `rod_halo_gain`).

## Ghost Rods (Optical/processing artifacts)
- A fraction of rods (`ghost_fraction`) rendered with lower intensity (`ghost_delta_rng`), optional blur/noise.
- Distortion modes: `wavy` (composite sinusoids), `kink` (tanh bend), `noisy` (correlated noise). Jitters: `ghost_width_jit_amp`, `ghost_edge_jit_amp`, `ghost_offset_jit_amp`.
- Edge irregularity via `ghost_ragged_p`/`ghost_ragged_corr`; composite blend via `ghost_mult_mix`.
- Curvature modulation via `ghost_curve_kappa_range`.

## Fused Crystals
- At stage parameter `t∈[0,1]`, fused probability `p_fused = lerp(fused_p0, fused_p1, t)`.
- Arms: 2–5 rods around a main angle with spread; arm dimensions jittered and offset slightly.

## Debris
- Random small blobs and dashes at rate `debris_rate`. Intensities sampled from `debris_int_delta`.
- Dash probability `debris_dash_prob`; sizes from `debris_size_px`.

## Stage Sampling for Batches
- We sample a physical “stage λ” log-uniform in `[λ_min, λ_max]` via `sample_lambda`.
- Map to synthesis `t` using `t = (log10(λ) + 1)/2`, clamped to `[0,1]`.
- This yields early-to-late growth distributions reflective of real progression.

## Outputs
- Per-image overlays of OBBs (oriented bounding boxes) can be requested for visualization.
- Scalebar legend: random corner placement, randomized text, and outline for contrast.

## Reproducibility
- Preview endpoints return `seed_used`. High-res re-renders reuse this seed with larger canvas to reproduce the same composition.

## Configuration and Presets
- All parameters configurable in the GUI; presets are JSON files stored under `data/synth_presets`.
- Standard preset loads by default; named presets can be saved and loaded via the GUI.