CrystalGUI: FastAPI GUI for Crystal Imaging and Synthetic Dataset Generation

Overview

CrystalGUI is a FastAPI-based web application and toolkit for:
- Uploading, inspecting, and preprocessing microscopy images
- Running model inference and visualizing per-image statistics
- Live frame streaming with WebSocket updates
- Generating realistic synthetic phase‑contrast crystal images and labels

The synthetic generator is designed to mimic DIC-like (Differential Interference Contrast) imagery of slender, rod‑like crystals. It supports interactive preview in the GUI and reproducible, batched dataset generation on Slurm clusters.


Quick Start

1) Install dependencies
- Python 3.10+
- pip install -r requirements.txt

2) Run locally
- uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
  Then open http://localhost:8000/

3) Run on Slurm (interactive)
- ./start_gui_interactive.sh -p <partition> -g <gres> -c <cpus> -m <mem> -t <hh:mm:ss> -w <workers> -P <port> -e </path/to/venv>
  The script will request an interactive allocation, activate your environment, and launch uvicorn. It prints an SSH port‑forwarding command you can run from your laptop to reach the compute node.

4) Run on Slurm (batch)
- sbatch run_gui.slurm
  This starts a gunicorn server with multiple uvicorn workers on port 8000.


Repository Layout

- app/
  - main.py: FastAPI application, CORS setup, static mounts, and routes. Includes endpoints for uploads, preprocessing, inference, live streaming, and synthetic image config/presets.
  - static/
    - css/styles.css
    - js/app.js
    - js/synth.js: Frontend logic for the Synthesis tab. On DOMContentLoaded, it fetches /synth_default_config, lets you edit parameters via a form, previews single images (/synth_preview), regenerates rows (/synth_preview_bulk), and starts batch generation (/synth_batch). It also supports presets via endpoints listed below.
- data_generator/
  - synth.py: OpenCV/Pillow‑based synthetic image renderer. Implements background formation, the DIC‑style rod shader, ghost rods, debris, optional fused crystals, scale legend, RNG/seeding, t‑parameter scheduling, and oriented bounding box recording. Exposes generate_image(), default_config(), sample_lambda(), lambda_to_t(), and params_for_t().
  - batch_job.py: CLI for batched dataset generation. Produces images plus DOTA quadrilateral labels and YOLO‑OBB labels.
- data/: Project data folders (uploads, results, preprocessed, etc.); created at runtime.
- models/: Place inference model files here; selected via /load_model or /select_model_folder.
- start_gui_interactive.sh: Interactive Slurm launcher that sets up the environment and starts uvicorn app.main:app.
- run_gui.slurm: Batch Slurm script that starts gunicorn with uvicorn workers.
- requirements.txt: Python dependencies.
- tests/: Project tests (if present).


Key FastAPI Endpoints (high level)

Uploads, preprocessing, and inference:
- /: Home page (renders index.html) and lists uploaded images
- /outputs_upload_folder: Upload a folder (multipart) preserving relative paths
- /outputs_inspect_dataset: Scan a dataset folder; return counts (readable, zero‑size, unreadable)
- /upload: Upload a single image
- /preprocess: Apply a selected preprocessing operation and save
- /preproc_preview: Return a base64 preview of a preprocessing pipeline
- /save_preprocessed: Save the pipeline result to disk
- /inference, /inference_compare, /inference_compare_preproc: Run model inference, compute stats/overlays, return results
- /available_models, /load_model, /select_model_folder: List and set the active inference model
- /system_info: Report GPU availability, model info
- /ws/live, /stream_frame, /live_stats: WebSocket live feed utilities

Synthetic image config, presets, and generation:
- /synth_default_config: Return the default configuration for synthetic generation (either a saved “standard” preset or library defaults)
- /synth_save_standard: Persist a provided configuration as the standard default
- /synth_save_preset: Save a named configuration
- /synth_presets: List available presets
- /synth_get_preset: Fetch a preset by name
- /synth_preview: Render a single image preview for the current form config
- /synth_preview_bulk: Regenerate preview rows in bulk
- /synth_batch: Trigger batched dataset generation


Synthetic Image Generation: Model and Implementation

At a glance
- The generator produces an RGB canvas as a composition of background terms plus additive contributions from rod‑like crystals, optional ghost rods, and sparse debris.
- A per‑image stage parameter t in [0,1] controls density and geometry ranges: number of rods, length, aspect ratio, and intensity bounds. t is derived from a log‑uniform physical knob λ via lambda_to_t().
- Each rod is rendered by a DIC‑style shader that creates a tapered body fill, an odd bright/dark edge pair across width, and a faint outer halo.
- Oriented bounding boxes (OBBs) are recorded for every rod and can be saved in DOTA and YOLO‑OBB files.

How the math maps to code

Background formation (synth.py)
- _apply_background(cfg, h, w, ...):
  - Base gray: bg_gray_range sampled per pixel.
  - Directional gradient (tilt): a linear ramp along a random direction (tilt_dir_deg) with peak‑to‑peak amplitude (tilt_ptp) and optional center shift (tilt_center).
  - Low‑frequency illumination: blurred Gaussian noise with amplitude illum_ampl and blur scale illum_sigma.
  - Vignette: multiplicative center darkening controlled by vignette_strength.
  - DIC‑style relief field: a smooth random height is blurred (relief_field_sigma_px), differentiated via Sobel, projected along relief_field_dir_deg, optionally extra‑blurred, and added with relief_field_gain.
  - Optional background noise via bg_noise_std.

Per‑rod shader and local frame
- _draw_phase_contrast_rod(...): Implements the object model in rod‑local coordinates.
  - Local coordinates (u along length, v across width) computed from oriented box geometry (cx, cy, L, W, ang_deg).
  - Tapered envelope: width w(u) shrinks towards the ends with taper_strength and taper_power; min_width_ratio enforces a lower bound. A smooth cap s(u)=_smooth_cap limits support near the ends.
  - Body fill: alpha_fill = exp(-0.5*(v/sigma_v)^2) * s(u). A small base_delta jitter is applied; the body adds a soft bright or dark delta to the patch luminance.
  - DIC edge pair: An odd response pc(v) centered with an offset shadow_offset_px, width shadow_width_mult·sigma_v, gain shadow_gain, and bias shadow_bias to skew bright/dark lobes. pc is multiplied by the longitudinal cap and optionally jittered (edge_jit_amp) or polarity‑flipped.
  - Halo: A faint outer glow from the blurred support mask (rod_halo_sigma) scaled by rod_halo_gain.
  - Composition: The shaded layer is added (or mixed multiplicatively if mult_mix>0) into the canvas patch.

Ghost rods and shape jitter
- If ghost_enable is true, a second layer of weaker‑contrast rods is rendered with additional distortions:
  - Width/edge/offset jitters (ghost_width_jit_amp, ghost_edge_jit_amp, ghost_offset_jit_amp)
  - Curvature (ghost_curve_kappa_range)
  - Local shape warp modes: wavy (_sin_wobble), kink (_kink), noisy (_noisy_wobble), or straight
  - Optional blur/noise applied after the ghost layer (ghost_blur_sigma, ghost_noise_std)

Debris
- Small discs or short dashes are added at random locations. Each debris element updates the gray channel locally by debris_int_delta with size from debris_size_px and dash probability debris_dash_prob.

Fused crystals
- With probability p_fused (linearly scheduled between fused_p0 and fused_p1 via t), a main rod sprouts 2–5 “arms” around its angle, each arm drawn as additional rods with slightly perturbed (L, W, angle, and center).

Stage scheduling (λ → t)
- sample_lambda(rng, cfg): Draws λ log‑uniformly from stage_lambda_range.
- lambda_to_t(λ): Maps to t∈[0,1] using a log scale, clamped to [0,1].
- params_for_t(cfg, t): Interpolates ranges for n_rods, rod_len_px, rod_aspect, and rod_delta (intensity bounds). Supports 2‑tuple ranges (lo, hi) and 3‑tuple ranges (lo, hi@t0, hi@t1) for flexible upper bound scheduling.

Labels: Oriented bounding boxes (OBBs)
- During rendering, each rod’s oriented rectangle is recorded (center cx,cy; L,W; angle_deg; corners).
- batch_job.py writes two label formats per image:
  - DOTA quadrilaterals (labels_dota/*.txt): x1 y1 x2 y2 x3 y3 x4 y4 class difficulty
  - YOLO‑OBB (labels_yolo_obb/*.txt): class_id cx cy w h angle(rad), all normalized by image width/height
  - classes.txt contains the single “Crystal” class by default.


Configuration Reference (SynthConfig)

Canvas
- width, height

Background
- bg_gray_range: base grayscale range
- vignette_strength
- tilt_enable, tilt_dir_deg, tilt_ptp, tilt_center
- bg_noise_std
- illum_ampl, illum_sigma
- relief_field_enable, relief_field_sigma_px, relief_field_gain, relief_field_dir_deg, relief_field_extra_blur

Rod shading and geometry
- rods_enable
- taper_strength, taper_power, min_width_ratio
- cross_soft_sigma: softness across width
- rod_halo_sigma, rod_halo_gain
- rod_noise_std
- shadow_gain, shadow_width_mult, shadow_bias, shadow_offset_px

Global scheduling (modulated by t)
- n_rods_rng_lo_hi
- rod_len_px_lo_hi
- rod_aspect_lo_hi
- rod_delta_rng
- fused_enable, fused_p0, fused_p1
- stage_lambda_range

Ghost layer
- ghost_enable, ghost_fraction, ghost_gain_mult
- ghost_blur_sigma, ghost_noise_std, ghost_curvature
- ghost_width_jit_amp, ghost_edge_jit_amp, ghost_offset_jit_amp
- ghost_curve_kappa_range, ghost_ragged_p, ghost_ragged_corr, ghost_mult_mix

Debris
- debris_rate, debris_int_delta, debris_dash_prob, debris_size_px

Scale legend
- scalebar_enable, scalebar_prob
- scalebar_len_px, scalebar_thick_px, scalebar_margin_px
- scalebar_outline, scalebar_font_px, scalebar_white_jit, scalebar_units, scalebar_value_range, scalebar_ttf

Performance
- parallel_workers: optional integer to enable thread‑parallel rendering of rods


Programmatic Usage

Obtain defaults
- from crystalGUI.data_generator.synth import default_config
- cfg = default_config()  # dict

Single image
- from crystalGUI.data_generator.synth import generate_image, lambda_to_t
- img = generate_image(cfg, t=0.5, seed=123)
- cv2.imwrite("out.jpg", img)

Batch generation with labels
- python -m crystalGUI.data_generator.batch_job \
  --n-images 1000 \
  --out-dir data/synth_dataset \
  --config-file data/synth_config.json \
  --seed-base 42 \
  --index-offset 0

Output structure
- data/synth_dataset/
  - images/*.jpg
  - labels_dota/*.txt
  - labels_yolo_obb/*.txt
  - classes.txt

Reproducibility
- generate_image() accepts an integer seed. When omitted, SystemRandom seeds are used for both Python’s random and NumPy. The batch_job.py CLI sets a deterministic per‑image seed from (seed_base + index_offset + i), so labels and images are reproducible across shards.


Notes and Tips

- Geometry vs. photometry: The renderer operates in float luminance space, then clamps to [0,255] and broadcasts to BGR for display. Most terms are gray and applied identically to all channels.
- Tuning realism: Use relief_field_* and illum_* to add low‑frequency structure; adjust shadow_* to control edge sharpness, bias, and phase‑contrast look; add ghost_* jitter modes for clutter.
- Fused crystals: Increase fused_p1 to create multi‑arm structures resembling bundles; arms are drawn as individual rods and labeled separately unless you post‑process labels.
- Performance: For large n_rods, set parallel_workers to a CPU count (e.g., 8–16) to thread‑parallelize rod shading.
- UI integration: The Synthesis tab (app/static/js/synth.js) fetches /synth_default_config, lets you modify and save presets, previews with /synth_preview and /synth_preview_bulk, and kicks off batches via /synth_batch.


License and Attribution

This project includes open‑source dependencies listed in requirements.txt. The synthetic image formulation and parameterization follow a DIC‑style shading model implemented in data_generator/synth.py.