from __future__ import annotations

"""
Minimal, parameterized synthetic image generator (OpenCV-based) that approximates
the look of rod-like "crystals" with DIC-style shading. Designed for interactive
use via the GUI: fast single-image previews, configurable global ranges, and a
per-row scalar parameter t in [0,1] that modulates density/length/aspect ranges.

Note: This does not read or modify synth_speckles/*.py; it's a standalone analog
with fewer features, intended for real-time tweaking. You can expand parameters
later to match more of the original pipeline.
"""

from dataclasses import dataclass, asdict, replace
from typing import Tuple, Dict, Any, Callable, Optional, Sequence, List
import math, random
import numpy as np
import cv2
from pathlib import Path

# Optional Pillow import for TTF legend; gracefully degrade if unavailable
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


@dataclass
class SynthConfig:
    # canvas
    width: int = 1024
    height: int = 768

    # Background
    bg_gray_range: Tuple[int, int] = (90, 97)
    vignette_strength: float = 0.0
    tilt_enable: bool = True
    tilt_dir_deg: Tuple[float, float] = (-30.0, 30.0)
    tilt_ptp: Tuple[float, float] = (12.0, 15.0)
    tilt_center: Tuple[float, float] = (0.0, 0.0)
    bg_noise_std: float = 0.0
    illum_ampl: float = 0.0
    illum_sigma: float = 0.0

    # Rod model (simplified)
    rods_enable: bool = True
    taper_strength: float = 0.45
    taper_power: float = 1.0
    min_width_ratio: float = 0.35
    blur_sigma: float = 0.0
    cross_soft_sigma: float = 0.30
    rod_halo_sigma: float = 3.2
    rod_halo_gain: float = 0.4
    rod_noise_std: float = 0.0

    # Shadow properties (simplified DIC-like edge pair)
    shadow_gain: Tuple[float, float] = (6.0, 12.0)
    shadow_width_mult: Tuple[float, float] = (0.02, 0.05)
    shadow_bias: Tuple[float, float] = (0.05, 0.12)
    shadow_offset_px: Tuple[float, float] = (0.05, 0.25)

    # Global density/geometry ranges that t modulates
    n_rods_rng_lo_hi: Tuple[int, int] = (150, 600)
    rod_len_px_lo_hi: Tuple[float, float] = (30.0, 380.0)
    rod_aspect_lo_hi: Tuple[float, float] = (0.02, 0.30)

    # Rod intensity delta ranges (relative to base)
    rod_delta_rng: Tuple[float, float] = (-12.0, 0.0)
    ghost_delta_rng: Tuple[float, float] = (-3.0, 0.0)

    # Scalebar (basic)
    scalebar_enable: bool = True
    scalebar_prob: float = 0.5
    scalebar_len_px: Tuple[int, int] = (80, 240)
    scalebar_thick_px: Tuple[int, int] = (2, 12)
    scalebar_margin_px: int = 24
    scalebar_outline: bool = True
    scalebar_font_px: Tuple[int, int] = (70, 80)
    scalebar_white_jit: Tuple[int, int] = (245, 255)
    scalebar_units: Tuple[str, ...] = ("μm", "um", "nm", "mm")
    scalebar_value_range: Tuple[int, int] = (10, 99)
    scalebar_ttf: Optional[str] = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

    # Ghost rods (optional additional layer with weaker contrast)
    ghost_enable: bool = False
    ghost_fraction: float = 0.2  # fraction of n_rods to add as ghosts
    ghost_gain_mult: float = 0.5  # multiply shadow_gain range for ghosts
    ghost_blur_sigma: float = 0.0  # extra blur applied after ghost rods
    ghost_noise_std: float = 0.0   # additive Gaussian noise std after ghost layer
    ghost_curvature: float = 0.0   # curvature coefficient for ghost shading
    # Ghost distortion/jitter controls
    ghost_width_jit_amp: float = 0.25
    ghost_edge_jit_amp: float = 0.25
    ghost_offset_jit_amp: float = 0.39
    ghost_curve_kappa_range: Tuple[float, float] = (-0.125, 0.125)
    ghost_ragged_p: float = 0.90
    ghost_ragged_corr: float = 0.90
    ghost_mult_mix: float = 0.90

    # DIC relief field
    relief_field_enable: bool = True
    relief_field_sigma_px: Tuple[float, float] = (20.0, 20.0)
    relief_field_gain: Tuple[float, float] = (-0.5, 0.0)
    relief_field_dir_deg: Tuple[float, float] = (0.0, 0.0)
    relief_field_extra_blur: float = 0.0

    # Debris
    debris_rate: float = 0.0
    debris_int_delta: Tuple[float, float] = (-6.0, 6.0)
    debris_dash_prob: float = 0.15
    debris_size_px: Tuple[int, int] = (1, 3)

    # Fused crystals
    fused_enable: bool = True
    fused_p0: float = 0.0001
    fused_p1: float = 0.003

    # Stage lambda sampling range (for batch generation)
    stage_lambda_range: Tuple[float, float] = (0.1, 10.0)

    # Concurrency (optional): number of threads to render rods in parallel.
    # None or <=1 keeps sequential rendering.
    parallel_workers: Optional[int] = None


def default_config() -> Dict[str, Any]:
    return asdict(SynthConfig())


def _lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)


def params_for_t(cfg: SynthConfig, t: float) -> Dict[str, Any]:
    """
    Compute per-t parameters. Supports both 2-length ranges (lo, hi) and 3-length
    ranges (lo, hi@t0, hi@t1). For 2-length ranges, values vary linearly: lo + t*(hi-lo).
    For 3-length ranges, the upper bound hi itself varies with t: hi(t) = _lerp(hi@t0, hi@t1, t),
    then the value is _lerp(lo, hi(t), t). This preserves backward compatibility.
    """
    t = float(max(0.0, min(1.0, t)))

    def _lo_and_hi_t(range_val):
        # Accept Tuple/List of length 2 or 3
        if isinstance(range_val, (list, tuple)):
            if len(range_val) == 3:
                lo, hi0, hi1 = range_val
                lo = float(lo)
                hi_t = float(_lerp(float(hi0), float(hi1), t))
                return lo, max(lo, hi_t)
            elif len(range_val) == 2:
                lo, hi = range_val
                lo = float(lo); hi = float(hi)
                return lo, max(lo, hi)
        # Fallback: treat as fixed upper equal to lower
        try:
            lo = float(range_val)
        except Exception:
            lo = 0.0
        return lo, lo

    # n_rods: map between lo and hi(t) with t
    n_lo, n_hi_t = _lo_and_hi_t(cfg.n_rods_rng_lo_hi)
    n_rods = int(round(_lerp(n_lo, n_hi_t, t)))

    # rod length max at t
    L_lo, L_hi_t = _lo_and_hi_t(cfg.rod_len_px_lo_hi)
    len_hi = float(L_hi_t)

    # aspect ratio max at t
    ar_lo, ar_hi_t = _lo_and_hi_t(cfg.rod_aspect_lo_hi)
    ar_hi_val = float(ar_hi_t)

    # rod delta (intensity) bounds at t
    d_lo, d_hi_t = _lo_and_hi_t(cfg.rod_delta_rng)

    # Fused crystals probability derived from t
    p_fused = _lerp(cfg.fused_p0, cfg.fused_p1, t)

    return {
        "n_rods_min": int(round(n_lo)),
        "n_rods_max": int(round(n_hi_t)),
        "n_rods": int(n_rods),
        "rod_len_min": float(L_lo),
        "rod_len_max": float(len_hi),
        "rod_aspect_min": float(ar_lo),
        "rod_aspect_max": float(ar_hi_val),
        "rod_delta_min": float(d_lo),
        "rod_delta_max": float(d_hi_t),
        "p_fused": float(p_fused),
        "t": t,
    }


def _apply_background(cfg: SynthConfig, h: int, w: int, np_rng: Optional[np.random.RandomState] = None, rng: Optional[random.Random] = None) -> np.ndarray:
    # Base gray
    gmin, gmax = cfg.bg_gray_range
    if np_rng is None:
        base = np.random.randint(gmin, gmax + 1, size=(h, w), dtype=np.uint8)
    else:
        base = np_rng.randint(gmin, gmax + 1, size=(h, w), dtype=np.uint8)
    img = base.astype(np.float32)

    # Directional gradient (tilt)
    if cfg.tilt_enable:
        umin, umax = cfg.tilt_dir_deg
        ang = math.radians((rng.uniform(umin, umax) if rng is not None else random.uniform(umin, umax)))
        ux, uy = math.cos(ang), math.sin(ang)
        cx_shift = max(-0.5, min(0.5, float(cfg.tilt_center[0])))
        cy_shift = max(-0.5, min(0.5, float(cfg.tilt_center[1])))
        x = np.linspace(-1.0 - 2 * cx_shift, 1.0 - 2 * cx_shift, w, dtype=np.float32)
        y = np.linspace(-1.0 - 2 * cy_shift, 1.0 - 2 * cy_shift, h, dtype=np.float32)
        ramp = uy * y[:, None] + ux * x[None, :]
        ramp /= (np.max(np.abs(ramp)) + 1e-6)
        ptp = (rng.uniform(*cfg.tilt_ptp) if rng is not None else random.uniform(*cfg.tilt_ptp))
        img += 0.5 * ptp * ramp

    # Low-frequency illumination
    if cfg.illum_ampl and cfg.illum_ampl > 0:
        noise = (np_rng.randn(h, w).astype(np.float32) if np_rng is not None else np.random.normal(0, 1, (h, w)).astype(np.float32))
        k = max(3, int(round(cfg.illum_sigma)) * 2 + 1)
        if k > 1:
            lf = cv2.GaussianBlur(noise, (k, k), cfg.illum_sigma).astype(np.float32)
        else:
            lf = noise
        rngv = float(np.ptp(lf))
        if rngv > 1e-6:
            lf = (lf - float(lf.min())) / (rngv + 1e-6)
            lf = (lf - 0.5) * 2.0
            img = img + float(cfg.illum_ampl) * lf

    # Vignette
    if cfg.vignette_strength > 0:
        yy, xx = np.ogrid[:h, :w]
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        r /= (r.max() + 1e-6)
        img = img * (1.0 - float(cfg.vignette_strength) * (r ** 2)).astype(np.float32)

    # DIC relief field
    if cfg.relief_field_enable:
        H = (np_rng.randn(h, w).astype(np.float32) if np_rng is not None else np.random.normal(0, 1, (h, w)).astype(np.float32))
        sig = (rng.uniform(*cfg.relief_field_sigma_px) if rng is not None else random.uniform(*cfg.relief_field_sigma_px))
        k = max(3, int(round(3 * sig)) * 2 + 1)
        H = cv2.GaussianBlur(H, (k, k), sig)
        phi = math.radians((rng.uniform(*cfg.relief_field_dir_deg) if rng is not None else random.uniform(*cfg.relief_field_dir_deg)))
        gx = cv2.Sobel(H, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(H, cv2.CV_32F, 0, 1, ksize=3)
        S = math.cos(phi) * gx + math.sin(phi) * gy
        S = S / (S.std() + 1e-6)
        if cfg.relief_field_extra_blur > 0:
            kb = max(3, int(round(cfg.relief_field_extra_blur * 3)) * 2 + 1)
            S = cv2.GaussianBlur(S, (kb, kb), cfg.relief_field_extra_blur)
        gain = (rng.uniform(*cfg.relief_field_gain) if rng is not None else random.uniform(*cfg.relief_field_gain))
        img = img + float(gain) * S

    # Background noise
    if cfg.bg_noise_std and cfg.bg_noise_std > 0:
        if np_rng is not None:
            noise = float(cfg.bg_noise_std) * np_rng.randn(h, w).astype(np.float32)
        else:
            noise = float(cfg.bg_noise_std) * np.random.randn(h, w).astype(np.float32)
        img = img + noise

    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def _noise1d_like(u: np.ndarray, corr: float = 0.25, amp: float = 1.0, seed: Optional[int] = None, np_rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    if np_rng is None:
        if seed is not None:
            np_rng = np.random.RandomState(seed)
        else:
            np_rng = np.random
    grid = np.linspace(-1, 1, 512, dtype=np.float32)
    prof = np_rng.randn(grid.size).astype(np.float32)
    k = max(3, int(round(3 * corr * 64)) * 2 + 1)
    prof = cv2.GaussianBlur(prof.reshape(1, -1), (k, 1), corr * 32).ravel()
    prof -= prof.mean()
    prof /= (prof.std() + 1e-6)
    n = np.interp(u, grid, prof).astype(np.float32)
    return amp * n


def _sin_wobble(u: np.ndarray, amp_px: float = 1.2, cycles: Tuple[float, float] = (0.6, 1.5), rng: Optional[random.Random] = None) -> np.ndarray:
    rr = rng if rng is not None else random
    f = rr.uniform(*cycles) * np.pi
    ph = rr.uniform(0, 2 * np.pi)
    wob = amp_px * (0.6 * np.sin(f * u + ph) + 0.4 * np.sin(1.8 * f * u + 2.0 * ph))
    return wob.astype(np.float32)


def _kink(u: np.ndarray, amp_px: float = 1.5, rng: Optional[random.Random] = None) -> np.ndarray:
    rr = rng if rng is not None else random
    u0 = rr.uniform(-0.3, 0.3)
    s = np.tanh((u - u0) * 6.0)
    return (amp_px * 0.5 * s).astype(np.float32)


def _noisy_wobble(u: np.ndarray, amp_px: float = 1.0, corr: float = 0.18, np_rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    n = _noise1d_like(u, corr=corr, amp=1.0, np_rng=np_rng).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), 1.5)
    n -= n.mean()
    n /= (n.std() + 1e-6)
    return (amp_px * n).astype(np.float32)


def _ragged_mask(u: np.ndarray, p: float = 0.08, corr: float = 0.20, np_rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    grid = np.linspace(-1, 1, 512, dtype=np.float32)
    rngv = np_rng if np_rng is not None else np.random
    keep = (rngv.rand(grid.size) > p).astype(np.float32)
    k = max(3, int(round(3 * corr * 64)) * 2 + 1)
    keep = cv2.GaussianBlur(keep.reshape(1, -1), (k, 1), corr * 32).ravel()
    mn = float(keep.min())
    rngv = float(np.ptp(keep))
    keep = (keep - mn) / (rngv + 1e-6)
    keep = 0.25 + 0.75 * keep
    return np.interp(u, grid, keep).astype(np.float32)


def _smooth_cap(u: np.ndarray, a: float, b: float) -> np.ndarray:
    t = np.clip((np.abs(u) - a) / (b - a + 1e-6), 0.0, 1.0)
    return 1.0 - (t * t * (3.0 - 2.0 * t))


def _draw_phase_contrast_rod(canvas: np.ndarray, cx: float, cy: float, L: float, W: float, ang_deg: float,
                             base_delta: float, cfg: SynthConfig, rng: random.Random,
                             *, width_jit_amp: float = 0.0, edge_jit_amp: float = 0.0,
                             offset_jit_amp: float = 0.0, curve_kappa: float = 0.0,
                             ragged_p: float = 0.0, ragged_corr: float = 0.2,
                             mult_mix: float = 0.0, v_warp_px: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                             polarity_flip_p: float = 0.0, halo_gain_mult: float = 1.0,
                             np_rng: Optional[np.random.RandomState] = None,
                             obb_list: Optional[List[Dict[str, Any]]] = None) -> None:
    """Ported DIC-like rod shader with tapered envelope and bright/dark phase edges."""
    h_img, w_img = canvas.shape[:2]
    # bounding patch around oriented rectangle
    rect = ((cx, cy), (L, W), ang_deg)
    corners = cv2.boxPoints(rect)
    # Record oriented bounding box for overlay visualization
    if obb_list is not None:
        try:
            obb_list.append({
                "cx": float(cx),
                "cy": float(cy),
                "L": float(L),
                "W": float(W),
                "angle_deg": float(ang_deg),
                "corners": [[float(p[0]), float(p[1])] for p in corners.tolist()],
            })
        except Exception:
            pass
    pad = max(6, int(max(cfg.rod_halo_sigma, 3))) + 6
    x0 = max(0, int(np.floor(corners[:, 0].min())) - pad)
    x1 = min(w_img, int(np.ceil(corners[:, 0].max())) + pad)
    y0 = max(0, int(np.floor(corners[:, 1].min())) - pad)
    y1 = min(h_img, int(np.ceil(corners[:, 1].max())) + pad)
    if x1 <= x0 or y1 <= y0:
        return

    h, w = y1 - y0, x1 - x0
    patch = canvas[y0:y1, x0:x1].astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    X = xx + x0 - cx
    Y = yy + y0 - cy
    th = math.radians(ang_deg)
    ct, st = math.cos(th), math.sin(th)
    u = (ct * X + st * Y) / (L / 2.0 + 1e-6)
    v = (-st * X + ct * Y)
    v_warp = None
    if callable(v_warp_px):
        v_warp = v_warp_px(u)
    elif v_warp_px is not None:
        v_warp = v_warp_px
    if v_warp is not None:
        if isinstance(v_warp, np.ndarray):
            if v_warp.ndim == 1 and v_warp.size == u.size:
                v_warp = v_warp.reshape(u.shape)
            elif v_warp.shape != u.shape:
                v_warp = np.broadcast_to(v_warp, u.shape)
        v = v - v_warp
    if curve_kappa != 0.0:
        v = v + curve_kappa * (u * u - 1 / 3) * L

    # tapered envelope
    taper = cfg.taper_strength * (np.abs(u) ** cfg.taper_power)
    w_u = np.maximum(cfg.min_width_ratio, 1.0 - taper) * (W + 1e-6)
    if width_jit_amp > 0:
        w_u = w_u * (1.0 + _noise1d_like(u, corr=0.22, amp=width_jit_amp))

    sigma_v = (w_u * max(1e-6, cfg.cross_soft_sigma))
    alpha_v = np.exp(-0.5 * (v / sigma_v) ** 2)
    alpha_u = _smooth_cap(u, a=0.78, b=1.00)
    alpha_fill = np.clip(alpha_v * alpha_u, 0.0, 1.0)
    # base delta body
    g = (np_rng.randn() if np_rng is not None else np.random.randn())
    delta_body = float(base_delta) * (1.0 + 0.05 * g)
    layer = delta_body * alpha_fill

    # phase contrast edge pair
    # Shadow params sampled outside; we use cfg ranges for convenience
    shadow_gain = rng.uniform(*cfg.shadow_gain)
    shadow_width_mult = rng.uniform(*cfg.shadow_width_mult)
    shadow_bias = rng.uniform(*cfg.shadow_bias)
    shadow_offset_px = rng.uniform(*cfg.shadow_offset_px)

    sigma_pc = np.clip(sigma_v * shadow_width_mult, 0.6, None)
    sign = 1.0 if (rng.random() < 0.5) else -1.0
    v_shift = v - (shadow_offset_px * (1.0 + offset_jit_amp * _noise1d_like(u, 0.25, 1.0, np_rng=np_rng)) * sign)
    pc = (v_shift / sigma_pc) * np.exp(-0.5 * (v_shift / sigma_pc) ** 2) / 0.60653066
    pc *= alpha_u
    polarity = 1.0 if (rng.random() < 0.5) else -1.0
    pc *= polarity
    if edge_jit_amp > 0:
        pc = pc * (1.0 + edge_jit_amp * _noise1d_like(u, corr=0.22, amp=1.0, np_rng=np_rng))
    if polarity_flip_p > 0.0 and (rng.random() < polarity_flip_p):
        flips = np.sign(_noise1d_like(u, corr=0.30, amp=1.0, np_rng=np_rng))
        flips[flips == 0] = 1.0
        pc *= flips
    pc_pos = np.maximum(pc, 0.0) * (1.0 - shadow_bias)
    pc_neg = np.minimum(pc, 0.0) * (1.0 + shadow_bias)
    pc = pc_pos + pc_neg
    if ragged_p > 0:
        mask_u = _ragged_mask(u, p=ragged_p, corr=ragged_corr, np_rng=np_rng)
        pc *= mask_u
        alpha_fill *= mask_u
    layer = layer + shadow_gain * pc

    # faint outer halo
    if cfg.rod_halo_sigma > 0 and cfg.rod_halo_gain != 0:
        support = (alpha_fill > 0.12).astype(np.float32)
        k2 = max(3, int(round(cfg.rod_halo_sigma * 3)) * 2 + 1)
        blurred = cv2.GaussianBlur(support, (k2, k2), cfg.rod_halo_sigma)
        halo = np.clip(blurred - support, 0, 1)
        layer = layer + (delta_body * (cfg.rod_halo_gain * halo_gain_mult)) * halo

    # composite into patch luminance
    base = patch.copy()
    layer3 = layer[..., None]  # broadcast layer across BGR channels
    if mult_mix > 0:
        add = np.clip(base + layer3, 0, 255)
        mul = np.clip(base * (1.0 + layer3 / 255.0), 0, 255)
        patch = (1.0 - mult_mix) * add + mult_mix * mul
    else:
        patch = np.clip(base + layer3, 0, 255)

    canvas[y0:y1, x0:x1] = patch.astype(np.uint8)


def _draw_rod(canvas: np.ndarray, cx: float, cy: float, L: float, W: float, ang_deg: float,
              cfg: SynthConfig, rng: random.Random, curvature: float = 0.0) -> None:
    # Simple DIC-like pair of edges across width with tapering envelope
    h, w = canvas.shape[:2]
    # Build a local patch bounding box
    W_eff = max(1.0, W)
    L_eff = max(5.0, L)
    rad = math.radians(ang_deg)
    # Oriented rectangle for mask
    rect = ((cx, cy), (L_eff, W_eff), ang_deg)
    box = cv2.boxPoints(rect).astype(np.float32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [box.astype(np.int32)], 255)

    # Compute gradient direction across width (unit normal)
    nx, ny = -math.sin(rad), math.cos(rad)
    # Shadow params
    gain = rng.uniform(*cfg.shadow_gain)
    width_mult = rng.uniform(*cfg.shadow_width_mult)
    bias = rng.uniform(*cfg.shadow_bias)
    offset_px = rng.uniform(*cfg.shadow_offset_px)

    # Signed distance along normal; compute shading only inside mask
    ys, xs = np.where(mask > 0)
    # project to local coordinates centered at (cx,cy)
    dx = xs.astype(np.float32) - cx
    dy = ys.astype(np.float32) - cy
    v = dx * nx + dy * ny  # across width
    u = dx * math.cos(rad) + dy * math.sin(rad)  # along length
    # Optional curvature: bend shading edges slightly as a function of u
    if curvature:
        v = v + float(curvature) * ((u / (L_eff + 1e-6)) ** 2) * W_eff
    # Taper along u in [-L/2, L/2]
    u_norm = np.clip((np.abs(u) / (0.5 * L_eff)), 0.0, 1.0)
    taper = 1.0 - cfg.taper_strength * (u_norm ** 1.0)

    # Pair of edges: positive/negative lobes (bright/dark)
    sigma = width_mult * W_eff
    # Offset asymmetry
    v0 = v - offset_px
    v1 = v + offset_px
    bright = np.exp(-0.5 * (v0 / (sigma + 1e-6)) ** 2)
    dark = -np.exp(-0.5 * (v1 / (sigma + 1e-6)) ** 2) * (1.0 + bias)
    shade = gain * (bright + dark) * taper

    # Apply to canvas luminance (vectorized update)
    if ys.size > 0:
        c = canvas[ys, xs, 0].astype(np.float32)
        c2 = np.clip(c + shade.astype(np.float32), 0, 255)
        stacked = np.stack([c2, c2, c2], axis=-1).astype(canvas.dtype)
        canvas[ys, xs] = stacked


def _find_ttf(cfg: SynthConfig) -> Optional[str]:
    if cfg.scalebar_ttf and Path(str(cfg.scalebar_ttf)).exists():
        return str(cfg.scalebar_ttf)
    try:
        import matplotlib
        p = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
        if p.exists():
            return str(p)
    except Exception:
        pass
    for p in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/google-noto/NotoSans-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]:
        if Path(p).exists():
            return p
    return None


def _draw_scale_legend_bgr(img_bgr: np.ndarray, cfg: SynthConfig, rng: random.Random) -> np.ndarray:
    if not cfg.scalebar_enable or rng.random() > cfg.scalebar_prob:
        return img_bgr
    h, w = img_bgr.shape[:2]
    # text
    val = rng.randint(cfg.scalebar_value_range[0], cfg.scalebar_value_range[1])
    units = list(cfg.scalebar_units)
    idx = int(rng.random() * len(units)) if units else 0
    unit = units[idx] if units else ""
    text = f"{val} {unit}"
    font_px = int(rng.uniform(*cfg.scalebar_font_px))
    ttf = _find_ttf(cfg) if PIL_AVAILABLE else None
    if ttf is None and ("μ" in unit):
        text = f"{val} um"
    # Pillow canvas
    if PIL_AVAILABLE:
        pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        font = ImageFont.truetype(ttf, font_px) if ttf else ImageFont.load_default()
        # bar geometry
        L = int(rng.uniform(*cfg.scalebar_len_px))
        thick = int(rng.uniform(*cfg.scalebar_thick_px))
        margin = cfg.scalebar_margin_px
        corners = ["tl", "tr", "bl", "br"]
        corner = corners[int(rng.random() * len(corners))]
        c = int(rng.randint(cfg.scalebar_white_jit[0], cfg.scalebar_white_jit[1]))
        fill = (c, c, c)
        if corner == "bl":
            x0, y0 = margin, h - margin
            x1, y1 = x0 + L, y0
        elif corner == "br":
            x1, y1 = w - margin, h - margin
            x0, y0 = x1 - L, y1
        elif corner == "tl":
            x0, y0 = margin, margin
            x1, y1 = x0 + L, y0
        else:
            x1, y1 = w - margin, margin
            x0, y0 = x1 - L, y1
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        pad = 6
        if corner in ("bl", "br"):
            tx = x1 - tw if corner == "br" else x0
            ty = y0 - pad - th
        else:
            tx = x1 - tw if corner == "tr" else x0
            ty = y0 + pad
        tx = max(0, min(tx, w - tw))
        ty = max(0, min(ty, h - th))
        if cfg.scalebar_outline:
            sw = max(1, int(round(font_px * 0.08)))
            draw.text((tx, ty), text, font=font, fill=fill, stroke_width=sw, stroke_fill=(0, 0, 0))
        else:
            draw.text((tx, ty), text, font=font, fill=fill)
        if cfg.scalebar_outline:
            draw.line([(x0, y0), (x1, y1)], fill=(0, 0, 0), width=thick + 2)
        draw.line([(x0, y0), (x1, y1)], fill=fill, width=thick)
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    else:
        # Fallback: simple bar only
        L = int(rng.uniform(*cfg.scalebar_len_px))
        thick = int(rng.uniform(*cfg.scalebar_thick_px))
        margin = cfg.scalebar_margin_px
        bl = rng.random() < 0.5
        if bl:
            x0, y0 = margin, h - margin
            x1, y1 = x0 + L, y0
        else:
            x1, y1 = w - margin, h - margin
            x0, y0 = x1 - L, y1
        cv2.line(img_bgr, (x0, y0), (x1, y1), (0, 0, 0), thickness=thick + 2, lineType=cv2.LINE_AA)
        cv2.line(img_bgr, (x0, y0), (x1, y1), (255, 255, 255), thickness=thick, lineType=cv2.LINE_AA)
        return img_bgr


def generate_image(config: Dict[str, Any], t: float, seed: int | None = None, return_obbs: bool = False, parallel_workers: int | None = None):
    """
    Generate a single synthetic image using the provided config and per-row t.
    The RNG is randomized by default to ensure different previews across clicks.
    If seed is provided, it will be used for reproducible results.
    """
    cfg = SynthConfig(**config)
    # Deterministic RNG: both Python random and NumPy
    if seed is None:
        rng = random.Random()
        rng.seed(random.SystemRandom().randint(0, 2**31 - 1))
        np_rng = np.random.RandomState(random.SystemRandom().randint(0, 2**31 - 1))
    else:
        rng = random.Random(seed)
        np_rng = np.random.RandomState(seed)

    w, h = cfg.width, cfg.height
    img = _apply_background(cfg, h, w, np_rng=np_rng, rng=rng)
    obbs = [] if return_obbs else None

    # Per-t parameters
    p = params_for_t(cfg, t)
    n_rods = int(p.get("n_rods", 0))
    if "n_rods_min" in p and "n_rods_max" in p:
        try:
            n_rods = rng.randint(int(p["n_rods_min"]), int(p["n_rods_max"]))
        except ValueError:
            # Fallback if bounds are invalid/equal
            n_rods = int(p.get("n_rods", max(0, int(p.get("n_rods_min", 0)))))
    len_max = p["rod_len_max"]
    ar_max = p["rod_aspect_max"]
    p_fused = p["p_fused"]

    # Draw rods (and occasional fused crystals)
    def rng_choice(rng_obj: random.Random, seq: Sequence):
        if not seq:
            return None
        i = int(rng_obj.random() * len(seq))
        return seq[i]

    # Collect rod entries for optional parallel rendering.
    rod_entries: List[Tuple[float, float, float, float, float, float, int]] = []  # (cx, cy, L, W, ang, delta, seed)
    if cfg.rods_enable:
        for _ in range(n_rods):
            L = rng.uniform(p["rod_len_min"], len_max)
            asp = rng.uniform(p["rod_aspect_min"], ar_max)
            W = max(2.0, L * asp)
            cx = rng.uniform(0.05 * w, 0.95 * w)
            cy = rng.uniform(0.05 * h, 0.95 * h)
            ang = rng.uniform(-90.0, 90.0)
            delta = rng.uniform(p["rod_delta_min"], p["rod_delta_max"])
            # fused crystal arms become additional rods around main angle
            if cfg.fused_enable and rng.random() < p_fused:
                n_arms = rng.randint(2, 5)
                spread = rng.uniform(10.0, 55.0)
                main = ang
                for _arm in range(n_arms):
                    d = np_rng.normal(0, spread * 0.25) if np_rng is not None else np.random.normal(0, spread * 0.25)
                    a = main + d
                    L_i = max(8.0, (np_rng.normal(L, 0.25 * L) if np_rng is not None else np.random.normal(L, 0.25 * L)))
                    W_i = max(3.0, (np_rng.normal(W, 0.25 * W) if np_rng is not None else np.random.normal(W, 0.25 * W)))
                    r = rng.uniform(0, 0.25 * W)
                    cx_i = cx + r * math.cos(math.radians(a + 90))
                    cy_i = cy + r * math.sin(math.radians(a + 90))
                    seed_i = rng.randint(0, 2**31 - 1)
                    rod_entries.append((cx_i, cy_i, L_i, W_i, a, delta, seed_i))
            else:
                seed_i = rng.randint(0, 2**31 - 1)
                rod_entries.append((cx, cy, L, W, ang, delta, seed_i))

    # Decide workers from function arg or config
    workers = parallel_workers if parallel_workers is not None else (cfg.parallel_workers or 1)

    def _render_single_rod(entry):
        cx_i, cy_i, L_i, W_i, a_i, delta_i, seed_i = entry
        layer = np.zeros_like(img)
        rng_local = random.Random(seed_i)
        np_rng_local = np.random.RandomState(seed_i)
        local_obbs: List[Any] = [] if return_obbs else None
        _draw_phase_contrast_rod(
            layer, cx_i, cy_i, L_i, W_i, a_i, delta_i, cfg, rng_local,
            width_jit_amp=0.0,
            edge_jit_amp=0.0,
            offset_jit_amp=0.0,
            curve_kappa=0.0,
            ragged_p=0.0,
            ragged_corr=0.2,
            mult_mix=0.0,
            v_warp_px=None,
            polarity_flip_p=0.0,
            halo_gain_mult=1.0,
            np_rng=np_rng_local,
            obb_list=local_obbs)
        return layer, (local_obbs or [])

    if workers and workers > 1 and len(rod_entries) > 1:
        try:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=int(workers)) as ex:
                for layer, local_obbs in ex.map(_render_single_rod, rod_entries):
                    img = np.clip(img.astype(np.int16) + layer.astype(np.int16), 0, 255).astype(np.uint8)
                    if return_obbs and obbs is not None:
                        obbs.extend(local_obbs)
        except Exception:
            # Fallback: sequential
            for (cx, cy, L, W, ang, delta, _seed) in rod_entries:
                _draw_phase_contrast_rod(
                    img, cx, cy, L, W, ang, delta, cfg, rng,
                    width_jit_amp=0.0,
                    edge_jit_amp=0.0,
                    offset_jit_amp=0.0,
                    curve_kappa=0.0,
                    ragged_p=0.0,
                    ragged_corr=0.2,
                    mult_mix=0.0,
                    v_warp_px=None,
                    polarity_flip_p=0.0,
                    halo_gain_mult=1.0,
                    np_rng=np_rng,
                    obb_list=obbs)
    else:
        # original sequential draw
        for (cx, cy, L, W, ang, delta, _seed) in rod_entries:
            _draw_phase_contrast_rod(
                img, cx, cy, L, W, ang, delta, cfg, rng,
                width_jit_amp=0.0,
                edge_jit_amp=0.0,
                offset_jit_amp=0.0,
                curve_kappa=0.0,
                ragged_p=0.0,
                ragged_corr=0.2,
                mult_mix=0.0,
                v_warp_px=None,
                polarity_flip_p=0.0,
                halo_gain_mult=1.0,
                np_rng=np_rng,
                obb_list=obbs)

    # Optional ghost rods layer with distortions
    if cfg.ghost_enable and cfg.ghost_fraction > 0:
        ghost_count = max(0, int(round(n_rods * float(cfg.ghost_fraction))))
        if ghost_count > 0:
            for _ in range(ghost_count):
                Lg = rng.uniform(12.0, len_max)
                arg = rng.uniform(0.01, min(0.8, ar_max * 2))
                Wg = max(2.0, Lg * arg)
                cxg = rng.uniform(0.05 * w, 0.95 * w)
                cyg = rng.uniform(0.05 * h, 0.95 * h)
                angg = rng.uniform(-90.0, 90.0)
                delt_g = rng.uniform(*cfg.ghost_delta_rng)
                # warp shape mode randomly
                shape_mode = rng_choice(rng, ["wavy", "kink", "noisy", "straight"]) or "straight"
                amp = 0.6 + 0.004 * Lg
                if shape_mode == "wavy":
                    v_warp_fn = lambda u: _sin_wobble(u, amp_px=amp, cycles=(0.7, 1.6), rng=rng)
                elif shape_mode == "kink":
                    v_warp_fn = lambda u: _kink(u, amp_px=1.0 + 0.006 * Lg, rng=rng)
                elif shape_mode == "noisy":
                    v_warp_fn = lambda u: _noisy_wobble(u, amp_px=1.0, corr=0.22, np_rng=np_rng)
                else:
                    v_warp_fn = None
                _draw_phase_contrast_rod(
                    img, cxg, cyg, Lg, Wg, angg, delt_g, cfg, rng,
                    width_jit_amp=cfg.ghost_width_jit_amp,
                    edge_jit_amp=cfg.ghost_edge_jit_amp,
                    offset_jit_amp=cfg.ghost_offset_jit_amp,
                    curve_kappa=rng.uniform(*cfg.ghost_curve_kappa_range),
                    ragged_p=cfg.ghost_ragged_p,
                    ragged_corr=cfg.ghost_ragged_corr,
                    mult_mix=cfg.ghost_mult_mix,
                    v_warp_px=v_warp_fn,
                    polarity_flip_p=0.0,
                    halo_gain_mult=1.0,
                    np_rng=np_rng,
                    obb_list=None)
            if cfg.ghost_blur_sigma and cfg.ghost_blur_sigma > 0:
                k = max(3, int(round(cfg.ghost_blur_sigma * 3)) * 2 + 1)
                img = cv2.GaussianBlur(img, (k, k), cfg.ghost_blur_sigma)
            if cfg.ghost_noise_std and cfg.ghost_noise_std > 0:
                noise = np_rng.randn(*img.shape).astype(np.float32) * float(cfg.ghost_noise_std)
                img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Debris
    if cfg.debris_rate and cfg.debris_rate > 0:
        h0, w0 = img.shape[:2]
        n_deb = int(w0 * h0 * float(cfg.debris_rate))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        for _ in range(n_deb):
            x = rng.randint(0, w0 - 1)
            y = rng.randint(0, h0 - 1)
            delta = rng.uniform(*cfg.debris_int_delta)
            if random.random() < cfg.debris_dash_prob:
                Ld = rng.randint(*cfg.debris_size_px) * 2 + 1
                angd = rng.uniform(-90, 90) * np.pi / 180.0
                dx, dy = math.cos(angd), math.sin(angd)
                for tt in np.linspace(-Ld, Ld, 2 * Ld + 1):
                    xi = int(round(x + tt * dx))
                    yi = int(round(y + tt * dy))
                    if 0 <= xi < w0 and 0 <= yi < h0:
                        gray[yi, xi] = np.clip(gray[yi, xi] + delta, 0, 255)
            else:
                r = rng.randint(*cfg.debris_size_px)
                x0 = max(0, x - r)
                x1 = min(w0, x + r + 1)
                y0 = max(0, y - r)
                y1 = min(h0, y + r + 1)
                patch = gray[y0:y1, x0:x1]
                yy, xx = np.ogrid[:patch.shape[0], :patch.shape[1]]
                cx0, cy0 = (patch.shape[1] - 1) / 2.0, (patch.shape[0] - 1) / 2.0
                m = ((xx - cx0) ** 2 + (yy - cy0) ** 2) <= r * r
                patch[m] = np.clip(patch[m] + delta, 0, 255)
                gray[y0:y1, x0:x1] = patch
        img = cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # Final blur
    if cfg.blur_sigma > 0:
        k = max(3, int(round(cfg.blur_sigma * 3)) * 2 + 1)
        img = cv2.GaussianBlur(img, (k, k), cfg.blur_sigma)

    # Scalebar legend with text
    img = _draw_scale_legend_bgr(img, cfg, rng)
    if return_obbs:
        return img, (obbs or [])
    return img


def sample_lambda(rng: random.Random, cfg: SynthConfig) -> float:
    lo, hi = cfg.stage_lambda_range
    log_lo, log_hi = math.log10(lo), math.log10(hi)
    return 10 ** rng.uniform(log_lo, log_hi)


def lambda_to_t(lmbda: float) -> float:
    t = (math.log10(max(1e-6, lmbda)) + 1.0) / 2.0
    return float(np.clip(t, 0.0, 1.0))