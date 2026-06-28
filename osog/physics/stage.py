"""Stage factor (t) and batch lambda sampling for OSOG."""

from __future__ import annotations

import copy
import math
import random
from typing import Any, Tuple, Union

import numpy as np

from ..config import SynthConfig


RangeVal = Union[Tuple[Any, ...], list]


def ensure_config(cfg: Union[SynthConfig, dict]) -> SynthConfig:
    if isinstance(cfg, SynthConfig):
        return cfg
    if isinstance(cfg, dict):
        if "canvas" in cfg or "physics" in cfg:
            return SynthConfig.from_dict(cfg)
        return SynthConfig.from_flat_dict(cfg)
    raise TypeError(f"Expected SynthConfig or dict, got {type(cfg)!r}")


def lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)


def sample_lambda(rng: random.Random, cfg: Union[SynthConfig, dict]) -> float:
    cfg = ensure_config(cfg)
    lo, hi = cfg.physics.stage_lambda_range
    log_lo, log_hi = math.log10(lo), math.log10(hi)
    return 10 ** rng.uniform(log_lo, log_hi)


def lambda_to_t(lmbda: float) -> float:
    t = (math.log10(max(1e-6, lmbda)) + 1.0) / 2.0
    return float(np.clip(t, 0.0, 1.0))


def clamp_t(t: float) -> float:
    return float(max(0.0, min(1.0, float(t))))


def stage_bounds(range_val: RangeVal, t: float) -> Tuple[float, float]:
    """Return (lo, hi_eff) sampling bounds at stage t.

    - 2-value [lo, hi]: hi_eff ramps lo → hi (early crystals small, late large).
    - 3-value [lo, hi0, hi1] with hi0 != hi1: ceiling ramps hi0 → hi1 (legacy).
    - 3-value [lo, hi, hi] (degenerate, common in presets/UI): treated as [lo, hi].
    """
    t = clamp_t(t)
    if isinstance(range_val, (list, tuple)):
        if len(range_val) >= 3:
            lo, hi0, hi1 = float(range_val[0]), float(range_val[1]), float(range_val[2])
            if abs(hi0 - hi1) < 1e-9:
                hi_eff = lerp(lo, hi0, t)
            else:
                hi_eff = lerp(hi0, hi1, t)
            return lo, max(lo, hi_eff)
        if len(range_val) == 2:
            lo, hi = float(range_val[0]), float(range_val[1])
            hi_eff = lerp(lo, hi, t)
            return lo, max(lo, hi_eff)
    try:
        lo = float(range_val)
    except Exception:
        lo = 0.0
    return lo, lo


def lo_and_hi_at_t(range_val: RangeVal, t: float) -> Tuple[float, float]:
    """Legacy alias for stage_bounds."""
    return stage_bounds(range_val, t)


def _hi_mature(range_val: RangeVal) -> float:
    if isinstance(range_val, (list, tuple)):
        if len(range_val) >= 3:
            return float(range_val[2])
        if len(range_val) == 2:
            return float(range_val[1])
    return float(range_val)


def stage_count_range(range_val: RangeVal, t: float) -> Tuple[int, int]:
    """Deterministic particle count at stage t."""
    t = clamp_t(t)
    if isinstance(range_val, (list, tuple)) and len(range_val) >= 3:
        lo, hi0, hi1 = float(range_val[0]), float(range_val[1]), float(range_val[2])
        if abs(hi0 - hi1) >= 1e-9:
            hi_t = lerp(hi0, hi1, t)
            n = int(round(lerp(lo, hi_t, t)))
        else:
            n = int(round(lerp(lo, hi0, t)))
    elif isinstance(range_val, (list, tuple)) and len(range_val) == 2:
        lo, hi = float(range_val[0]), float(range_val[1])
        n = int(round(lerp(lo, hi, t)))
    else:
        n = int(round(_hi_mature(range_val)))
    n = max(0, n)
    return n, n


def stage_float_range(range_val: RangeVal, t: float) -> Tuple[float, float]:
    """Continuous size/aspect range at stage t."""
    lo, hi_eff = stage_bounds(range_val, t)
    return float(lo), float(hi_eff)


def _apply_count_and_ranges(obj, t: float, count_attr: str, range_attrs: tuple[str, ...]) -> None:
    if not hasattr(obj, count_attr):
        return
    count_val = getattr(obj, count_attr)
    if count_val is not None:
        setattr(obj, count_attr, stage_count_range(count_val, t))
    for attr in range_attrs:
        if hasattr(obj, attr):
            val = getattr(obj, attr)
            if val is not None:
                setattr(obj, attr, stage_float_range(val, t))


def apply_stage_to_config(cfg: Union[SynthConfig, dict], t: float) -> SynthConfig:
    """Return a copy of cfg with stage t applied to counts, sizes, and fusion."""
    cfg = copy.deepcopy(ensure_config(cfg))
    t = clamp_t(t)
    phys = cfg.physics

    rods = phys.rods
    rs = phys.rod_specs
    if rs.enable:
        count_src = rods.n_rods_rng_lo_hi if rods else rs.count_range
        len_src = rods.rod_len_px_lo_hi if rods else rs.length_range
        asp_src = rods.rod_aspect_lo_hi if rods else rs.aspect_range
        rs.count_range = stage_count_range(count_src, t)
        rs.length_range = stage_float_range(len_src, t)
        rs.aspect_range = stage_float_range(asp_src, t)

    _apply_count_and_ranges(phys.sphere_specs, t, "count_range", ("diameter_range",))
    _apply_count_and_ranges(phys.cube_specs, t, "count_range", ("size_range",))
    _apply_count_and_ranges(
        phys.plate_specs,
        t,
        "count_range",
        ("size_range", "aspect_range", "thickness_range"),
    )
    _apply_count_and_ranges(phys.polyhedra_specs, t, "count_range", ("size_range",))
    _apply_count_and_ranges(phys.bubble_specs, t, "count_range", ("diameter_range",))
    _apply_count_and_ranges(phys.droplet_specs, t, "count_range", ("diameter_range",))
    _apply_count_and_ranges(phys.incrustation_specs, t, "count_range", ("size_range",))

    fused = phys.fused
    if fused.enable:
        fused.p1 = float(lerp(fused.p0, fused.p1, t))

    return cfg


def params_for_t(cfg: Union[SynthConfig, dict], t: float) -> dict:
    """Legacy-compatible stage parameter dict (rods + fused)."""
    cfg = ensure_config(cfg)
    t = clamp_t(t)
    rods = cfg.physics.rods
    fused = cfg.physics.fused

    n_lo, n_hi_t = lo_and_hi_at_t(rods.n_rods_rng_lo_hi, t)
    n_rods = int(round(lerp(n_lo, n_hi_t, t)))
    l_lo, l_hi_t = lo_and_hi_at_t(rods.rod_len_px_lo_hi, t)
    ar_lo, ar_hi_t = lo_and_hi_at_t(rods.rod_aspect_lo_hi, t)
    d_lo, d_hi_t = lo_and_hi_at_t(getattr(rods, "rod_delta_rng", (0.0, 0.0)), t)
    p_fused = float(lerp(fused.p0, fused.p1, t))

    return {
        "t": t,
        "n_rods_min": int(round(n_lo)),
        "n_rods_max": int(round(n_hi_t)),
        "n_rods": int(n_rods),
        "rod_len_min": float(l_lo),
        "rod_len_max": float(l_hi_t),
        "rod_aspect_min": float(ar_lo),
        "rod_aspect_max": float(ar_hi_t),
        "rod_delta_min": float(d_lo),
        "rod_delta_max": float(d_hi_t),
        "p_fused": p_fused,
    }
