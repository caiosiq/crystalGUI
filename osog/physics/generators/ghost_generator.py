
import torch
import random
from ...config import SynthConfig
from ..constants import SHAPE_ROD, SHAPE_PLATE
from .utils import rand_uniform
from .main_generator import generate_main_particles


def _empty_results():
    return {
        "cx": [], "cy": [], "z": [],
        "L": [], "W": [], "H": [],
        "alpha": [], "beta": [], "gamma": [],
        "delta": [], "seed": [], "group_id": [],
        "req_label": [], "shape_id": [],
        "curv": [], "w_jit": [], "off_jit": [], "edge_jit": [],
        "pol_p": [], "rag_p": [], "rag_corr": [], "shape_mode": [],
        "ref_index": [], "birefringence": [], "opacity": [],
        "tex_type": [], "surf_rough": [], "grain_size": [], "inclusions": [],
        "turbidity": [],
        "anisotropy": [], "anisotropy_angle": [],
        "reflectivity": [], "dispersion": [], "absorption_color": [],
    }


def _apply_ghost_transforms(results, cfg: SynthConfig, generator: torch.Generator):
    """Same-species particles with reduced size, contrast, and defocus."""
    gs = cfg.physics.ghosts
    med_ri = cfg.optics.medium_refractive_index
    focus_z = cfg.optics.focus_z
    z_min, z_max = cfg.physics.z_range
    atten = float(max(0.0, min(1.0, gs.delta_attenuation)))
    size_center = float(max(0.05, min(1.0, gs.size_scale)))
    size_jitter = float(max(0.0, min(0.25, gs.size_scale_jitter)))
    curv_p = float(max(0.0, min(1.0, gs.curvature)))

    for i in range(len(results["L"])):
        n = len(results["L"][i])
        if n == 0:
            continue

        lo = max(0.05, size_center - size_jitter)
        hi = min(1.0, size_center + size_jitter)
        scale = rand_uniform(n, lo, hi, generator)
        results["L"][i] = results["L"][i] * scale
        results["W"][i] = results["W"][i] * scale
        results["H"][i] = results["H"][i] * scale

        results["delta"][i] = results["delta"][i] * atten
        results["ref_index"][i] = med_ri + results["delta"][i]

        z = results["z"][i]
        sign = torch.where(z >= focus_z, torch.ones(n), -torch.ones(n))
        offset = rand_uniform(
            n, gs.defocus_offset_range[0], gs.defocus_offset_range[1], generator
        )
        new_z = focus_z + sign * (torch.abs(z - focus_z) + offset)
        results["z"][i] = torch.clamp(new_z, z_min, z_max)

        if curv_p > 0:
            shape_id = results["shape_id"][i]
            is_elongated = (shape_id == SHAPE_ROD) | (shape_id == SHAPE_PLATE)
            bent = (torch.rand(n, generator=generator) < curv_p) & is_elongated
            if torch.any(bent):
                new_modes = torch.randint(1, 4, (n,), generator=generator)
                results["shape_mode"][i] = torch.where(bent, new_modes, results["shape_mode"][i])

        results["group_id"][i] = torch.full((n,), -1, dtype=torch.long)


def generate_ghosts(
    cfg: SynthConfig,
    n_total_main: int,
    w: int,
    h: int,
    generator: torch.Generator,
    rng: random.Random,
):
    """
    Generate out-of-focus ghost particles using the same enabled species as mains,
    scaled down with lower contrast and no training labels.
    """
    gs = cfg.physics.ghosts
    if not (gs.enable and gs.fraction > 0 and n_total_main > 0):
        return _empty_results()

    results = generate_main_particles(
        cfg,
        w,
        h,
        generator,
        rng,
        count_scale=float(gs.fraction),
        requires_label=False,
    )

    if not results["cx"]:
        return _empty_results()

    _apply_ghost_transforms(results, cfg, generator)
    return results
