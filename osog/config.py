from dataclasses import dataclass, field, asdict
from typing import Tuple, Dict, Any, Optional

@dataclass
class CanvasConfig:
    width: int = 1024
    height: int = 768
    parallel_workers: Optional[int] = None
    use_gpu: bool = False

@dataclass
class RodsConfig:
    enable: bool = True
    n_rods_rng_lo_hi: Tuple[int, int, Optional[int]] = (150, 600, 600) # lo, hi_t0, hi_t1
    rod_len_px_lo_hi: Tuple[float, float, Optional[float]] = (30.0, 380.0, 380.0)
    rod_aspect_lo_hi: Tuple[float, float, Optional[float]] = (0.02, 0.30, 0.30)
    rod_delta_rng: Tuple[float, float, Optional[float]] = (-12.0, 0.0, 0.0)
    
    # Legacy compatibility helper for 2-tuple vs 3-tuple
    def __post_init__(self):
        # Ensure tuples are length 3 if they were passed as length 2 (legacy)
        # This is a bit tricky with dataclasses, usually handled by the caller or factory
        pass

@dataclass
class GhostsConfig:
    enable: bool = False
    fraction: float = 0.2
    gain_mult: float = 0.5
    blur_sigma: float = 0.0
    delta_rng: Tuple[float, float] = (-3.0, 0.0)
    curvature: float = 0.0
    width_jit_amp: float = 0.25
    edge_jit_amp: float = 0.25
    offset_jit_amp: float = 0.39
    curve_kappa_range: Tuple[float, float] = (-0.125, 0.125)
    ragged_p: float = 0.90
    ragged_corr: float = 0.90
    mult_mix: float = 0.90

@dataclass
class DebrisConfig:
    rate: float = 0.0
    int_delta: Tuple[float, float] = (-6.0, 6.0)
    dash_prob: float = 0.15
    size_px: Tuple[int, int] = (1, 3)

@dataclass
class FusedConfig:
    enable: bool = True
    p0: float = 0.0001
    p1: float = 0.003

@dataclass
class PhysicsConfig:
    rods: RodsConfig = field(default_factory=RodsConfig)
    ghosts: GhostsConfig = field(default_factory=GhostsConfig)
    debris: DebrisConfig = field(default_factory=DebrisConfig)
    fused: FusedConfig = field(default_factory=FusedConfig)
    stage_lambda_range: Tuple[float, float] = (0.1, 10.0)

@dataclass
class OpticsConfig:
    # Rod model optics
    taper_strength: float = 0.45
    taper_power: float = 1.0
    min_width_ratio: float = 0.35
    cross_soft_sigma: float = 0.30
    rod_halo_sigma: float = 3.2
    rod_halo_gain: float = 0.4
    
    # Shadow properties (DIC)
    shadow_gain: Tuple[float, float] = (6.0, 12.0)
    shadow_width_mult: Tuple[float, float] = (0.02, 0.05)
    shadow_bias: Tuple[float, float] = (0.05, 0.12)
    shadow_offset_px: Tuple[float, float] = (0.05, 0.25)

@dataclass
class ScalebarConfig:
    enable: bool = True
    prob: float = 0.5
    len_px: Tuple[int, int] = (80, 240)
    thick_px: Tuple[int, int] = (2, 12)
    margin_px: int = 24
    outline: bool = True
    font_px: Tuple[int, int] = (70, 80)
    white_jit: Tuple[int, int] = (245, 255)
    units: Tuple[str, ...] = ("μm", "um", "nm", "mm")
    value_range: Tuple[int, int] = (10, 99)
    ttf: Optional[str] = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

@dataclass
class SensorConfig:
    bg_gray_range: Tuple[int, int] = (90, 97)
    vignette_strength: float = 0.0
    bg_noise_std: float = 0.0
    
    tilt_enable: bool = True
    tilt_dir_deg: Tuple[float, float] = (-30.0, 30.0)
    tilt_ptp: Tuple[float, float] = (12.0, 15.0)
    tilt_center: Tuple[float, float] = (0.0, 0.0)
    
    illum_ampl: float = 0.0
    illum_sigma: float = 0.0
    
    relief_field_enable: bool = True
    relief_field_sigma_px: Tuple[float, float] = (20.0, 20.0)
    relief_field_gain: Tuple[float, float] = (-0.5, 0.0)
    relief_field_dir_deg: Tuple[float, float] = (0.0, 0.0)
    relief_field_extra_blur: float = 0.0
    
    blur_sigma: float = 0.0 # Global blur
    scalebar: ScalebarConfig = field(default_factory=ScalebarConfig)

@dataclass
class SynthConfig:
    """
    Main configuration for the synthesis pipeline.
    Now supports nested architecture (Canvas, Physics, Optics, Sensor).
    """
    canvas: CanvasConfig = field(default_factory=CanvasConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    optics: OpticsConfig = field(default_factory=OpticsConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SynthConfig':
        """
        Construct SynthConfig from a nested dictionary, converting sub-dicts to dataclasses.
        Falls back to from_flat_dict if the structure looks flat (missing 'canvas' key).
        """
        if 'canvas' not in data and 'physics' not in data:
            return cls.from_flat_dict(data)

        def _to_obj(dataclass_type, value):
            if isinstance(value, dict):
                return dataclass_type(**value)
            return value

        # Deep conversion for physics substructures
        physics_data = data.get('physics', {})
        if isinstance(physics_data, dict):
            physics = PhysicsConfig(
                rods=_to_obj(RodsConfig, physics_data.get('rods', {})),
                ghosts=_to_obj(GhostsConfig, physics_data.get('ghosts', {})),
                debris=_to_obj(DebrisConfig, physics_data.get('debris', {})),
                fused=_to_obj(FusedConfig, physics_data.get('fused', {})),
                stage_lambda_range=physics_data.get('stage_lambda_range', (0.1, 10.0))
            )
        else:
            physics = physics_data if isinstance(physics_data, PhysicsConfig) else PhysicsConfig()

        # Deep conversion for sensor substructures
        sensor_data = data.get('sensor', {})
        if isinstance(sensor_data, dict):
            sensor = SensorConfig(
                **{k: v for k, v in sensor_data.items() if k != 'scalebar'},
                scalebar=_to_obj(ScalebarConfig, sensor_data.get('scalebar', {}))
            )
        else:
            sensor = sensor_data if isinstance(sensor_data, SensorConfig) else SensorConfig()

        return cls(
            canvas=_to_obj(CanvasConfig, data.get('canvas', {})),
            physics=physics,
            optics=_to_obj(OpticsConfig, data.get('optics', {})),
            sensor=sensor
        )

    @classmethod
    def from_flat_dict(cls, data: Dict[str, Any]) -> 'SynthConfig':
        """
        Helper to construct SynthConfig from a flat dictionary (legacy support).
        """
        # Create default instances
        canvas = CanvasConfig()
        rods = RodsConfig()
        ghosts = GhostsConfig()
        debris = DebrisConfig()
        fused = FusedConfig()
        physics = PhysicsConfig(rods=rods, ghosts=ghosts, debris=debris, fused=fused)
        optics = OpticsConfig()
        scalebar = ScalebarConfig()
        sensor = SensorConfig(scalebar=scalebar)
        
        # Map fields
        # Canvas
        if 'width' in data: canvas.width = data['width']
        if 'height' in data: canvas.height = data['height']
        if 'parallel_workers' in data: canvas.parallel_workers = data['parallel_workers']
        if 'use_gpu' in data: canvas.use_gpu = data['use_gpu']
        
        # Physics - Rods
        if 'rods_enable' in data: rods.enable = data['rods_enable']
        if 'n_rods_rng_lo_hi' in data: rods.n_rods_rng_lo_hi = data['n_rods_rng_lo_hi']
        if 'rod_len_px_lo_hi' in data: rods.rod_len_px_lo_hi = data['rod_len_px_lo_hi']
        if 'rod_aspect_lo_hi' in data: rods.rod_aspect_lo_hi = data['rod_aspect_lo_hi']
        if 'rod_delta_rng' in data: rods.rod_delta_rng = data['rod_delta_rng']
        
        # Physics - Ghosts
        if 'ghost_enable' in data: ghosts.enable = data['ghost_enable']
        if 'ghost_fraction' in data: ghosts.fraction = data['ghost_fraction']
        if 'ghost_gain_mult' in data: ghosts.gain_mult = data['ghost_gain_mult']
        if 'ghost_blur_sigma' in data: ghosts.blur_sigma = data['ghost_blur_sigma']
        if 'ghost_delta_rng' in data: ghosts.delta_rng = data['ghost_delta_rng']
        if 'ghost_curvature' in data: ghosts.curvature = data['ghost_curvature']
        if 'ghost_width_jit_amp' in data: ghosts.width_jit_amp = data['ghost_width_jit_amp']
        if 'ghost_edge_jit_amp' in data: ghosts.edge_jit_amp = data['ghost_edge_jit_amp']
        if 'ghost_offset_jit_amp' in data: ghosts.offset_jit_amp = data['ghost_offset_jit_amp']
        if 'ghost_curve_kappa_range' in data: ghosts.curve_kappa_range = data['ghost_curve_kappa_range']
        if 'ghost_ragged_p' in data: ghosts.ragged_p = data['ghost_ragged_p']
        if 'ghost_ragged_corr' in data: ghosts.ragged_corr = data['ghost_ragged_corr']
        if 'ghost_mult_mix' in data: ghosts.mult_mix = data['ghost_mult_mix']
        
        # Physics - Debris
        if 'debris_rate' in data: debris.rate = data['debris_rate']
        if 'debris_int_delta' in data: debris.int_delta = data['debris_int_delta']
        if 'debris_dash_prob' in data: debris.dash_prob = data['debris_dash_prob']
        if 'debris_size_px' in data: debris.size_px = data['debris_size_px']
        
        # Physics - Fused
        if 'fused_enable' in data: fused.enable = data['fused_enable']
        if 'fused_p0' in data: fused.p0 = data['fused_p0']
        if 'fused_p1' in data: fused.p1 = data['fused_p1']
        
        # Physics - Global
        if 'stage_lambda_range' in data: physics.stage_lambda_range = data['stage_lambda_range']
        
        # Optics
        if 'taper_strength' in data: optics.taper_strength = data['taper_strength']
        if 'taper_power' in data: optics.taper_power = data['taper_power']
        if 'min_width_ratio' in data: optics.min_width_ratio = data['min_width_ratio']
        if 'cross_soft_sigma' in data: optics.cross_soft_sigma = data['cross_soft_sigma']
        if 'rod_halo_sigma' in data: optics.rod_halo_sigma = data['rod_halo_sigma']
        if 'rod_halo_gain' in data: optics.rod_halo_gain = data['rod_halo_gain']
        if 'shadow_gain' in data: optics.shadow_gain = data['shadow_gain']
        if 'shadow_width_mult' in data: optics.shadow_width_mult = data['shadow_width_mult']
        if 'shadow_bias' in data: optics.shadow_bias = data['shadow_bias']
        if 'shadow_offset_px' in data: optics.shadow_offset_px = data['shadow_offset_px']
        
        # Sensor - Scalebar
        if 'scalebar_enable' in data: scalebar.enable = data['scalebar_enable']
        if 'scalebar_prob' in data: scalebar.prob = data['scalebar_prob']
        if 'scalebar_len_px' in data: scalebar.len_px = data['scalebar_len_px']
        if 'scalebar_thick_px' in data: scalebar.thick_px = data['scalebar_thick_px']
        if 'scalebar_margin_px' in data: scalebar.margin_px = data['scalebar_margin_px']
        if 'scalebar_outline' in data: scalebar.outline = data['scalebar_outline']
        if 'scalebar_font_px' in data: scalebar.font_px = data['scalebar_font_px']
        if 'scalebar_white_jit' in data: scalebar.white_jit = data['scalebar_white_jit']
        if 'scalebar_units' in data: scalebar.units = data['scalebar_units']
        if 'scalebar_value_range' in data: scalebar.value_range = data['scalebar_value_range']
        if 'scalebar_ttf' in data: scalebar.ttf = data['scalebar_ttf']
        
        # Sensor
        if 'bg_gray_range' in data: sensor.bg_gray_range = data['bg_gray_range']
        if 'vignette_strength' in data: sensor.vignette_strength = data['vignette_strength']
        if 'bg_noise_std' in data: sensor.bg_noise_std = data['bg_noise_std']
        if 'tilt_enable' in data: sensor.tilt_enable = data['tilt_enable']
        if 'tilt_dir_deg' in data: sensor.tilt_dir_deg = data['tilt_dir_deg']
        if 'tilt_ptp' in data: sensor.tilt_ptp = data['tilt_ptp']
        if 'tilt_center' in data: sensor.tilt_center = data['tilt_center']
        if 'illum_ampl' in data: sensor.illum_ampl = data['illum_ampl']
        if 'illum_sigma' in data: sensor.illum_sigma = data['illum_sigma']
        if 'relief_field_enable' in data: sensor.relief_field_enable = data['relief_field_enable']
        if 'relief_field_sigma_px' in data: sensor.relief_field_sigma_px = data['relief_field_sigma_px']
        if 'relief_field_gain' in data: sensor.relief_field_gain = data['relief_field_gain']
        if 'relief_field_dir_deg' in data: sensor.relief_field_dir_deg = data['relief_field_dir_deg']
        if 'relief_field_extra_blur' in data: sensor.relief_field_extra_blur = data['relief_field_extra_blur']
        if 'blur_sigma' in data: sensor.blur_sigma = data['blur_sigma']

        return cls(canvas=canvas, physics=physics, optics=optics, sensor=sensor)

def default_config() -> Dict[str, Any]:
    # Return flat dict for now to maintain some compat, 
    # OR return nested dict if we are fully switching.
    # The user wants "re-organized", implying a change.
    # Let's return the nested dict structure via asdict.
    return asdict(SynthConfig())
