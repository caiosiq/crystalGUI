from dataclasses import dataclass, field, asdict
from typing import Tuple, Dict, Any, Optional

@dataclass
class CanvasConfig:
    width: int = 1024
    height: int = 768
    parallel_workers: Optional[int] = None
    use_gpu: bool = False

@dataclass
class ParticlesConfig:
    """
    General configuration for 3D particles (Rods, Plates, Cubes, Spheres).
    Replaces/Extends the old RodsConfig.
    """
    enable: bool = True
    
    # Counts
    n_rods_rng_lo_hi: Tuple[int, int, Optional[int]] = (150, 600, 600) # Keep name for compat
    
    # Dimensions (L, W, H)
    rod_len_px_lo_hi: Tuple[float, float, Optional[float]] = (30.0, 150.0, 150.0) # Length
    rod_aspect_lo_hi: Tuple[float, float, Optional[float]] = (0.02, 0.1, 0.1) # W/L
    thickness_ratio_lo_hi: Tuple[float, float] = (0.1, 1.0) # H/W (New for 3D)
    
    # Optical
    rod_delta_rng: Tuple[float, float, Optional[float]] = (-1.0, -0.2, -0.5) # Phase shift (Increased for visibility)
    birefringence: float = 0.015 # Intrinsic optical anisotropy (Delta n)
    # Shape Distribution (Probabilities)
    # sums to <= 1.0. Remainder is Rods? Or normalize?
    # Default to 100% rods to match old behavior
    prob_plate: float = 0.0
    prob_cube: float = 0.0
    prob_sphere: float = 0.0 
    prob_bubble: float = 0.0
    prob_droplet: float = 0.0
    
    # 3D Orientation
    # If disabled, beta=0, gamma=0 (2D mode)
    enable_3d: bool = False
    
    # Legacy compatibility helper
    def __post_init__(self):
        pass

# Alias for backward compatibility
RodsConfig = ParticlesConfig

@dataclass
class RodSpecs:
    """Specific configuration for Rods"""
    enable: bool = True
    count_range: Tuple[int, int] = (50, 200)
    length_range: Tuple[float, float] = (30.0, 150.0)
    aspect_range: Tuple[float, float] = (0.02, 0.1)
    material: str = "standard"
    birefringence: float = 0.02 # High birefringence for rods
    
    # Physics 2.0
    ragged_p: float = 0.0
    ragged_corr: float = 0.2
    polarity_p: float = 0.0
    shape_mode: str = "straight" # straight, wavy, kink, noisy
    inclusions: float = 0.0
    
    # Phase 4.4.1.5 Texture Pass
    texture_type: str = "none" # none, striated, pitted, granular
    surf_roughness: float = 0.0
    internal_inclusions: float = 0.0
    polarity_flip_p: float = 0.0

@dataclass
class SphereSpecs:
    """Specific configuration for Spheres"""
    enable: bool = False
    count_range: Tuple[int, int] = (10, 50)
    diameter_range: Tuple[float, float] = (20.0, 100.0)
    material: str = "standard"
    
    # Phase 4.4.1.5 Texture Pass
    texture_type: str = "none"
    surf_roughness: float = 0.0
    internal_inclusions: float = 0.0
    polarity_flip_p: float = 0.0

@dataclass
class CubeSpecs:
    """Specific configuration for Cubes"""
    enable: bool = False
    count_range: Tuple[int, int] = (10, 50)
    size_range: Tuple[float, float] = (20.0, 100.0)
    material: str = "standard"
    inclusions: float = 0.0
    
    # Phase 4.4.1.5 Texture Pass
    texture_type: str = "none"
    surf_roughness: float = 0.0
    internal_inclusions: float = 0.0
    polarity_flip_p: float = 0.0

@dataclass
class PlateSpecs:
    """Specific configuration for Plates"""
    enable: bool = False
    count_range: Tuple[int, int] = (10, 50)
    size_range: Tuple[float, float] = (30.0, 150.0)
    aspect_range: Tuple[float, float] = (0.1, 0.8) # W/L
    thickness_range: Tuple[float, float] = (0.05, 0.2) # H/W
    material: str = "standard"
    birefringence: float = 0.005 # Lower birefringence for plates
    
    # Physics 2.0
    ragged_p: float = 0.0
    ragged_corr: float = 0.2
    polarity_p: float = 0.0
    shape_mode: str = "straight"
    
    # Phase 4.4.1.5 Texture Pass
    texture_type: str = "none"
    surf_roughness: float = 0.0
    internal_inclusions: float = 0.0
    polarity_flip_p: float = 0.0

@dataclass
class BubbleSpecs:
    """Specific configuration for Bubbles"""
    enable: bool = False
    count_range: Tuple[int, int] = (5, 20)
    diameter_range: Tuple[float, float] = (10.0, 50.0)
    material: str = "air" # Special material for bubbles
    attach_prob: float = 0.0 # Probability of attaching to a crystal

@dataclass
class DropletSpecs:
    """Specific configuration for Droplets"""
    enable: bool = False
    count_range: Tuple[int, int] = (5, 20)
    diameter_range: Tuple[float, float] = (10.0, 50.0)
    material: str = "oil" # Special material for droplets
    coalesce_enable: bool = True # Enable necking/merging
    coalesce_prob: float = 0.3 # Probability of forming doublets

@dataclass
class GhostsConfig:
    enable: bool = False
    fraction: float = 0.2
    gain_mult: float = 0.5
    blur_sigma: float = 0.0
    delta_range: Tuple[float, float] = (-3.0, 0.0)
    curvature: float = 0.0
    
    # Missing fields for distribution.py
    width_jit_amp: float = 0.1
    offset_jit_amp: float = 0.5
    edge_jit_amp: float = 0.5
    curve_kappa_range: Tuple[float, float] = (0.0, 0.02)
    ragged_p: float = 0.0
    ragged_corr: float = 0.2

@dataclass
class DebrisConfig:
    rate: float = 0.0
    int_delta: Tuple[float, float] = (-6.0, 6.0)
    dash_prob: float = 0.15
    size_px: Tuple[int, int] = (1, 3)

@dataclass
class FusedConfig:
    enable: bool = True
    p0: float = 0.0001 # Spontaneous nucleation prob
    p1: float = 0.003  # Agglomeration prob (parent)
    cluster_weights: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 0.0, 0.0)
    
    # Phase 4
    dlca_enable: bool = False # Recursive fractal growth
    sintering_strength: float = 0.0 # Overlap/Necking strength

@dataclass
class PhysicsConfig:
    rods: ParticlesConfig = field(default_factory=ParticlesConfig) # Main particles (Legacy/General)
    
    # New Specific Configs for Playground
    use_specific_specs: bool = False
    rod_specs: RodSpecs = field(default_factory=RodSpecs)
    sphere_specs: SphereSpecs = field(default_factory=SphereSpecs)
    cube_specs: CubeSpecs = field(default_factory=CubeSpecs)
    plate_specs: PlateSpecs = field(default_factory=PlateSpecs)
    bubble_specs: BubbleSpecs = field(default_factory=BubbleSpecs)
    droplet_specs: DropletSpecs = field(default_factory=DropletSpecs)

    ghosts: GhostsConfig = field(default_factory=GhostsConfig)
    debris: DebrisConfig = field(default_factory=DebrisConfig)
    fused: FusedConfig = field(default_factory=FusedConfig)
    stage_lambda_range: Tuple[float, float] = (0.1, 10.0)
    
    # Phase 4: Dynamics
    flow_enable: bool = False
    flow_direction: float = 0.0 # Degrees
    flow_shear_rate: float = 0.0 # Alignment strength (0-1)
    
    sedimentation_enable: bool = False
    sedimentation_strength: float = 0.0 # Bias towards bottom (0-1)
    size_segregation_enable: bool = False # Brazil Nut Effect (Large particles rise)
    
    def __post_init__(self):
        # Optional: Validation or derived fields can go here
        pass

@dataclass
class OpticsConfig:
    # Mode: "dic", "brightfield", "polarization", "fluorescence", "shadowgraphy", "pvm"
    mode: str = "dic"
    
    # Polarization
    polarizer_angle_deg: float = 90.0 # Crossed polars default
    
    # Laser (PVM)
    laser_wavelength_nm: float = 660.0 # Red laser default
    
    # Rod model optics
    taper_strength: float = 0.45
    taper_power: float = 1.0
    min_width_ratio: float = 0.35
    cross_soft_sigma: float = 0.30
    rod_halo_sigma: float = 3.2
    rod_halo_gain: float = 0.4
    
    # Environment
    medium_refractive_index: float = 1.333 # Water/Buffer default
    
    # Lighting & Focus (Phase 3)
    lighting_angle_deg: float = 45.0 # Direction of light/shear for DIC
    focus_z: float = 0.0 # Focal plane position (-1.0 to 1.0)
    aperture: float = 0.0 # Numerical Aperture (controls DoF strength). 0 = Infinite DoF
    
    # Shadow properties (DIC)
    shadow_gain: Tuple[float, float] = (6.0, 12.0)
    shadow_width_mult: Tuple[float, float] = (0.02, 0.05)
    shadow_bias: Tuple[float, float] = (0.05, 0.12)
    shadow_offset_px: Tuple[float, float] = (0.05, 0.25)
    
    def __post_init__(self):
        # Validate DIC mode parameters if needed
        pass

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
    
    # Fouling & Smudges (Phase 1)
    fouling_enable: bool = False
    fouling_prob: float = 0.3
    fouling_count_range: Tuple[int, int] = (1, 5)
    fouling_sigma_range: Tuple[float, float] = (10.0, 50.0)
    fouling_opacity: float = 0.5
    
    chromatic_aberration_strength: float = 0.0 # Pixel shift for R/B channels
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

    def to_dict(self) -> Dict[str, Any]:
        """Convert the entire configuration to a nested dictionary."""
        return asdict(self)

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
                rods=_to_obj(ParticlesConfig, physics_data.get('rods', {})),
                
                use_specific_specs=physics_data.get('use_specific_specs', False),
                rod_specs=_to_obj(RodSpecs, physics_data.get('rod_specs', {})),
                sphere_specs=_to_obj(SphereSpecs, physics_data.get('sphere_specs', {})),
                cube_specs=_to_obj(CubeSpecs, physics_data.get('cube_specs', {})),
                plate_specs=_to_obj(PlateSpecs, physics_data.get('plate_specs', {})),
                bubble_specs=_to_obj(BubbleSpecs, physics_data.get('bubble_specs', {})),
                droplet_specs=_to_obj(DropletSpecs, physics_data.get('droplet_specs', {})),

                ghosts=_to_obj(GhostsConfig, physics_data.get('ghosts', {})),
                debris=_to_obj(DebrisConfig, physics_data.get('debris', {})),
                fused=_to_obj(FusedConfig, physics_data.get('fused', {})),
                stage_lambda_range=physics_data.get('stage_lambda_range', (0.1, 10.0)),
                
                # Phase 4
                flow_enable=physics_data.get('flow_enable', False),
                flow_direction=physics_data.get('flow_direction', 0.0),
                flow_shear_rate=physics_data.get('flow_shear_rate', 0.0),
                sedimentation_enable=physics_data.get('sedimentation_enable', False),
                sedimentation_strength=physics_data.get('sedimentation_strength', 0.0),
                size_segregation_enable=physics_data.get('size_segregation_enable', False)
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
        rods = ParticlesConfig() # Mapped to ParticlesConfig
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
        
        # Physics - Rods (Now Particles)
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
        if 'ghost_delta_range' in data: ghosts.delta_range = data['ghost_delta_range']
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
        if 'fused_cluster_weights' in data: fused.cluster_weights = data['fused_cluster_weights']
        
        # Physics - Global
        if 'stage_lambda_range' in data: physics.stage_lambda_range = data['stage_lambda_range']
        
        # Physics - Phase 4
        if 'flow_enable' in data: physics.flow_enable = data['flow_enable']
        if 'flow_direction' in data: physics.flow_direction = data['flow_direction']
        if 'flow_shear_rate' in data: physics.flow_shear_rate = data['flow_shear_rate']
        if 'sedimentation_enable' in data: physics.sedimentation_enable = data['sedimentation_enable']
        if 'sedimentation_strength' in data: physics.sedimentation_strength = data['sedimentation_strength']
        if 'size_segregation_enable' in data: physics.size_segregation_enable = data['size_segregation_enable']
        
        # Optics
        if 'taper_strength' in data: optics.taper_strength = data['taper_strength']
        if 'taper_power' in data: optics.taper_power = data['taper_power']
        if 'min_width_ratio' in data: optics.min_width_ratio = data['min_width_ratio']
        if 'cross_soft_sigma' in data: optics.cross_soft_sigma = data['cross_soft_sigma']
        if 'rod_halo_sigma' in data: optics.rod_halo_sigma = data['rod_halo_sigma']
        if 'rod_halo_gain' in data: optics.rod_halo_gain = data['rod_halo_gain']
        if 'medium_refractive_index' in data: optics.medium_refractive_index = data['medium_refractive_index']
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
        
        # Fouling
        if 'fouling_enable' in data: sensor.fouling_enable = data['fouling_enable']
        if 'fouling_prob' in data: sensor.fouling_prob = data['fouling_prob']
        if 'fouling_count_range' in data: sensor.fouling_count_range = data['fouling_count_range']
        if 'fouling_sigma_range' in data: sensor.fouling_sigma_range = data['fouling_sigma_range']
        if 'fouling_opacity' in data: sensor.fouling_opacity = data['fouling_opacity']
        
        if 'blur_sigma' in data: sensor.blur_sigma = data['blur_sigma']

        return cls(canvas=canvas, physics=physics, optics=optics, sensor=sensor)

    @classmethod
    def default_config(cls) -> Dict[str, Any]:
        return asdict(cls())

def default_config() -> Dict[str, Any]:
    return asdict(SynthConfig())
