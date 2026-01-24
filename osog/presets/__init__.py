
# Expose presets
from .particles.rod import RodPreset
from .particles.sphere import SpherePreset
from .particles.bubble import BubblePreset
from .optics.dic import DICPreset

# Expose full configurations
from .defaults import PRESETS, load_preset, get_optimization_bounds

# Registry for easy access
COMPONENT_PRESETS = {
    "rod": RodPreset(),
    "sphere": SpherePreset(),
    "bubble": BubblePreset(),
    "dic": DICPreset()
}
