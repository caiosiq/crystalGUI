
# Expose presets
from .particles.rod import RodPreset
from .particles.sphere import SpherePreset
from .particles.bubble import BubblePreset
from .optics.dic import DICPreset

# Registry for easy access
COMPONENT_PRESETS = {
    "rod": RodPreset(),
    "sphere": SpherePreset(),
    "bubble": BubblePreset(),
    "dic": DICPreset()
}
