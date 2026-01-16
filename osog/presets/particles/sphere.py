
from ..base import ComponentPreset, CannyParam

class SpherePreset(ComponentPreset):
    """
    Defines the 'Safe Rendering Zone' for Spheres.
    """
    def get_canny_constraints(self):
        return {
            # GEOMETRY
            "diameter_range": CannyParam(default=(20.0, 80.0), hard_min=5.0, hard_max=300.0),
            
            # OPTICS
            "delta_range": CannyParam(default=(-12.0, 0.0), hard_min=-20.0, hard_max=20.0),
            
            "count_range": CannyParam(default=(10, 50), hard_min=0, hard_max=1000)
        }
