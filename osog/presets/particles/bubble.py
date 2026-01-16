
from ..base import ComponentPreset, CannyParam

class BubblePreset(ComponentPreset):
    """
    Defines the 'Safe Rendering Zone' for Bubbles.
    Bubbles have sharp refractive rims.
    """
    def get_canny_constraints(self):
        return {
            # GEOMETRY
            "diameter_range": CannyParam(default=(10.0, 50.0), hard_min=2.0, hard_max=200.0),
            
            # OPTICS
            # Bubbles are usually negative delta (lower index than water?) or just strong refraction
            "delta_range": CannyParam(default=(-12.0, -5.0), hard_min=-30.0, hard_max=0.0),
            
            "count_range": CannyParam(default=(5, 20), hard_min=0, hard_max=500)
        }
