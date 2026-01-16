
from ..base import ComponentPreset, CannyParam

class RodPreset(ComponentPreset):
    """
    Defines the 'Safe Rendering Zone' for Rod particles.
    If you go outside these bounds, the shader looks fake (plastic tubes).
    """
    def get_canny_constraints(self):
        return {
            # GEOMETRY
            # Rods bigger than 380px (original config max) start to look odd, 
            # but user mentioned 500px is "weird".
            # Let's set safe max around 500px now that we have texture scaling.
            # But "Canny" region is where it looks *good*.
            "length_range": CannyParam(default=(30.0, 150.0), hard_min=10.0, hard_max=800.0),
            
            # Aspect Ratio: Width/Length
            # If too thin (<0.02), aliasing.
            # If too fat (>0.5), it's a plate/rectangle, not a rod.
            "aspect_range": CannyParam(default=(0.02, 0.1), hard_min=0.01, hard_max=0.6),
            
            # TEXTURE
            # Raggedness needs to be small to look like crystal defects
            "ragged_p":   CannyParam(default=0.0,  hard_min=0.0,  hard_max=1.0),
            
            # OPTICS (Usually in optics config, but if specified here)
            # RodSpecs has delta_range
            "delta_range": CannyParam(default=(-12.0, 0.0), hard_min=-20.0, hard_max=20.0),
            
            "count_range": CannyParam(default=(10, 50), hard_min=0, hard_max=1000)
        }
