
from ..base import ComponentPreset, CannyParam

class DICPreset(ComponentPreset):
    """
    Defines 'Safe' parameters for Differential Interference Contrast.
    """
    def get_canny_constraints(self):
        return {
            # Shadow gain determines contrast.
            # Too high (>50) = binary black/white, loss of detail.
            # Too low (<1) = invisible.
            "shadow_gain": CannyParam(default=(6.0, 12.0), hard_min=1.0, hard_max=40.0),
            
            # Shear/Offset (shadow_offset_px)
            # Should be small relative to particle size.
            "shadow_offset_px": CannyParam(default=(0.05, 0.25), hard_min=0.0, hard_max=2.0),
            
            # Bias (Background gray level shift?)
            "shadow_bias": CannyParam(default=(0.05, 0.12), hard_min=0.0, hard_max=0.5)
        }
