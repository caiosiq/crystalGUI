import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple
from osog.core.pipeline import Pipeline
from osog.config import SynthConfig

class DiffOSOG(nn.Module):
    """
    Differentiable Wrapper for OSOG Pipeline.
    Allows optimizing selected parameters via Gradient Descent.
    """
    def __init__(self, config: SynthConfig, active_params: List[str]):
        super().__init__()
        self.cfg = config
        self.pipeline = Pipeline(self.cfg)
        self.active_params = active_params
        
        # Parameter Mapping: Name -> (Section, Attribute, [Index])
        self.param_map = {
            # Sensor
            'blur_sigma': ('sensor', 'blur_sigma'),
            'noise_scale': ('sensor', 'bg_noise_std'),
            'fouling_opacity': ('sensor', 'fouling_opacity'),
            'distractor_opacity': ('sensor', 'distractor_opacity'),
            'distractor_anisotropy': ('sensor', 'distractor_anisotropy'),
            
            # Optics
            'focus_z': ('optics', 'focus_z'),
            'aperture': ('optics', 'aperture'),
            'chromatic_aberration': ('sensor', 'chromatic_aberration_strength'),
            
            # Lighting (Tuples need special handling)
            # We map specific indices of tuples
            'shadow_gain_0': ('optics', 'shadow_gain', 0),
            'shadow_gain_1': ('optics', 'shadow_gain', 1),
            'light_dir_x': ('optics', 'light_direction', 0),
            'light_dir_y': ('optics', 'light_direction', 1),
            'light_dir_z': ('optics', 'light_direction', 2),
        }

        # Register Parameters
        for param_name in active_params:
            if param_name not in self.param_map:
                print(f"[DiffOSOG] Warning: Parameter '{param_name}' is not supported yet.")
                continue
                
            loc = self.param_map[param_name]
            section_name = loc[0]
            attr_name = loc[1]
            
            section = getattr(self.cfg, section_name)
            original_value = getattr(section, attr_name)
            
            if len(loc) > 2:
                # Tuple Value
                idx = loc[2]
                val = original_value[idx]
            else:
                # Scalar Value
                val = original_value
                
            # Create Parameter
            # Ensure we start with a valid float
            val_float = float(val)
            param = nn.Parameter(torch.tensor(val_float, dtype=torch.float32))
            self.register_parameter(param_name, param)
            print(f"[DiffOSOG] Registered parameter: {param_name} = {val_float}")

    def forward(self, seed: Optional[int] = None):
        """
        Run the differentiable pipeline.
        Returns: (3, H, W) Tensor
        """
        # 1. Inject Parameters into Config
        for param_name in self.active_params:
            if param_name not in self.param_map: continue
            
            # Get the registered parameter (Tensor with grad)
            param_tensor = getattr(self, param_name)
            
            loc = self.param_map[param_name]
            section = getattr(self.cfg, loc[0])
            attr_name = loc[1]
            
            if len(loc) > 2:
                # Tuple update
                idx = loc[2]
                original_tuple = getattr(section, attr_name)
                # Convert tuple to list to modify
                val_list = list(original_tuple)
                # Replace value with Tensor
                val_list[idx] = param_tensor
                # Convert back to tuple (now containing a Tensor!)
                setattr(section, attr_name, tuple(val_list))
            else:
                # Scalar update
                setattr(section, attr_name, param_tensor)

        # 2. Run Pipeline (returning Tensor)
        # Note: We use t=0.0 as default time
        output_tensor = self.pipeline.generate(t=0.0, seed=seed, return_tensor=True, differentiable=True)
        
        # Normalize to 0-1 for Loss calculation?
        # OSOG returns 0-255 float tensor.
        # VGG expects normalized inputs usually.
        # But for now, we just return the raw 0-255 tensor.
        
        return output_tensor
