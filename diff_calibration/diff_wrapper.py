
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple
from osog.core.pipeline import Pipeline
from osog.config import SynthConfig

def get_nested_attr(obj, attr_path):
    """
    Helper to get attribute from nested object using dot notation.
    e.g. 'physics.rod_specs' -> obj.physics.rod_specs
    """
    parts = attr_path.split('.')
    current = obj
    for part in parts:
        current = getattr(current, part)
    return current

def set_nested_attr(obj, attr_path, value):
    """
    Helper to set attribute from nested object using dot notation.
    """
    parts = attr_path.split('.')
    current = obj
    # Traverse to parent
    for part in parts[:-1]:
        current = getattr(current, part)
    # Set on parent
    setattr(current, parts[-1], value)

class DiffOSOG(nn.Module):
    """
    Differentiable Wrapper for OSOG Pipeline.
    Allows optimizing selected parameters via Gradient Descent.
    """
    def __init__(self, config: SynthConfig, device: str = "cpu", batch_size: int = 1):
        super().__init__()
        self.cfg = config
        self.device = torch.device(device)
        self.batch_size = batch_size
        
        # Pass device to pipeline if supported, otherwise assume it uses config or defaults
        # Pipeline constructor in osog/core/pipeline.py doesn't accept device arg
        # It determines device from config.canvas.use_gpu and torch.cuda.is_available()
        # We need to ensure config matches requested device
        if device == 'cuda':
            self.cfg.canvas.use_gpu = True
        else:
            self.cfg.canvas.use_gpu = False
            
        self.pipeline = Pipeline(self.cfg) 
        self.active_params = [] # Will be populated by register_active_params
        
        # Parameter Mapping: Name -> (Nested Attribute Path, [Index])
        # If Index is present, it targets a tuple/list element.
        self.param_map = {
            # --- Sensor ---
            'blur_sigma': ('sensor.blur_sigma',),
            'noise_scale': ('sensor.bg_noise_std',),
            'fouling_opacity': ('sensor.fouling_opacity',),
            'distractor_opacity': ('sensor.distractor_opacity',),
            'distractor_anisotropy': ('sensor.distractor_anisotropy',),
            'chromatic_aberration': ('sensor.chromatic_aberration_strength',),
            
            # --- Optics ---
            'optics.blur_sigma': ('optics.blur_sigma',), # Alias
            'optics.noise_scale': ('optics.noise_scale',),
            'optics.shadow_gain': ('optics.shadow_gain',), # Can be scalar or tuple
            
            'focus_z': ('optics.focus_z',),
            'aperture': ('optics.aperture',),
            
            # Lighting (Tuples)
            'shadow_gain_0': ('optics.shadow_gain', 0),
            'shadow_gain_1': ('optics.shadow_gain', 1),
            'light_dir_x': ('optics.light_direction', 0),
            'light_dir_y': ('optics.light_direction', 1),
            'light_dir_z': ('optics.light_direction', 2),
            
            # --- Physics: Rods ---
            'rod_length_min': ('physics.rod_specs.length_range', 0),
            'rod_length_max': ('physics.rod_specs.length_range', 1),
            'rod_aspect_min': ('physics.rod_specs.aspect_range', 0),
            'rod_aspect_max': ('physics.rod_specs.aspect_range', 1),
            'rod_width_jit': ('physics.rod_specs.width_jit_amp',),
            'rod_edge_jit': ('physics.rod_specs.edge_jit_amp',),
            'rod_offset_jit': ('physics.rod_specs.offset_jit_amp',),
            'rod_raggedness': ('physics.rod_specs.ragged_p',),
            
            # --- Physics: Spheres ---
            'sphere_diameter_min': ('physics.sphere_specs.diameter_range', 0),
            'sphere_diameter_max': ('physics.sphere_specs.diameter_range', 1),
            
            # --- Physics: Cubes ---
            'cube_size_min': ('physics.cube_specs.size_range', 0),
            'cube_size_max': ('physics.cube_specs.size_range', 1),
        }

    def register_active_params(self, param_names: List[str]):
        """
        Register parameters to be optimized.
        """
        self.active_params = param_names
        
        for param_name in param_names:
            # Handle direct dot notation if not in map
            if param_name not in self.param_map:
                # Assume it's a direct path
                # e.g. 'optics.blur_sigma' -> ('optics.blur_sigma',)
                self.param_map[param_name] = (param_name,)
                
            loc = self.param_map[param_name]
            attr_path = loc[0]
            
            # Get current value
            try:
                parent_obj_val = get_nested_attr(self.cfg, attr_path)
            except AttributeError:
                print(f"[DiffOSOG] Error: Path '{attr_path}' not found in config.")
                continue
            
            if len(loc) > 1:
                # Tuple Value
                idx = loc[1]
                val = parent_obj_val[idx]
            else:
                # Scalar Value
                val = parent_obj_val
                
            # Create Parameter
            # Ensure we start with a valid float
            try:
                # Handle tuples that are registered as scalars (e.g. shadow_gain=(1.2, 1.2))
                if isinstance(val, (tuple, list)):
                    val_float = float(val[0])
                else:
                    val_float = float(val)
            except (ValueError, TypeError):
                 if val is None:
                     print(f"[DiffOSOG] Warning: Parameter '{param_name}' is None. Skipping.")
                     continue
                 raise
                 
            # Register as nn.Parameter
            # Use dot notation for name to avoid conflicts? No, use passed name.
            # Convert dots to underscores for PyTorch parameter naming conventions if needed
            safe_name = param_name.replace('.', '_')
            
            param = nn.Parameter(torch.tensor(val_float, dtype=torch.float32, device=self.device))
            self.register_parameter(safe_name, param)
            print(f"[DiffOSOG] Registered parameter: {safe_name} = {val_float}")

    def forward(self, seed: Optional[int] = None):
        """
        Run the differentiable pipeline.
        Returns: (B, 3, H, W) Tensor, where B is self.batch_size
        """
        # 1. Inject Parameters into Config
        for param_name in self.active_params:
            # Get the registered parameter
            safe_name = param_name.replace('.', '_')
            if not hasattr(self, safe_name): continue
            
            param_tensor = getattr(self, safe_name)
            
            # Find where it goes
            if param_name in self.param_map:
                loc = self.param_map[param_name]
            else:
                loc = (param_name,)
                
            attr_path = loc[0]
            
            if len(loc) > 1:
                # Tuple update (targeting specific index)
                idx = loc[1]
                # We need to reconstruct the tuple
                # This is tricky because we can't modify config in-place easily if it's a tuple
                # We need to get the current tuple
                current_val = get_nested_attr(self.cfg, attr_path)
                if isinstance(current_val, (tuple, list)):
                    val_list = list(current_val)
                    val_list[idx] = param_tensor
                    set_nested_attr(self.cfg, attr_path, tuple(val_list))
            else:
                # Scalar update OR Tuple broadcast
                # If target is a tuple but we optimize a scalar, broadcast it?
                # E.g. shadow_gain is (1.2, 1.2). We optimize scalar 'g'.
                # We should set it to (g, g) or just g if supported.
                # OSOG config usually expects tuples for ranges.
                # If param is shadow_gain, it might expect a tuple.
                
                # Check target type
                current_val = get_nested_attr(self.cfg, attr_path)
                if isinstance(current_val, (tuple, list)) and param_tensor.numel() == 1:
                     # Broadcast scalar to tuple
                     # This assumes homogeneous tuple (e.g. gain_min, gain_max)
                     # If we want to optimize them together
                     new_val = tuple([param_tensor for _ in current_val])
                     set_nested_attr(self.cfg, attr_path, new_val)
                else:
                     set_nested_attr(self.cfg, attr_path, param_tensor)

        # 2. Run Pipeline (returning Tensor)
        # Generate batch_size images
        images = []
        for i in range(self.batch_size):
            # If seed is provided, offset it to ensure diversity across batch
            # If seed is None, Pipeline generates random seed
            s = seed + i if seed is not None else None
            
            output_tensor = self.pipeline.generate(t=0.0, seed=s, return_tensor=True, differentiable=True, soft_mode=True, no_detail=True)
            
            # Ensure (1, 3, H, W)
            if output_tensor.dim() == 3:
                output_tensor = output_tensor.unsqueeze(0)
            
            images.append(output_tensor)
            
        # Stack into (B, 3, H, W)
        return torch.cat(images, dim=0)
