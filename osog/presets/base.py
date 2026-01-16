
from dataclasses import dataclass, field
from typing import Any, Optional, Union, Dict, List
import copy

@dataclass
class CannyParam:
    """
    Defines a parameter's 'Safe Zone'.
    default: The starting value (center of the canny region).
    hard_min: The absolute technical limit (below this, rendering breaks).
    hard_max: The absolute technical limit (above this, rendering breaks).
    """
    default: Any
    hard_min: Optional[float] = None
    hard_max: Optional[float] = None
    
    def validate(self, value):
        """Ensures a Config doesn't violate the Canny Region."""
        # Handle tuple values (like range tuples) by validating the bounds
        if isinstance(value, (list, tuple)) and isinstance(value[0], (int, float)):
             valid_vals = []
             for v in value:
                 if self.hard_min is not None and v < self.hard_min:
                     print(f"WARNING: Value {v} is below Canny limit {self.hard_min}. Clamping.")
                     valid_vals.append(self.hard_min)
                 elif self.hard_max is not None and v > self.hard_max:
                     print(f"WARNING: Value {v} is above Canny limit {self.hard_max}. Clamping.")
                     valid_vals.append(self.hard_max)
                 else:
                     valid_vals.append(v)
             return type(value)(valid_vals)

        if isinstance(value, (int, float)):
            if self.hard_min is not None and value < self.hard_min:
                print(f"WARNING: Value {value} is below Canny limit {self.hard_min}. Rendering artifacts may occur.")
                return self.hard_min
            if self.hard_max is not None and value > self.hard_max:
                print(f"WARNING: Value {value} is above Canny limit {self.hard_max}. Rendering artifacts may occur.")
                return self.hard_max
        return value

# Helper to define a parameter with constraints (used in Config Overrides)
Param = lambda val, min_val=None, max_val=None: {"val": val, "min": min_val, "max": max_val}

@dataclass
class ComponentPreset:
    """Base class for any component (Rod, Sphere, Optics) that needs constraints."""
    
    def get_canny_constraints(self) -> Dict[str, CannyParam]:
        """Returns the dictionary of safe zones for this component."""
        raise NotImplementedError
    
    def apply_defaults(self, config_dict: dict):
        """Applies safe defaults to a config dictionary if keys are missing."""
        constraints = self.get_canny_constraints()
        for key, param in constraints.items():
            if key not in config_dict:
                config_dict[key] = param.default

@dataclass
class OSOGPreset:
    name: str
    description: str
    
    # We use a nested dictionary structure that mirrors SynthConfig
    # Keys missing from here will simply be left alone (Partial Application)
    config_overrides: Dict[str, Any] = field(default_factory=dict)

    def apply(self, current_config):
        """
        Merges this preset into a SynthConfig object.
        """
        # Deep copy to avoid mutating the original preset storage
        cfg_dict = current_config.to_dict() if hasattr(current_config, 'to_dict') else copy.deepcopy(current_config.__dict__)
        
        # If current_config is a dataclass, we might need to be careful about asdict/to_dict behavior
        # Assuming SynthConfig has a way to get a dict or is a dataclass
        if not isinstance(cfg_dict, dict):
            # Fallback for dataclasses
            from dataclasses import asdict
            cfg_dict = asdict(current_config)

        self._recursive_update(cfg_dict, self.config_overrides)
        
        # If config is a Pydantic/Dataclass, reload it
        if hasattr(current_config, 'from_dict'):
            return current_config.__class__.from_dict(cfg_dict)
        elif hasattr(current_config, '__class__'):
             # Special case for PhysicsConfig which is nested dataclass
             try:
                return current_config.__class__(**cfg_dict)
             except Exception:
                return cfg_dict
             
        return cfg_dict # Or generic object wrapper

    def _recursive_update(self, target: dict, source: dict):
        for key, value in source.items():
            if isinstance(value, dict) and "val" not in value:
                # It's a nested category (e.g., 'particles')
                
                # Check if target[key] is a dict
                if key not in target:
                    target[key] = {}
                
                # FIX: We must ensure we are working with a pure dict tree.
                if not isinstance(target[key], dict) and hasattr(target[key], '__dataclass_fields__'):
                     from dataclasses import asdict
                     target[key] = asdict(target[key])
                     
                self._recursive_update(target[key], value)
            else:
                # It's a parameter leaf node
                # Check if it's our special Param format or just a raw value
                if isinstance(value, dict) and "val" in value:
                    target[key] = value["val"] # For Rendering, we just want the value
                else:
                    target[key] = value
