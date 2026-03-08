import torch
import torch.nn as nn
import json
import os
import math
from typing import Dict, List, Tuple, Any, Optional

class ParameterManager(nn.Module):
    """
    The Brain of the Calibration Engine.
    
    Responsibilities:
    1. Manages Bounded Latent Space (Optimizer sees unconstrained reals, Physics sees bounded values).
    2. Enforces Optimization Rules (from JSON).
    3. Handles Normalization (All gradients operate in 0-1 space).
    """
    def __init__(self, diff_model, rules_path: str = None):
        super().__init__()
        self.diff_model = diff_model # Reference to DiffOSOG
        
        # Load Rules
        if rules_path is None:
            # Default to adjacent file (now in parent dir relative to src)
            rules_path = os.path.join(os.path.dirname(__file__), "../optimization_rules.json")
            
        with open(rules_path, 'r') as f:
            self.rules = json.load(f)
            
        # Latent Parameters (The ones optimizer actually sees)
        # We store them in a ParameterDict for easy access
        self.latent_params = nn.ParameterDict()
        
        # Metadata for mapping
        self.active_param_names = []
        
    def init_parameters(self, param_names: List[str]):
        """
        Initialize specific parameters for optimization.
        Values are pulled from the current DiffOSOG config state.
        """
        self.active_param_names = []
        self.latent_params.clear()
    
        
        for name in param_names:
            if name not in self.rules:
                print(f"[ParamManager] Warning: No rules found for '{name}'. Skipping.")
                continue
                
            rule = self.rules[name]
            bounds = rule['bounds']
            scale_type = rule.get('scale', 'linear')
            
            # 1. Get Current Physical Value from DiffOSOG Config
            # We use the param_map logic from DiffOSOG to find the value
            # NOTE: We must check self.diff_model.param_map. 
            # If the parameter is not there, we assume it is a direct attribute path.
            if name in self.diff_model.param_map:
                loc = self.diff_model.param_map[name]
            else:
                # Assume direct path
                loc = (name,)
                
            attr_path = loc[0]
            idx = loc[1] if len(loc) > 1 else None
            
            # Helper to traverse config
            try:
                val = self._get_config_value(self.diff_model.cfg, attr_path, idx)
            except AttributeError:
                print(f"[ParamManager] Error: Path '{attr_path}' not found in config. Skipping.")
                continue
            
            # 2. Inverse Map: Physical -> Latent (Unbounded)
            # Physical = min + (max-min) * sigmoid(latent)
            # sigmoid(latent) = (Physical - min) / (max - min)
            # latent = logit(norm_val)
            
            phys_min, phys_max = bounds
            
            if scale_type == 'log':
                # For log scale, we work in log domain
                # val -> log(val)
                # bounds -> log(min), log(max)
                val = math.log(max(1e-6, val))
                phys_min = math.log(max(1e-6, phys_min))
                phys_max = math.log(max(1e-6, phys_max))
            
            # Normalize to 0-1
            norm_val = (val - phys_min) / (phys_max - phys_min + 1e-9)
            norm_val = max(0.01, min(0.99, norm_val)) # Clamp to avoid inf in logit
            
            # Inverse Sigmoid (Logit)
            latent_val = math.log(norm_val / (1.0 - norm_val))
            
            # Create Parameter
            # Replace dots with underscores for ParameterDict keys
            safe_name = name.replace('.', '_')
            self.latent_params[safe_name] = nn.Parameter(torch.tensor(float(latent_val)))
            self.active_param_names.append(name)
            
            print(f"[ParamManager] Initialized '{name}': Phys={val:.4f} -> Norm={norm_val:.4f} -> Latent={latent_val:.4f}")

    def update_model_config(self):
        """
        Push current latent parameters -> physical values -> DiffOSOG Config.
        Must be called before diff_model() forward pass.
        """
        for name in self.active_param_names:
            safe_name = name.replace('.', '_')
            latent = self.latent_params[safe_name]
            rule = self.rules[name]
            bounds = rule['bounds']
            scale_type = rule.get('scale', 'linear')
            
            # 1. Latent -> Normalized (0-1)
            norm = torch.sigmoid(latent)
            
            # 2. Normalized -> Physical
            phys_min, phys_max = bounds
            
            if scale_type == 'log':
                # Log domain interpolation
                # val = exp( log_min + norm * (log_max - log_min) )
                log_min = math.log(max(1e-6, phys_min))
                log_max = math.log(max(1e-6, phys_max))
                phys_val = torch.exp(log_min + norm * (log_max - log_min))
            else:
                # Linear domain
                phys_val = phys_min + norm * (phys_max - phys_min)
                
            # 3. Push to DiffOSOG (This is the "Injection" step)
            # We bypass DiffOSOG.forward's internal injection and do it here directly
            # or we set the attributes on DiffOSOG's config
            
            if name in self.diff_model.param_map:
                loc = self.diff_model.param_map[name]
            else:
                loc = (name,)
                
            attr_path = loc[0]
            idx = loc[1] if len(loc) > 1 else None
            
            self._set_config_value(self.diff_model.cfg, attr_path, idx, phys_val)

    def get_parameter_groups(self, base_lr: float = 0.05) -> List[Dict]:
        """
        Groups parameters by their learning rate multipliers for the optimizer.
        Returns: List of dicts suitable for optim.Adam(params=...)
        """
        groups = {}
        
        for name in self.active_param_names:
            safe_name = name.replace('.', '_')
            param = self.latent_params[safe_name]
            rule = self.rules[name]
            
            lr_mult = rule.get('lr_mult', 1.0)
            target_lr = base_lr * lr_mult
            
            if target_lr not in groups:
                groups[target_lr] = []
            groups[target_lr].append(param)
            
        # Convert to optimizer format
        optim_params = []
        for lr, params in groups.items():
            optim_params.append({'params': params, 'lr': lr})
            
        return optim_params
    def get_physical_values(self) -> Dict[str, float]:
        """Return current physical values for logging."""
        vals = {}
        with torch.no_grad():
            for name in self.active_param_names:
                safe_name = name.replace('.', '_')
                latent = self.latent_params[safe_name]
                rule = self.rules[name]
                bounds = rule['bounds']
                scale_type = rule.get('scale', 'linear')
                
                norm = torch.sigmoid(latent)
                phys_min, phys_max = bounds
                
                if scale_type == 'log':
                    log_min = math.log(max(1e-6, phys_min))
                    log_max = math.log(max(1e-6, phys_max))
                    phys_val = math.exp(log_min + norm.item() * (log_max - log_min))
                else:
                    phys_val = phys_min + norm.item() * (phys_max - phys_min)
                
                vals[name] = phys_val
        return vals

    # --- Helpers ---
    def _get_config_value(self, cfg, path, idx=None):
        parts = path.split('.')
        curr = cfg
        for p in parts:
            curr = getattr(curr, p)
        if idx is not None:
            return curr[idx]
        return curr

    def _set_config_value(self, cfg, path, idx, value):
        parts = path.split('.')
        curr = cfg
        # Traverse to parent
        for p in parts[:-1]:
            curr = getattr(curr, p)
        
        final_attr = parts[-1]
        
        if idx is not None:
            # Tuple update
            # Get current tuple
            current_tuple = getattr(curr, final_attr)
            # Convert to list, update, convert back
            val_list = list(current_tuple)
            val_list[idx] = value
            setattr(curr, final_attr, tuple(val_list))
        else:
            # Scalar update
            setattr(curr, final_attr, value)
