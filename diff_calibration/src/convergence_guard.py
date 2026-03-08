import torch
import numpy as np
from collections import deque
from typing import Dict, List

class ConvergenceGuard:
    """
    Monitors Parameter-Space Velocity to detect true convergence.
    Ignores loss noise.
    """
    def __init__(self, window_size: int = 20, velocity_threshold: float = 1e-4):
        self.window_size = window_size
        self.threshold = velocity_threshold
        
        # History: {param_name: deque([val1, val2, ...])}
        # We store NORMALIZED (0-1) values to have consistent velocity units
        self.history = {}
        
    def update(self, param_manager, active_params: List[str]) -> Dict[str, float]:
        """
        Record current parameter state for ACTIVE parameters only.
        Returns current velocity per parameter.
        """
        velocities = {}
        
        # Access latent params directly to get normalized values
        with torch.no_grad():
            for name in active_params:
                # Convert real name -> safe name for lookup
                safe_name = name.replace('.', '_')
                
                if safe_name not in param_manager.latent_params:
                    continue
                    
                param = param_manager.latent_params[safe_name]
                
                # Convert latent (unbounded) -> normalized (0-1)
                norm_val = torch.sigmoid(param).item()
                
                if safe_name not in self.history:
                    self.history[safe_name] = deque(maxlen=self.window_size)
                    
                self.history[safe_name].append(norm_val)
                
                # Calculate Velocity (Standard Deviation of the window)
                # If param is moving, std > 0. If stuck/converged, std -> 0.
                if len(self.history[safe_name]) >= 5:
                    vel = np.std(self.history[safe_name])
                    velocities[safe_name] = vel
                else:
                    velocities[safe_name] = 1.0 # Assume moving initially
                    
        return velocities
        
    def reset(self):
        """Clear history. Call this when advancing stages!"""
        self.history.clear()
        
    def check_convergence(self) -> bool:
        """
        Returns True if ALL tracked parameters have stopped moving.
        """
        if not self.history:
            return False
            
        converged_count = 0
        total_count = 0
        
        for name, values in self.history.items():
            if len(values) < self.window_size:
                return False # Not enough data yet
                
            vel = np.std(values)
            
            # Check if velocity is below threshold
            if vel < self.threshold:
                converged_count += 1
            total_count += 1
            
        # If everything is static, we are done
        return converged_count == total_count
