import torch
import torch.nn as nn
from typing import Dict, List, Optional
import copy

class RobustnessGuard:
    """
    Ensures optimization stability.
    
    Responsibilities:
    1. Gradient Clipping (Explosion protection).
    2. NaN/Inf Detection (Rollback).
    3. History Tracking (Best State Restoration).
    """
    def __init__(self, params: torch.nn.ParameterDict, clip_norm: float = 1.0):
        self.params = params
        self.clip_norm = clip_norm
        self.best_loss = float('inf')
        self.best_state_dict = None
        self.history = []
        
    def check_gradients(self):
        """
        Clip gradients and check for NaNs.
        Returns True if safe, False if NaNs detected.
        """
        # Check for NaNs before clipping
        for name, param in self.params.items():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"[Guard] NaN/Inf gradient detected in {name}!")
                    return False
                    
        # Clip
        # We need to pass a list of params to clip_grad_norm_
        torch.nn.utils.clip_grad_norm_(self.params.values(), self.clip_norm)
        return True
        
    def save_best(self, loss: float):
        """Save state if loss is best seen so far."""
        if loss < self.best_loss:
            self.best_loss = loss
            # Deep copy the state dict to ensure we own the data
            self.best_state_dict = copy.deepcopy(self.params.state_dict())
            
    def restore_best(self):
        """Restore the best known state."""
        if self.best_state_dict is not None:
            print(f"[Guard] Restoring best state (Loss: {self.best_loss:.4f})")
            self.params.load_state_dict(self.best_state_dict)
            
    def check_rollback(self, loss: float) -> bool:
        """
        Check if loss exploded (e.g. > 10x best loss).
        If so, trigger rollback.
        """
        if loss > self.best_loss * 10.0 and self.best_loss < 1000.0:
            print(f"[Guard] Loss Explosion ({loss:.4f} vs Best {self.best_loss:.4f}). Rolling back.")
            self.restore_best()
            return True
        return False
