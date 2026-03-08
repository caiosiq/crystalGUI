
import torch
import torch.nn as nn

class LossBalancer(nn.Module):
    def __init__(self, names, weights=None, dynamic_scaling=False):
        """
        Balances multiple loss terms to ensure they are on the same scale.
        
        Args:
            names (list[str]): Names of the loss terms (e.g. ['vgg', 'spectral']).
            weights (dict[str, float]): Initial manual weights. If None, defaults to 1.0.
            dynamic_scaling (bool): If True, automatically scales losses to be ~1.0 using a running average.
                                    (Use with caution: Can cause instability if one loss spikes).
        """
        super().__init__()
        self.names = names
        self.dynamic = dynamic_scaling
        
        # Manual weights
        if weights is None:
            self.weights = {n: 1.0 for n in names}
        else:
            self.weights = weights
            
        # Running statistics for dynamic scaling
        # Store running mean as a buffer (persistent state but not parameter)
        self.register_buffer('running_means', torch.ones(len(names)))
        self.momentum = 0.05
        self.warmup_steps = 100
        self.step_count = 0
        
    def forward(self, losses):
        """
        Args:
            losses (dict[str, torch.Tensor]): Dictionary of scalar loss terms.
            
        Returns:
            total_loss (torch.Tensor): Weighted sum.
            log_dict (dict[str, float]): Detached values for logging.
        """
        total_loss = 0.0
        log_dict = {}
        
        for i, name in enumerate(self.names):
            if name not in losses:
                continue
                
            val = losses[name]
            # Ensure scalar
            if val.dim() > 0:
                val = val.mean()
            
            current_scale = 1.0
            
            # Update running mean if dynamic
            if self.dynamic:
                with torch.no_grad():
                    # Update running mean (Exponential Moving Average)
                    current_val = val.item()
                    if self.step_count < self.warmup_steps:
                        # During warmup, average quickly
                        alpha = 0.5
                    else:
                        alpha = self.momentum
                        
                    self.running_means[i] = (1 - alpha) * self.running_means[i] + alpha * current_val
                
                # Dynamic scale: normalize so mean loss is ~1.0
                # Scale = 1.0 / mean
                mean_val = self.running_means[i]
                if mean_val > 1e-6:
                    current_scale = 1.0 / mean_val
                else:
                    current_scale = 1.0
                
                # Clamp extreme scaling factors for stability
                # Use item() to get float, avoiding tensor copy warning
                current_scale = torch.clamp(torch.as_tensor(current_scale), 0.001, 1000.0).item()
            
            # Apply manual weight * dynamic scale
            w = self.weights.get(name, 1.0)
            weighted_val = val * w * current_scale
            
            total_loss = total_loss + weighted_val
            
            # Logging
            log_dict[f"loss/{name}_raw"] = val.item()
            log_dict[f"loss/{name}_weighted"] = weighted_val.item()
            if self.dynamic:
                log_dict[f"loss/{name}_scale"] = current_scale
            
        self.step_count += 1
        return total_loss, log_dict
