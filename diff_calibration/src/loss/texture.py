import torch
import torch.nn as nn
import torch.nn.functional as F
from diff_calibration.src.loss.perceptual import VGGPerceptualLoss

class GramMatrixLoss(nn.Module):
    """
    Computes the Style Loss using Gram Matrices from VGG features.
    This is effectively a wrapper around VGGPerceptualLoss with use_gram=True.
    
    Why use this? 
    - It ignores spatial position (good for stochastic texture).
    - It captures correlations between feature maps (e.g. "graininess").
    """
    def __init__(self, device='cpu'):
        super().__init__()
        # Initialize VGG with Gram Matrix mode enabled
        self.vgg = VGGPerceptualLoss(use_gram=True, resize=False)
        self.vgg.to(device)
        
    def forward(self, input, target):
        return self.vgg(input, target)

class HistogramLoss(nn.Module):
    """
    Matches the pixel intensity distribution of the images.
    Crucial for tuning noise levels (bg_noise_std) and exposure (shadow_gain).
    
    Method:
    1. Compute differentiable histogram using Kernel Density Estimation (KDE) 
       or soft binning.
    2. Compare histograms using MSE or KL Divergence.
    """
    def __init__(self, bins=64, min_val=0.0, max_val=1.0, sigma=0.01):
        super().__init__()
        self.bins = bins
        self.min_val = min_val
        self.max_val = max_val
        self.sigma = sigma
        # Bin centers
        self.register_buffer('centers', torch.linspace(min_val, max_val, bins))
        
    def forward(self, input, target):
        """
        Args:
            input: (N, C, H, W) in [0, 1]
            target: (N, C, H, W) in [0, 1]
        """
        # Auto-broadcast target if batch size mismatches
        if input.shape[0] != target.shape[0]:
            if target.shape[0] == 1:
                target = target.expand(input.shape[0], -1, -1, -1)
            else:
                raise ValueError(f"Batch size mismatch: {input.shape} vs {target.shape} and target is not 1")

        # Flatten spatial dims: (N, C, H*W)
        input_flat = input.view(input.shape[0], input.shape[1], -1)
        target_flat = target.view(target.shape[0], target.shape[1], -1)
        
        # Compute Soft Histograms
        hist_input = self.soft_histogram(input_flat)
        hist_target = self.soft_histogram(target_flat)
        
        # Compare (MSE is usually stable enough)
        return F.mse_loss(hist_input, hist_target)
        
    def soft_histogram(self, x):
        """
        Differentiable Histogram using Gaussian Soft Binning.
        x: (N, C, L)
        Returns: (N, C, Bins)
        """
        # x: (N, C, L, 1)
        # centers: (1, 1, 1, Bins)
        x = x.unsqueeze(-1)
        centers = self.centers.view(1, 1, 1, -1)
        
        # Distance from center
        # (N, C, L, Bins)
        dist = x - centers
        
        # Gaussian Kernel
        # exp(-dist^2 / sigma^2)
        weights = torch.exp(-torch.square(dist) / (2 * self.sigma ** 2))
        
        # Sum over pixels (L) -> (N, C, Bins)
        hist = torch.sum(weights, dim=2)
        
        # Normalize to sum to 1 (Probability Density)
        hist = hist / (hist.sum(dim=2, keepdim=True) + 1e-6)
        
        return hist
