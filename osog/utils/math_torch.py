import torch
import torch.nn.functional as F
import math
import numpy as np
from typing import Tuple, Optional, Union

def lerp(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return a + t * (b - a)

def smooth_cap(u: torch.Tensor, a: float, b: float) -> torch.Tensor:
    t = torch.clamp((torch.abs(u) - a) / (b - a + 1e-6), 0.0, 1.0)
    return 1.0 - (t * t * (3.0 - 2.0 * t))

def interp1d(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """
    One-dimensional linear interpolation for monotonically increasing sample points.
    Returns the one-dimensional piecewise linear interpolant to a function with given discrete data points (xp, fp), evaluated at x.
    Same as np.interp.
    """
    # Flatten input x to 1D for interpolation, then reshape back
    original_shape = x.shape
    x_flat = x.view(-1)
    
    # Sort xp just in case, though usually we pass sorted grid
    # xp, sort_idx = torch.sort(xp)
    # fp = fp[sort_idx]
    
    # Handle boundaries
    m = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1] + 1e-8)
    b = fp[:-1] - (m * xp[:-1])
    
    ind = torch.searchsorted(xp, x_flat, right=True)
    ind = torch.clamp(ind - 1, 0, len(xp) - 2)
    
    res = m[ind] * x_flat + b[ind]
    
    # Clamp out of bounds to endpoints (like np.interp default)
    # Actually np.interp returns left/right values for out of bounds
    min_mask = x_flat < xp[0]
    max_mask = x_flat > xp[-1]
    
    res[min_mask] = fp[0]
    res[max_mask] = fp[-1]
    
    return res.view(original_shape)

def interp1d_batch(x: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """
    Batched 1D interpolation using grid_sample.
    Assumes xp is linspace(-1, 1).
    x: (N, H, W) or (N, L) - values in [-1, 1]
    fp: (N, S) - samples corresponding to linspace(-1, 1, S)
    """
    N = x.shape[0]
    S = fp.shape[1]
    
    # grid_sample expects input (N, C, H_in, W_in) and grid (N, H_out, W_out, 2)
    # We map 1D problem to 2D:
    # Input: fp -> (N, 1, 1, S)
    img = fp.view(N, 1, 1, S)
    
    # Grid: x -> (N, 1, M, 2) where M is flattened size of H*W
    # y coordinate is always 0
    # x coordinate is x
    flat_x = x.view(N, -1) # (N, M)
    M = flat_x.shape[1]
    
    zeros = torch.zeros_like(flat_x)
    # stack to (N, M, 2)
    # Note: grid_sample coordinates are (x, y). We want to sample along width (x).
    grid = torch.stack([flat_x, zeros], dim=-1).unsqueeze(1) # (N, 1, M, 2)
    
    # Sample
    # align_corners=True matches linspace(-1, 1) inclusive
    out = F.grid_sample(img, grid, mode='bilinear', padding_mode='border', align_corners=True)
    # out is (N, 1, 1, M)
    
    return out.view(x.shape)

def gaussian_blur_1d(signal: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Apply 1D Gaussian blur to a signal (N,).
    Uses Conv1d.
    """
    # Flatten input signal to ensure it's 1D
    original_shape = signal.shape
    signal_flat = signal.view(-1)
    
    k_size = max(3, int(round(3 * sigma)) * 2 + 1)
    # Create 1D Gaussian kernel
    x = torch.arange(k_size, device=signal.device, dtype=signal.dtype) - (k_size - 1) / 2
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    
    # Reshape for conv1d: (OutC, InC/Group, K) -> (1, 1, K)
    kernel = kernel.view(1, 1, -1)
    
    # Signal: (Batch, Channel, Length) -> (1, 1, N)
    inp = signal_flat.view(1, 1, -1)
    
    # Pad to keep size same
    pad = k_size // 2
    out = torch.nn.functional.conv1d(inp, kernel, padding=pad)
    
    return out.view(original_shape)

def gaussian_blur_1d_batch(signal: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Apply 1D Gaussian blur to a batch of signals (N, L).
    signal: (N, L)
    """
    N, L = signal.shape
    k_size = max(3, int(round(3 * sigma)) * 2 + 1)
    
    x = torch.arange(k_size, device=signal.device, dtype=signal.dtype) - (k_size - 1) / 2
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    
    # Groups=N for independent filtering
    # Weights: (N, 1, K)
    weights = kernel.view(1, 1, -1).repeat(N, 1, 1)
    
    # Input: (1, N, L) -> Wait, Conv1d is (N, C, L)
    # If we want independent N signals, we can treat them as N channels with groups=N
    # Input: (1, N, L)
    inp = signal.view(1, N, L)
    
    pad = k_size // 2
    out = F.conv1d(inp, weights, padding=pad, groups=N)
    
    return out.view(N, L)

def noise1d_like(u: torch.Tensor, corr: float = 0.25, amp: float = 1.0, seed: Optional[int] = None) -> torch.Tensor:
    if seed is not None:
        gen = torch.Generator(device=u.device)
        gen.manual_seed(seed)
    else:
        gen = None
        
    # Fixed grid like numpy version
    grid = torch.linspace(-1, 1, 512, device=u.device, dtype=u.dtype)
    prof = torch.randn(grid.size(0), device=u.device, dtype=u.dtype, generator=gen)
    
    # Blur
    # In numpy: k = max(3, int(round(3 * corr * 64)) * 2 + 1), sigma = corr * 32
    sigma = corr * 32
    prof = gaussian_blur_1d(prof, sigma)
    
    prof = prof - prof.mean()
    prof = prof / (prof.std() + 1e-6)
    
    n = interp1d(u, grid, prof)
    return amp * n

def noise1d_like_batch(u: torch.Tensor, corr: float = 0.25, amp: float = 1.0, seed: Optional[int] = None) -> torch.Tensor:
    """
    u: (N, H, W)
    """
    N = u.shape[0]
    gen = torch.Generator(device=u.device)
    if seed is not None:
        gen.manual_seed(seed)
    
    # (N, 512)
    prof = torch.randn(N, 512, device=u.device, dtype=u.dtype, generator=gen)
    
    sigma = corr * 32
    prof = gaussian_blur_1d_batch(prof, sigma)
    
    # Normalize per row
    prof = prof - prof.mean(dim=1, keepdim=True)
    prof = prof / (prof.std(dim=1, keepdim=True) + 1e-6)
    
    n = interp1d_batch(u, prof)
    return amp * n

def sin_wobble(u: torch.Tensor, amp_px: float = 1.2, cycles: Tuple[float, float] = (0.6, 1.5), seed: Optional[int] = None) -> torch.Tensor:
    if seed is not None:
        gen = torch.Generator(device=u.device)
        gen.manual_seed(seed)
    else:
        gen = None
        
    # Generate random params
    # uniform in torch needs a buffer or use rand
    rand_vals = torch.rand(2, device=u.device, dtype=u.dtype, generator=gen)
    
    f = (cycles[0] + (cycles[1] - cycles[0]) * rand_vals[0]) * math.pi
    ph = rand_vals[1] * 2 * math.pi
    
    wob = amp_px * (0.6 * torch.sin(f * u + ph) + 0.4 * torch.sin(1.8 * f * u + 2.0 * ph))
    return wob

def sin_wobble_batch(u: torch.Tensor, amp_px: float = 1.2, cycles: Tuple[float, float] = (0.6, 1.5), seed: Optional[int] = None) -> torch.Tensor:
    """
    u: (N, H, W)
    """
    N = u.shape[0]
    gen = torch.Generator(device=u.device)
    if seed is not None:
        gen.manual_seed(seed)
        
    rand_vals = torch.rand(N, 2, device=u.device, dtype=u.dtype, generator=gen)
    
    f = (cycles[0] + (cycles[1] - cycles[0]) * rand_vals[:, 0]) * math.pi
    ph = rand_vals[:, 1] * 2 * math.pi
    
    # Broadcast to (N, H, W)
    f = f.view(N, 1, 1)
    ph = ph.view(N, 1, 1)
    
    wob = amp_px * (0.6 * torch.sin(f * u + ph) + 0.4 * torch.sin(1.8 * f * u + 2.0 * ph))
    return wob

def kink(u: torch.Tensor, amp_px: float = 1.5, seed: Optional[int] = None) -> torch.Tensor:
    if seed is not None:
        gen = torch.Generator(device=u.device)
        gen.manual_seed(seed)
    else:
        gen = None
        
    u0 = (torch.rand(1, device=u.device, dtype=u.dtype, generator=gen).item() * 0.6) - 0.3
    s = torch.tanh((u - u0) * 6.0)
    return amp_px * 0.5 * s

def kink_batch(u: torch.Tensor, amp_px: float = 1.5, seed: Optional[int] = None) -> torch.Tensor:
    N = u.shape[0]
    gen = torch.Generator(device=u.device)
    if seed is not None:
        gen.manual_seed(seed)
    
    u0 = (torch.rand(N, 1, 1, device=u.device, dtype=u.dtype, generator=gen) * 0.6) - 0.3
    s = torch.tanh((u - u0) * 6.0)
    return amp_px * 0.5 * s

def noisy_wobble(u: torch.Tensor, amp_px: float = 1.0, corr: float = 0.18, seed: Optional[int] = None) -> torch.Tensor:
    # Use noise1d_like which already includes blurring/smoothness via 'corr'
    # Avoiding extra gaussian_blur_1d here prevents potential scanline artifacts
    # from flattening 2D tensors.
    n = noise1d_like(u, corr=corr, amp=1.0, seed=seed)
    
    # Just normalize
    n = n - n.mean()
    n = n / (n.std() + 1e-6)
    return amp_px * n

def noisy_wobble_batch(u: torch.Tensor, amp_px: float = 1.0, corr: float = 0.18, seed: Optional[int] = None) -> torch.Tensor:
    n = noise1d_like_batch(u, corr=corr, amp=1.0, seed=seed)
    
    # Normalize per item
    # u is (N, H, W)
    # we normalize over H,W dims
    n = n - n.mean(dim=(1, 2), keepdim=True)
    n = n / (n.std(dim=(1, 2), keepdim=True) + 1e-6)
    return amp_px * n

def ragged_mask(u: torch.Tensor, p: float = 0.08, corr: float = 0.20, seed: Optional[int] = None) -> torch.Tensor:
    if seed is not None:
        gen = torch.Generator(device=u.device)
        gen.manual_seed(seed)
    else:
        gen = None
        
    grid = torch.linspace(-1, 1, 512, device=u.device, dtype=u.dtype)
    rand_vals = torch.rand(grid.size(0), device=u.device, dtype=u.dtype, generator=gen)
    keep = (rand_vals > p).to(u.dtype)
    
    sigma = corr * 32
    keep = gaussian_blur_1d(keep, sigma)
    
    mn = keep.min()
    rngv = keep.max() - mn
    keep = (keep - mn) / (rngv + 1e-6)
    keep = 0.25 + 0.75 * keep
    
    return interp1d(u, grid, keep)

def ragged_mask_batch(u: torch.Tensor, p: float = 0.08, corr: float = 0.20, seed: Optional[int] = None) -> torch.Tensor:
    N = u.shape[0]
    gen = torch.Generator(device=u.device)
    if seed is not None:
        gen.manual_seed(seed)
        
    rand_vals = torch.rand(N, 512, device=u.device, dtype=u.dtype, generator=gen)
    keep = (rand_vals > p).to(u.dtype)
    
    sigma = corr * 32
    keep = gaussian_blur_1d_batch(keep, sigma)
    
    mn = keep.min(dim=1, keepdim=True)[0]
    mx = keep.max(dim=1, keepdim=True)[0]
    rngv = mx - mn
    
    keep = (keep - mn) / (rngv + 1e-6)
    keep = 0.25 + 0.75 * keep
    
    return interp1d_batch(u, keep)
