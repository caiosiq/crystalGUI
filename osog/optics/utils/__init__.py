import torch
import torch.nn.functional as F

def gaussian_blur_2d(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Apply Gaussian blur to a (C, H, W) or (H, W) tensor.
    """
    if sigma <= 0:
        return img
        
    if img.dim() == 2:
        img = img.unsqueeze(0) # (1, H, W)
        squeeze = True
    else:
        squeeze = False
        
    C, H, W = img.shape
    
    k_ideal = int(round(3 * sigma)) * 2 + 1
    max_k = min(H, W)
    if max_k % 2 == 0:
        max_k -= 1
    k_size = max(1, min(k_ideal, max_k))
    
    if k_size == 1:
        return img.squeeze(0) if squeeze else img

    pad = k_size // 2
    x = torch.arange(k_size, device=img.device, dtype=img.dtype) - pad
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / (kernel.sum() + 1e-6)
    kernel = kernel.view(1, 1, -1)
    
    # Blur Rows
    # img: (C, H, W) -> view as (1, C*H, W) for conv1d
    inp_rows = img.view(1, C * H, W)
    k_rows = kernel.repeat(C * H, 1, 1) # (C*H, 1, K)
    out_rows = F.conv1d(inp_rows, k_rows, padding=pad, groups=C * H)
    out_rows = out_rows.view(C, H, W)
    
    # Blur Cols
    inp_cols = out_rows.view(C, H, W).transpose(1, 2).reshape(1, C * W, H)
    k_cols = kernel.repeat(C * W, 1, 1)
    out_cols = F.conv1d(inp_cols, k_cols, padding=pad, groups=C * W)
    
    out = out_cols.view(C, W, H).transpose(1, 2)
    
    return out.squeeze(0) if squeeze else out

def gaussian_blur_batch(img: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
    """
    Apply varying Gaussian blur to a batch (N, H, W) or (N, C, H, W).
    Currently supports (N, H, W) mainly, logic below is for (N, H, W).
    """
    if img.dim() == 3:
        N, H, W = img.shape
        # Ensure sigmas is (N,)
        if isinstance(sigmas, float):
            sigmas = torch.full((N,), sigmas, device=img.device)
            
        sigmas = torch.clamp(sigmas, min=0.1)
        max_sigma = sigmas.max().item()
        k_ideal = int(round(3 * max_sigma)) * 2 + 1
        max_k = min(H, W)
        if max_k % 2 == 0: max_k -= 1
        k_size = max(1, min(k_ideal, max_k))
        if k_size == 1: return img

        pad = k_size // 2
        x = torch.arange(k_size, device=img.device, dtype=img.dtype) - pad
        kernels = torch.exp(-0.5 * (x.unsqueeze(0) / sigmas.unsqueeze(1)) ** 2)
        kernels = kernels / (kernels.sum(dim=1, keepdim=True) + 1e-6)
        
        inp_rows = img.view(1, N * H, W)
        k_rows = kernels.unsqueeze(1).repeat(1, H, 1).view(N * H, 1, k_size)
        out_rows = F.conv1d(inp_rows, k_rows, padding=pad, groups=N * H)
        out_rows = out_rows.view(N, H, W)
        
        inp_cols = out_rows.transpose(1, 2).reshape(1, N * W, H)
        k_cols = kernels.unsqueeze(1).repeat(1, W, 1).view(N * W, 1, k_size)
        out_cols = F.conv1d(inp_cols, k_cols, padding=pad, groups=N * W)
        
        return out_cols.view(N, W, H).transpose(1, 2)
    else:
        # TODO: Handle (N, C, H, W) if needed
        raise NotImplementedError("gaussian_blur_batch only supports (N, H, W) input")
