
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralLoss(nn.Module):
    def __init__(self, log_scale=True, epsilon=1e-8):
        """
        Spectral Loss using Fast Fourier Transform (FFT).
        Compares the frequency domain amplitude spectrum of images.
        Crucial for matching blur (high freq decay) and noise (high freq floor).
        
        Args:
            log_scale (bool): If True, compares log(amplitude + eps). 
                              This prevents dominance of low-frequency components (DC term).
            epsilon (float): Small constant to avoid log(0).
        """
        super().__init__()
        self.log_scale = log_scale
        self.eps = epsilon

    def forward(self, input, target):
        """
        Args:
            input: (N, C, H, W) tensor, range [0, 1]
            target: (N, C, H, W) tensor, range [0, 1]
        """
        # Auto-broadcast target if batch size mismatches
        if input.shape[0] != target.shape[0]:
            # Expand target to match input batch size
            if target.shape[0] == 1:
                target = target.expand(input.shape[0], -1, -1, -1)
            else:
                raise ValueError(f"Batch size mismatch: {input.shape} vs {target.shape} and target is not 1")
                
        # Input validation
        if input.shape != target.shape:
            raise ValueError(f"Shape mismatch: {input.shape} vs {target.shape}")
            
        # 1. Compute 2D FFT
        # rfft2 computes FFT for real input, returning only the non-redundant half of frequencies
        # Output shape: (N, C, H, W/2 + 1) complex64
        # 'ortho' normalization makes FFT unitary (energy preserving)
        fft_input = torch.fft.rfft2(input, norm='ortho')
        fft_target = torch.fft.rfft2(target, norm='ortho')
        
        # 2. Compute Amplitude Spectrum (Magnitude)
        amp_input = torch.abs(fft_input)
        amp_target = torch.abs(fft_target)
        
        # 3. Log Scaling (Optional but Recommended)
        # Without log, the DC component (average brightness) dominates the loss by orders of magnitude.
        if self.log_scale:
            amp_input = torch.log(amp_input + self.eps)
            amp_target = torch.log(amp_target + self.eps)
            
        # 4. MSE Loss on Spectrum
        loss = F.mse_loss(amp_input, amp_target)
        
        return loss
