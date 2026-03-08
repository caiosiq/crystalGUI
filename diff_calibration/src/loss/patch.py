
import torch
import torch.nn as nn
import random

class RandomCropLoss(nn.Module):
    def __init__(self, loss_fn, crop_size=256, num_crops=4):
        """
        Wraps any loss function to operate on random patches instead of the full image.
        This drastically reduces VRAM usage for high-res optimization (e.g. 1024x1024)
        while still capturing local details (texture/noise/blur).
        
        Args:
            loss_fn (nn.Module): The base loss function (e.g., VGGPerceptualLoss, SpectralLoss).
            crop_size (int): Size of the square crop (H=W=crop_size).
            num_crops (int): Number of random crops to sample per forward pass.
        """
        super().__init__()
        self.loss_fn = loss_fn
        self.crop_size = crop_size
        self.num_crops = num_crops
        
    def forward(self, input, target):
        """
        Args:
            input: (N, C, H, W)
            target: (N, C, H, W)
        """
        N, C, H, W = input.shape
        
        # If image is smaller than crop, just compute full loss
        if H <= self.crop_size or W <= self.crop_size:
            return self.loss_fn(input, target)
            
        total_loss = 0.0
        
        # Vectorized Cropping?
        # We can stack patches and run forward once, which is faster but higher peak VRAM.
        # But for heavy losses (VGG), sequential patches is safer for VRAM.
        # Let's do sequential for now to guarantee VRAM savings.
        
        for _ in range(self.num_crops):
            # Random coordinates
            # We want random crops, but matched between input and target
            # So input[y:y+h, x:x+w] matched with target[y:y+h, x:x+w]
            
            y = random.randint(0, H - self.crop_size)
            x = random.randint(0, W - self.crop_size)
            
            # Slicing is zero-copy in PyTorch (view), so this is fast
            input_patch = input[:, :, y:y+self.crop_size, x:x+self.crop_size]
            target_patch = target[:, :, y:y+self.crop_size, x:x+self.crop_size]
            
            # Accumulate loss
            total_loss += self.loss_fn(input_patch, target_patch)
            
        # Average the loss
        return total_loss / self.num_crops
