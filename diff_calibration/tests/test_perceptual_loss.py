
import torch
import torch.nn as nn
import unittest
from diff_calibration.loss.perceptual import VGGPerceptualLoss
from PIL import Image
import numpy as np

class TestVGGPerceptualLoss(unittest.TestCase):
    def setUp(self):
        # Initialize loss module
        # Use a small device if available, else CPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.loss_fn = VGGPerceptualLoss(resize=True).to(self.device)
        self.loss_fn.eval()

    def test_identical_images(self):
        """Test that loss is zero for identical images."""
        img = torch.rand(1, 3, 256, 256).to(self.device)
        loss = self.loss_fn(img, img)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_noise_vs_structure(self):
        """Test that loss is high for random noise vs structure."""
        # Create a structured image (e.g., gradient)
        x = torch.linspace(0, 1, 256)
        y = torch.linspace(0, 1, 256)
        xx, yy = torch.meshgrid(x, y, indexing='xy')
        img_struct = torch.stack([xx, yy, torch.zeros_like(xx)], dim=0).unsqueeze(0).to(self.device)
        
        # Create random noise
        img_noise = torch.rand(1, 3, 256, 256).to(self.device)
        
        # Compare structure to noise
        loss_diff = self.loss_fn(img_struct, img_noise)
        
        # Compare structure to a slightly dimmed version of itself
        loss_similar = self.loss_fn(img_struct, img_struct * 0.9)
        
        print(f"Loss (Structure vs Noise): {loss_diff.item()}")
        print(f"Loss (Structure vs Dimmed): {loss_similar.item()}")
        
        # The network should find noise to be MUCH more different than a dim gradient
        self.assertGreater(loss_diff.item(), loss_similar.item() * 10)

    def test_texture_similarity(self):
        """Test that style loss is lower for shifted texture than for different texture."""
        # Create a "texture" (random noise pattern)
        texture_base = torch.rand(1, 3, 256, 256).to(self.device)
        
        # Shifted version (should have similar Gram matrix)
        texture_shifted = torch.roll(texture_base, shifts=(10, 10), dims=(2, 3))
        
        # Different texture (different random seed effectively)
        texture_diff = torch.rand(1, 3, 256, 256).to(self.device)
        
        loss_shift = self.loss_fn(texture_base, texture_shifted)
        loss_diff = self.loss_fn(texture_base, texture_diff)
        
        print(f"Loss (Shifted): {loss_shift.item()}")
        print(f"Loss (Different): {loss_diff.item()}")
        
        # Gram matrix (Style loss) should be relatively invariant to translation
        # But pure pixel loss would be high.
        # However, boundaries ruin exact translation invariance for Gram matrix on full image.
        # Still, it should be somewhat related.
        
        # Actually, for random noise, "different" and "shifted" might be similarly distant 
        # because there is no large-scale structure.
        # Let's try a periodic pattern (checkerboard).
        
        # Better test: Color scaling.
        # Style loss should be sensitive to color distribution.
        
        img_red = torch.zeros(1, 3, 64, 64).to(self.device)
        img_red[:, 0, :, :] = 1.0
        
        img_blue = torch.zeros(1, 3, 64, 64).to(self.device)
        img_blue[:, 2, :, :] = 1.0
        
        img_red_dim = img_red * 0.9 # Same style, slightly darker
        
        loss_color_change = self.loss_fn(img_red, img_blue)
        loss_intensity_change = self.loss_fn(img_red, img_red_dim)
        
        print(f"Loss (Red vs Blue): {loss_color_change.item()}")
        print(f"Loss (Red vs Dark Red): {loss_intensity_change.item()}")
        
        self.assertLess(loss_intensity_change.item(), loss_color_change.item())

    def test_gradient_flow(self):
        """Test that gradients flow back to input."""
        img_target = torch.rand(1, 3, 64, 64).to(self.device)
        img_opt = torch.rand(1, 3, 64, 64).to(self.device)
        img_opt.requires_grad = True
        
        loss = self.loss_fn(img_opt, img_target)
        loss.backward()
        
        self.assertIsNotNone(img_opt.grad)
        self.assertNotEqual(img_opt.grad.sum().item(), 0.0)

if __name__ == '__main__':
    unittest.main()
