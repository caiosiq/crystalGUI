
import torch
import torch.nn as nn
import unittest
from diff_calibration.loss.spectral import SpectralLoss

class TestSpectralLoss(unittest.TestCase):
    def setUp(self):
        # Initialize loss module
        # Use a small device if available, else CPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.loss_fn = SpectralLoss(log_scale=True).to(self.device)
        self.loss_fn.eval()

    def test_identical_images(self):
        """Test that loss is zero for identical images."""
        img = torch.rand(1, 3, 256, 256).to(self.device)
        loss = self.loss_fn(img, img)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_blur_sensitivity(self):
        """
        Test that spectral loss detects blur differences.
        A blurred image loses high frequencies, so its spectrum should differ significantly.
        """
        # Create a sharp image (random noise has lots of high freq)
        img_sharp = torch.rand(1, 3, 256, 256).to(self.device)
        
        # Create a blurred version using simple average pooling or conv2d
        # Simple box blur kernel
        kernel_size = 5
        padding = kernel_size // 2
        # Manually construct kernel without relying on external utils
        kernel = torch.ones(1, 1, kernel_size, kernel_size).to(self.device) / (kernel_size**2)
        kernel = kernel.repeat(3, 1, 1, 1) # Repeat for 3 channels
        
        # Apply depthwise convolution (groups=3) to blur each channel independently
        img_blur = torch.nn.functional.conv2d(img_sharp, kernel, padding=padding, groups=3)
        
        loss = self.loss_fn(img_sharp, img_blur)
        print(f"Loss (Sharp vs Blur): {loss.item()}")
        
        # Compare with Pixel Loss (MSE)
        pixel_loss = torch.nn.functional.mse_loss(img_sharp, img_blur)
        print(f"Pixel Loss (MSE): {pixel_loss.item()}")
        
        # Spectral loss should be significant because high freqs are gone.
        self.assertGreater(loss.item(), 0.1)

    def test_noise_floor_sensitivity(self):
        """
        Test that spectral loss detects noise level differences.
        Adding noise raises the high-frequency floor.
        """
        img_clean = torch.zeros(1, 3, 256, 256).to(self.device) + 0.5
        
        noise_level = 0.1
        img_noisy = img_clean + torch.randn_like(img_clean) * noise_level
        img_noisy = torch.clamp(img_noisy, 0, 1)
        
        loss = self.loss_fn(img_clean, img_noisy)
        print(f"Loss (Clean vs Noisy): {loss.item()}")
        self.assertGreater(loss.item(), 0.1)

    def test_translation_invariance(self):
        """
        Test that spectral amplitude is invariant to translation (shift).
        FFT magnitude |F(x)| is shift-invariant. Phase changes, but magnitude stays same.
        """
        img = torch.rand(1, 3, 256, 256).to(self.device)
        
        # Shift by rolling
        shift_x, shift_y = 50, 50
        img_shifted = torch.roll(img, shifts=(shift_x, shift_y), dims=(2, 3))
        
        # Spectral Loss uses AMPLITUDE only, so loss should be 0.0 (perfect invariance)
        # Note: torch.roll wraps around, which matches FFT periodicity assumption perfectly.
        
        loss = self.loss_fn(img, img_shifted)
        print(f"Loss (Shifted): {loss.item()}")
        
        # Should be extremely close to 0
        self.assertAlmostEqual(loss.item(), 0.0, places=5)
        
        # Compare with Pixel Loss
        pixel_loss = torch.nn.functional.mse_loss(img, img_shifted)
        print(f"Pixel Loss (Shifted): {pixel_loss.item()}")
        self.assertGreater(pixel_loss.item(), 0.1)

    def test_gradient_flow(self):
        """Test that gradients flow back to input."""
        img_target = torch.rand(1, 3, 64, 64).to(self.device)
        img_opt = torch.rand(1, 3, 64, 64).to(self.device)
        img_opt.requires_grad = True
        
        loss = self.loss_fn(img_opt, img_target)
        loss.backward()
        
        self.assertIsNotNone(img_opt.grad)
        # Check gradient magnitude
        grad_norm = img_opt.grad.norm().item()
        print(f"Gradient Norm: {grad_norm}")
        self.assertNotEqual(grad_norm, 0.0)

if __name__ == '__main__':
    unittest.main()
