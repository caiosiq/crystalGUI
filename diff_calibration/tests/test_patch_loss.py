
import torch
import torch.nn as nn
import unittest
from diff_calibration.loss.patch import RandomCropLoss

# Mock Loss Function to verify inputs
class MockLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.call_count = 0
        self.last_input_shape = None
        
    def forward(self, input, target):
        self.call_count += 1
        self.last_input_shape = input.shape
        # Return scalar loss
        return torch.mean((input - target)**2)

class TestRandomCropLoss(unittest.TestCase):
    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def test_crop_logic(self):
        """Test that patches are cropped to correct size and called num_crops times."""
        mock_loss = MockLoss()
        cropper = RandomCropLoss(mock_loss, crop_size=64, num_crops=5)
        
        # Large input
        img = torch.rand(1, 3, 256, 256).to(self.device)
        
        loss = cropper(img, img)
        
        # Check calls
        self.assertEqual(mock_loss.call_count, 5)
        
        # Check shape
        self.assertEqual(mock_loss.last_input_shape, (1, 3, 64, 64))
        
        # Loss should be 0 for identical input
        self.assertAlmostEqual(loss.item(), 0.0)

    def test_small_image_fallback(self):
        """Test that if image is smaller than crop, it passes full image once."""
        mock_loss = MockLoss()
        cropper = RandomCropLoss(mock_loss, crop_size=128, num_crops=5)
        
        # Small input (64x64)
        img = torch.rand(1, 3, 64, 64).to(self.device)
        
        loss = cropper(img, img)
        
        # Should only call ONCE with full image
        self.assertEqual(mock_loss.call_count, 1)
        self.assertEqual(mock_loss.last_input_shape, (1, 3, 64, 64))

    def test_gradient_flow(self):
        """Test that gradients flow through the crops back to the full image."""
        mock_loss = MockLoss() # MSE allows gradients
        cropper = RandomCropLoss(mock_loss, crop_size=64, num_crops=4)
        
        img = torch.rand(1, 3, 256, 256, requires_grad=True, device=self.device)
        target = torch.rand(1, 3, 256, 256, device=self.device)
        
        loss = cropper(img, target)
        loss.backward()
        
        self.assertIsNotNone(img.grad)
        # Gradient should be sparse (only where crops happened), but definitely non-zero
        self.assertNotEqual(img.grad.sum().item(), 0.0)
        
        print(f"Gradient Non-Zero Elements: {torch.count_nonzero(img.grad).item()}")

if __name__ == '__main__':
    unittest.main()
