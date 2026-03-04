
import torch
import unittest
from osog.optics.optical_engine import OpticalEngine

class TestSoftClamp(unittest.TestCase):
    def test_soft_clamp_values(self):
        """Test that soft_clamp bounds values correctly."""
        x = torch.tensor([-10.0, 0.0, 10.0, 100.0])
        min_val = 0.0
        max_val = 1.0
        
        y = OpticalEngine.soft_clamp(x, min_val, max_val, temp=0.1)
        
        # Check bounds
        self.assertTrue(torch.all(y >= min_val))
        self.assertTrue(torch.all(y <= max_val))
        
        # Check mid value (should be 0.5)
        # soft_clamp(0, 0, 1) -> mid + half * tanh(0) = 0.5
        y_mid = OpticalEngine.soft_clamp(torch.tensor([0.5]), 0.0, 1.0)
        # Wait, if x is in range, it should be close to x.
        # My implementation: mid + half * tanh((x - mid) / (half * temp))
        # if x=0.5, mid=0.5. tanh(0) = 0. out = 0.5. Correct.
        self.assertAlmostEqual(y_mid.item(), 0.5, places=4)

    def test_soft_clamp_gradient(self):
        """Test that gradients flow through soft_clamp even for out-of-bound values."""
        # Value moderately outside range (e.g. 1.5 vs max 1.0)
        # If we go too far (e.g. 10.0), even soft clamp saturates.
        x = torch.tensor([1.5], requires_grad=True)
        min_val = 0.0
        max_val = 1.0
        
        # Hard clamp would give 0 grad
        y_hard = torch.clamp(x, min_val, max_val)
        y_hard.backward()
        self.assertEqual(x.grad, 0.0) # Should be 0 for hard clamp
        
        x.grad = None
        
        # Soft clamp should give non-zero grad
        # Use temp=1.0 (default)
        y_soft = OpticalEngine.soft_clamp(x, min_val, max_val, temp=1.0)
        y_soft.backward()
        
        print(f"Gradient at x={x.item()}: {x.grad.item()}")
        self.assertNotEqual(x.grad.item(), 0.0)
        self.assertTrue(x.grad.item() > 0.0)

if __name__ == '__main__':
    unittest.main()
