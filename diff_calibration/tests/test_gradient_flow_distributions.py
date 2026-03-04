
import torch
import unittest
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from osog.config import SynthConfig
from diff_calibration.diff_wrapper import DiffOSOG

class TestGradientFlowDistributions(unittest.TestCase):
    def setUp(self):
        self.cfg = SynthConfig()
        self.cfg.canvas.width = 64
        self.cfg.canvas.height = 64
        self.cfg.canvas.use_gpu = False # CPU for testing
        
        # Enable Rods
        self.cfg.physics.rods.enable = False # Disable generic rods
        self.cfg.physics.use_specific_specs = True
        self.cfg.physics.rod_specs.enable = True
        self.cfg.physics.rod_specs.count_range = (1, 1) # Just 1 rod for clarity
        self.cfg.physics.rod_specs.length_range = (20.0, 30.0)
        self.cfg.physics.rod_specs.aspect_range = (0.2, 0.2) # Fixed aspect
        
        # Disable other stuff
        self.cfg.sensor.bg_noise_std = 0.0
        self.cfg.sensor.fouling_enable = False
        
    def test_rod_length_gradient(self):
        """Test if gradients flow from image pixels back to rod_length_max."""
        active_params = ['rod_length_max']
        model = DiffOSOG(self.cfg, active_params)
        
        # Check initial value
        self.assertAlmostEqual(model.rod_length_max.item(), 30.0)
        self.assertTrue(model.rod_length_max.requires_grad)
        
        # Forward pass
        # Use fixed seed for determinism
        output = model(seed=42)
        
        # Loss: Total brightness (Area of particles)
        # Increasing max length -> larger particles -> more brightness
        loss = output.sum()
        
        loss.backward()
        
        print(f"Rod Length Max Grad: {model.rod_length_max.grad}")
        
        self.assertIsNotNone(model.rod_length_max.grad)
        # Gradient should be positive (larger length -> more pixels)
        self.assertGreater(model.rod_length_max.grad.item(), 0.0)

    def test_rod_width_jit_gradient(self):
        """Test if gradients flow back to width jitter amplitude."""
        # Note: Width jitter increases randomness, but might not increase mean area linearly.
        # But if we use a loss that favors "roughness" or variance, maybe?
        # Or simpler: jitter adds noise.
        pass

if __name__ == '__main__':
    unittest.main()
