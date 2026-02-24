
import torch
import unittest
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from osog.config import SynthConfig
from diff_calibration.diff_wrapper import DiffOSOG

class TestDiffWrapper(unittest.TestCase):
    def setUp(self):
        # Create a simple config
        self.cfg = SynthConfig()
        # Enable GPU if available, else CPU
        self.cfg.canvas.use_gpu = torch.cuda.is_available()
        self.cfg.canvas.width = 256
        self.cfg.canvas.height = 256
        
        # Enable sensor features that use gradients
        self.cfg.sensor.blur_sigma = 1.0
        self.cfg.sensor.bg_noise_std = 5.0
        
        # Enable Distractors (Soup) to test gradients there too
        self.cfg.sensor.distractor_enable = True
        self.cfg.sensor.distractor_opacity = 0.5
        
    def test_parameter_registration(self):
        """Test if parameters are correctly registered and mapped."""
        active_params = ['blur_sigma', 'noise_scale']
        model = DiffOSOG(self.cfg, active_params)
        
        # Check if parameters exist
        self.assertTrue(hasattr(model, 'blur_sigma'))
        self.assertTrue(hasattr(model, 'noise_scale'))
        
        # Check if they require grad
        self.assertTrue(model.blur_sigma.requires_grad)
        self.assertTrue(model.noise_scale.requires_grad)
        
        # Check values
        self.assertAlmostEqual(model.blur_sigma.item(), 1.0)
        self.assertAlmostEqual(model.noise_scale.item(), 5.0)

    def test_gradient_flow_blur(self):
        """Test if gradient flows through blur_sigma."""
        active_params = ['blur_sigma']
        model = DiffOSOG(self.cfg, active_params)
        
        # Forward
        output = model(seed=42)
        
        # Loss: Maximize intensity (arbitrary)
        loss = output.mean()
        
        # Backward
        loss.backward()
        
        # Check grad
        print(f"Blur Sigma Grad: {model.blur_sigma.grad}")
        self.assertIsNotNone(model.blur_sigma.grad)
        self.assertNotEqual(model.blur_sigma.grad.item(), 0.0)

    def test_gradient_flow_soup(self):
        """Test if gradient flows through distractor_opacity."""
        active_params = ['distractor_opacity']
        model = DiffOSOG(self.cfg, active_params)
        
        # Forward
        output = model(seed=42)
        
        # Loss
        loss = output.mean()
        loss.backward()
        
        print(f"Soup Opacity Grad: {model.distractor_opacity.grad}")
        self.assertIsNotNone(model.distractor_opacity.grad)
        # Note: Opacity is linear multiplier, so grad should be effectively mean(soup_layer)
        self.assertNotEqual(model.distractor_opacity.grad.item(), 0.0)

if __name__ == '__main__':
    unittest.main()
