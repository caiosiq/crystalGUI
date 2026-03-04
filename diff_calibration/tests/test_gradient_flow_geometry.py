
import torch
import unittest
import random
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from osog.config import SynthConfig
from osog.optics.shaders.geometry import GeometryShader
from osog.physics.particles import ParticleBatch, SHAPE_ROD, SHAPE_SPHERE

def create_dummy_batch(device, **kwargs):
    """Helper to create a ParticleBatch with default values, overriding specific fields."""
    N = 1
    defaults = {
        'cx': torch.tensor([32.0], device=device, requires_grad=True),
        'cy': torch.tensor([32.0], device=device, requires_grad=True),
        'z': torch.tensor([0.0], device=device),
        'L': torch.tensor([20.0], device=device, requires_grad=True),
        'W': torch.tensor([5.0], device=device, requires_grad=True),
        'H': torch.tensor([5.0], device=device),
        'alpha': torch.tensor([0.0], device=device, requires_grad=True),
        'beta': torch.tensor([0.0], device=device),
        'gamma': torch.tensor([0.0], device=device),
        'delta': torch.tensor([0.1], device=device),
        'refractive_index': torch.tensor([1.5], device=device),
        'birefringence': torch.tensor([0.0], device=device),
        'opacity': torch.tensor([0.0], device=device),
        'reflectivity': torch.tensor([0.04], device=device),
        'dispersion': torch.tensor([0.01], device=device),
        'absorption_color': torch.tensor([[0.0, 0.0, 0.0]], device=device),
        'texture_type': torch.tensor([0], device=device),
        'surf_roughness': torch.tensor([0.0], device=device),
        'grain_size': torch.tensor([1.0], device=device),
        'internal_inclusions': torch.tensor([0.0], device=device),
        'turbidity': torch.tensor([0.0], device=device),
        'anisotropy': torch.tensor([0.0], device=device),
        'anisotropy_angle': torch.tensor([0.0], device=device),
        'requires_label': torch.tensor([True], device=device),
        'shape_id': torch.tensor([SHAPE_ROD], device=device),
        'curvature': torch.tensor([0.0], device=device),
        'width_jit_amp': torch.tensor([0.0], device=device),
        'edge_jit_amp': torch.tensor([0.0], device=device),
        'offset_jit_amp': torch.tensor([0.0], device=device),
        'ragged_p': torch.tensor([0.0], device=device),
        'ragged_corr': torch.tensor([0.2], device=device),
        'polarity_flip_p': torch.tensor([0.0], device=device),
        'shape_mode': torch.tensor([0], device=device),
        'seed': torch.tensor([123], device=device),
        'group_id': torch.tensor([0], device=device)
    }
    
    # Override defaults
    for k, v in kwargs.items():
        if k in defaults:
            defaults[k] = v
            
    return ParticleBatch(**defaults)

class TestGradientFlowGeometry(unittest.TestCase):
    def setUp(self):
        self.cfg = SynthConfig()
        self.cfg.canvas.width = 64
        self.cfg.canvas.height = 64
        self.device = torch.device('cpu')
        self.shader = GeometryShader(self.cfg, self.device)
        self.rng = random.Random(42)

    def test_soft_mask_gradient_rod(self):
        """Test if changing L/W of a Rod changes the mask via gradients."""
        # Tensors requiring grad
        cx = torch.tensor([32.0], requires_grad=True)
        cy = torch.tensor([32.0], requires_grad=True)
        L = torch.tensor([20.0], requires_grad=True)
        W = torch.tensor([5.0], requires_grad=True)
        angle = torch.tensor([45.0], requires_grad=True)
        
        batch = create_dummy_batch(self.device, cx=cx, cy=cy, L=L, W=W, H=W, alpha=angle, shape_id=torch.tensor([SHAPE_ROD]))

        # Forward Pass with Soft Edge
        g_buffer, _, _, _ = self.shader.render_batch(batch, self.rng, soft_edge_mode=True)
        
        # g_buffer: (N, 4, H, W)
        # Channel 1 is Mask
        height = g_buffer[:, 0, :, :]
        mask = g_buffer[:, 1, :, :]
        
        print(f"Height range: {height.min().item()} - {height.max().item()}")
        print(f"Mask range: {mask.min().item()} - {mask.max().item()}")
        
        # Loss: Sum of mask (Area)
        loss = mask.sum()
        loss.backward()
        
        print(f"Rod L Grad: {L.grad}")
        print(f"Rod W Grad: {W.grad}")
        print(f"Rod CX Grad: {cx.grad}")
        
        self.assertIsNotNone(L.grad)
        self.assertNotEqual(L.grad.item(), 0.0)
        self.assertGreater(L.grad.item(), 0.0) 
        
        self.assertIsNotNone(W.grad)
        self.assertGreater(W.grad.item(), 0.0)
        
        # CX gradient should be near 0 because shifting center doesn't change Area significantly if not clipped
        # But if clipped, it might.
        # Here we are in center of 64x64, so it shouldn't change area.
        # Actually, discrete pixel sampling might cause slight changes?
        # But with soft mask, area is continuous integral.
        # So dArea/dPos should be 0.
        # But let's check if we were to compute something position dependent.
        # E.g. overlap with a target mask.

    def test_soft_mask_gradient_sphere(self):
        """Test gradient flow for Sphere profile."""
        D = torch.tensor([15.0], requires_grad=True)
        
        batch = create_dummy_batch(self.device, L=D, W=D, H=D, shape_id=torch.tensor([SHAPE_SPHERE]))

        # Forward
        g_buffer, _, _, _ = self.shader.render_batch(batch, self.rng, soft_edge_mode=True)
        mask = g_buffer[:, 1, :, :]
        loss = mask.sum()
        loss.backward()
        
        print(f"Sphere D Grad: {D.grad}")
        
        self.assertIsNotNone(D.grad)
        self.assertGreater(D.grad.item(), 0.0)

if __name__ == '__main__':
    unittest.main()
