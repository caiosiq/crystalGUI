
from dataclasses import dataclass, field
from typing import List, Tuple, Any, Optional
import numpy as np
import cv2

try:
    import torch
except ImportError:
    torch = None

from .constants import (
    SHAPE_ROD, SHAPE_PLATE, SHAPE_CUBE, SHAPE_SPHERE, SHAPE_BUBBLE, SHAPE_DROPLET
)

@dataclass
class RenderableObject:
    """Base class for all renderable entities in the OSOG engine."""
    cx: float
    cy: float
    z: float = 0.0  # Depth for focus/blur logic
    requires_label: bool = True # Flag to distinguish Signal (Real) from Noise (Ghost)
    
    @property
    def bounding_box(self) -> Any:
        raise NotImplementedError

@dataclass
class Rod(RenderableObject):
    """
    CPU-side representation of a single Rod/Particle.
    Used mainly for legacy CPU rendering or individual object manipulation.
    """
    L: float = 0.0
    W: float = 0.0
    angle_deg: float = 0.0
    delta: float = 0.0
    material: str = "standard"
    seed: int = 0
    
    # Optional parameters for ghosts/variants
    curvature: float = 0.0
    width_jit_amp: float = 0.0
    edge_jit_amp: float = 0.0
    offset_jit_amp: float = 0.0
    ragged_p: float = 0.0
    ragged_corr: float = 0.2
    mult_mix: float = 0.0
    polarity_flip_p: float = 0.0
    shape_mode: str = "straight"
    
    @property
    def corners(self) -> np.ndarray:
        rect = ((self.cx, self.cy), (self.L, self.W), self.angle_deg)
        return cv2.boxPoints(rect)

    @property
    def bounding_box(self) -> Tuple[float, float, float, float]:
        # returns x_min, y_min, x_max, y_max
        c = self.corners
        return c[:, 0].min(), c[:, 1].min(), c[:, 0].max(), c[:, 1].max()

@dataclass
class ParticleBatch:
    """
    Unified batch structure for all 3D particle types (Rods, Plates, Cubes, Spheres).
    This is the primary data structure passed to the GPU shader.
    """
    # Geometry (N,)
    cx: 'torch.Tensor'
    cy: 'torch.Tensor'
    z: 'torch.Tensor'
    
    # Dimensions
    L: 'torch.Tensor' # Length / Major Axis / Diameter
    W: 'torch.Tensor' # Width / Minor Axis
    H: 'torch.Tensor' # Height / Thickness
    
    # Orientation (Euler angles in degrees)
    alpha: 'torch.Tensor' # Z-rot (in-plane spin)
    beta: 'torch.Tensor'  # X-rot (tumble)
    gamma: 'torch.Tensor' # Y-rot (roll)
    
    # Optical Properties (Physics Based)
    delta: 'torch.Tensor' # Legacy: Computed effective delta (n_obj - n_med)
    refractive_index: 'torch.Tensor'
    birefringence: 'torch.Tensor'
    opacity: 'torch.Tensor'
    
    # Material / Surface Properties
    texture_type: 'torch.Tensor' # 0=smooth, 1=striated, 2=pitted, 3=granular
    surf_roughness: 'torch.Tensor'
    grain_size: 'torch.Tensor'
    internal_inclusions: 'torch.Tensor'
    
    # Flags (N,)
    requires_label: 'torch.Tensor' # bool
    
    # Shape ID (N,) - See .constants.py
    shape_id: 'torch.Tensor'
    
    # Optional / Variant params (N,)
    curvature: 'torch.Tensor'
    width_jit_amp: 'torch.Tensor'
    edge_jit_amp: 'torch.Tensor'
    offset_jit_amp: 'torch.Tensor'
    ragged_p: 'torch.Tensor'
    ragged_corr: 'torch.Tensor'
    polarity_flip_p: 'torch.Tensor'
    
    # Shape Mode: Integer Enum (0: straight, 1: wavy, 2: kink, 3: noisy)
    shape_mode: 'torch.Tensor' 
    
    # Seeds (N,)
    seed: 'torch.Tensor'
    
    # Grouping for Agglomerates (N,)
    group_id: 'torch.Tensor'

    def __len__(self):
        return self.cx.shape[0]
        
    def to(self, device):
        for f in self.__dataclass_fields__:
            val = getattr(self, f)
            if torch is not None and torch.is_tensor(val):
                setattr(self, f, val.to(device))
        return self

# Legacy Alias
RodBatch = ParticleBatch

@dataclass
class Agglomerate(RenderableObject):
    """
    A cluster of particles (Rods) fused together.
    """
    children: List[Rod] = field(default_factory=list)

    @property
    def bounding_box(self) -> Tuple[float, float, float, float]:
        if not self.children:
            return self.cx, self.cy, self.cx, self.cy
        
        boxes = [c.bounding_box for c in self.children]
        x_min = min(b[0] for b in boxes)
        y_min = min(b[1] for b in boxes)
        x_max = max(b[2] for b in boxes)
        y_max = max(b[3] for b in boxes)
        return x_min, y_min, x_max, y_max
    
    @property
    def corners(self) -> np.ndarray:
        # Collect all corners from all children
        all_pts = []
        for child in self.children:
            all_pts.append(child.corners)
        
        if not all_pts:
             return np.array([], dtype=np.float32)

        # Stack into one big array of points (N*4, 2)
        pts = np.vstack(all_pts)
        
        # Calculate the "MinAreaRect" (Rotated Bounding Box) enclosing everything
        rect = cv2.minAreaRect(pts)
        return cv2.boxPoints(rect)

@dataclass
class Debris(RenderableObject):
    """Single debris particle (CPU side)."""
    size_px: int = 1
    delta: float = 0.0
    is_dash: bool = False
    angle_deg: float = 0.0
    seed: int = 0
    requires_label: bool = False 
    
    @property
    def bounding_box(self):
        r = self.size_px
        return self.cx - r, self.cy - r, self.cx + r, self.cy + r

@dataclass
class DebrisBatch:
    """Batch of debris particles (GPU side)."""
    cx: 'torch.Tensor'
    cy: 'torch.Tensor'
    z: 'torch.Tensor'
    size_px: 'torch.Tensor'
    delta: 'torch.Tensor'
    angle_deg: 'torch.Tensor'
    is_dash: 'torch.Tensor' # bool
    seed: 'torch.Tensor'
    
    def __len__(self):
        return self.cx.shape[0]

    def to(self, device):
        for f in self.__dataclass_fields__:
            val = getattr(self, f)
            if torch is not None and torch.is_tensor(val):
                setattr(self, f, val.to(device))
        return self
