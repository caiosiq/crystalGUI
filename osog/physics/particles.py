from dataclasses import dataclass, field
from typing import List, Tuple, Any, Optional
import math
import numpy as np
import cv2
try:
    import torch
except ImportError:
    torch = None

@dataclass
class RenderableObject:
    cx: float
    cy: float
    z: float = 0.0  # Depth for focus/blur logic
    requires_label: bool = True # Flag to distinguish Signal (Real) from Noise (Ghost)
    
    @property
    def bounding_box(self) -> Any:
        raise NotImplementedError

@dataclass
class Rod(RenderableObject):
    L: float = 0.0
    W: float = 0.0
    angle_deg: float = 0.0
    delta: float = 0.0
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
class RodBatch:
    # Geometry (N,)
    cx: 'torch.Tensor'
    cy: 'torch.Tensor'
    z: 'torch.Tensor'
    L: 'torch.Tensor'
    W: 'torch.Tensor'
    angle_deg: 'torch.Tensor'
    delta: 'torch.Tensor'
    
    # Flags (N,)
    requires_label: 'torch.Tensor' # bool
    
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
    # Rods sharing the same group_id belong to the same agglomerate
    group_id: 'torch.Tensor'

    def __len__(self):
        return self.cx.shape[0]
        
    def to(self, device):
        for f in self.__dataclass_fields__:
            val = getattr(self, f)
            if torch.is_tensor(val):
                setattr(self, f, val.to(device))
        return self

@dataclass
class Agglomerate(RenderableObject):
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
    size_px: int = 1
    delta: float = 0.0
    is_dash: bool = False
    angle_deg: float = 0.0
    seed: int = 0  # Added seed for reproducibility
    requires_label: bool = False # Debris is usually noise
    
    @property
    def bounding_box(self):
        r = self.size_px
        return self.cx - r, self.cy - r, self.cx + r, self.cy + r

@dataclass
class DebrisBatch:
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
            if torch.is_tensor(val):
                setattr(self, f, val.to(device))
        return self
