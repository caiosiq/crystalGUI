
from dataclasses import dataclass, field
from typing import List, Tuple, Any, Optional
import numpy as np
import cv2

try:
    import torch
except ImportError:
    torch = None

from .constants import (
    SHAPE_ROD, SHAPE_PLATE, SHAPE_CUBE, SHAPE_SPHERE, SHAPE_BUBBLE, SHAPE_DROPLET, SHAPE_POLYHEDRA
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
        # Check for 3D rotation
        beta = getattr(self, 'beta', 0.0)
        gamma = getattr(self, 'gamma', 0.0)
        H = getattr(self, 'H', self.W) # Default to W (cylindrical) if H missing
        
        if abs(beta) < 0.1 and abs(gamma) < 0.1:
            # Simple 2D Case
            rect = ((self.cx, self.cy), (self.L, self.W), self.angle_deg)
            return cv2.boxPoints(rect)
        else:
            # 3D Projection Case
            # 1. Define 8 corners of the box in local frame centered at 0
            lx, ly, lz = self.L / 2, self.W / 2, H / 2
            # 8 corners: (x, y, z)
            local_pts = np.array([
                [-lx, -ly, -lz], [lx, -ly, -lz], [lx, ly, -lz], [-lx, ly, -lz],
                [-lx, -ly,  lz], [lx, -ly,  lz], [lx, ly,  lz], [-lx, ly,  lz]
            ])
            
            # 2. Rotation Matrices
            # Order: We rotate by gamma (roll/Y), then beta (pitch/X), then alpha (yaw/Z)
            # Or whatever convention matches the shader.
            # Shader: M_inv = Rx(-gamma) * Ry(-beta) to go World->Box. 
            # So Box->World is Ry(beta) * Rx(gamma)? 
            # Wait, shader uses:
            # D_local = M_inv * (0,0,1). M_inv constructed from beta/gamma.
            # Shader uses:
            # X_rot = ct*X + st*Y (Alpha rotation first?)
            # No, shader says "Unnormalized Rotated Coordinates (aligned with alpha)".
            # Then it does Slab intersection in frame rotated by beta/gamma relative to alpha-frame.
            # So Total Rotation R = Rz(alpha) * Ry(beta) * Rx(gamma)
            
            def Rz(deg):
                rad = np.deg2rad(deg)
                c, s = np.cos(rad), np.sin(rad)
                return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            
            def Ry(deg): # Pitch around Y axis? Shader calls beta "rotation around Y (Pitch)"?
                # Shader: "beta is rotation around Y (Pitch)"
                # Actually usually Pitch is X or Y. Let's assume Y.
                rad = np.deg2rad(deg)
                c, s = np.cos(rad), np.sin(rad)
                return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
                
            def Rx(deg): # Roll around X axis?
                # Shader: "gamma around X (Roll)"
                rad = np.deg2rad(deg)
                c, s = np.cos(rad), np.sin(rad)
                return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
            
            # Match shader convention:
            # The shader transforms World (X,Y) -> Alpha-Aligned (X_rot, Y_rot) -> Box (x,y,z)
            # P_box = R_gamma^-1 * R_beta^-1 * R_alpha^-1 * P_world
            # So P_world = R_alpha * R_beta * R_gamma * P_box
            
            R = Rz(self.angle_deg) @ Ry(beta) @ Rx(gamma)
            
            # 3. Rotate Points
            # points is (8, 3). R is (3, 3). 
            # (R @ P.T).T = P @ R.T
            rotated_pts = local_pts @ R.T
            
            # 4. Translate to Center (cx, cy)
            # We only care about X, Y projection
            proj_pts = rotated_pts[:, :2] # Drop Z
            proj_pts[:, 0] += self.cx
            proj_pts[:, 1] += self.cy
            
            # 5. Find MinAreaRect of these 8 points
            rect = cv2.minAreaRect(proj_pts.astype(np.float32))
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
    
    # Phase 4.3: Technicolor
    reflectivity: 'torch.Tensor'
    dispersion: 'torch.Tensor'
    absorption_color: 'torch.Tensor' # (N, 3)
    
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
