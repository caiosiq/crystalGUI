import torch
import torch.nn.functional as F
import math
from typing import Tuple, Optional

def generate_fractal_noise_3d(
    shape: Tuple[int, int, int, int, int], # (N, C, D, H, W)
    frequency: float = 4.0,
    octaves: int = 3,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    device: torch.device = torch.device('cpu'),
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Generates 3D Fractal Noise using trilinear interpolation of random lattices.
    Fast approximation of Perlin Noise.
    """
    N, C, D, H, W = shape
    
    if seed is not None:
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)
    else:
        rng = None
        
    noise = torch.zeros(shape, device=device)
    amplitude = 1.0
    freq = frequency
    
    # Grid coordinates for sampling [-1, 1]
    z = torch.linspace(-1, 1, D, device=device)
    y = torch.linspace(-1, 1, H, device=device)
    x = torch.linspace(-1, 1, W, device=device)
    
    # (D, H, W)
    grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing='ij')
    
    # (N, D, H, W, 3) -> (x, y, z) order for grid_sample
    base_grid = torch.stack([grid_x, grid_y, grid_z], dim=-1).unsqueeze(0).repeat(N, 1, 1, 1, 1)
    
    for _ in range(octaves):
        # Lattice resolution
        # We ensure at least 2x2x2
        res_d = max(2, int(freq))
        res_h = max(2, int(freq))
        res_w = max(2, int(freq))
        
        # Generate random lattice
        lattice = torch.randn(N, C, res_d, res_h, res_w, device=device, generator=rng)
        
        # Sample
        # We add a random offset to the grid to decorrelate octaves
        offset = torch.rand(N, 1, 1, 1, 3, device=device, generator=rng) * 2.0 - 1.0
        
        # padding_mode='reflection' creates nice continuity
        sampled = F.grid_sample(lattice, base_grid + offset * 0.5, mode='bilinear', padding_mode='reflection', align_corners=False)
        
        noise += sampled * amplitude
        
        amplitude *= persistence
        freq *= lacunarity
        
    # Normalize
    noise = (noise - noise.mean()) / (noise.std() + 1e-6)
    return noise

def generate_cellular_noise_3d(
    shape: Tuple[int, int, int, int, int], # (N, C, D, H, W)
    scale: float = 5.0, # Grid size
    jitter: float = 1.0,
    device: torch.device = torch.device('cpu'),
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Generates 3D Cellular (Voronoi) Noise using a Look-Up Table approach.
    Returns: Distance to nearest feature point (Worley Noise).
    """
    N, C, D, H, W = shape
    if seed is not None:
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)
    else:
        rng = None
        
    # 1. Generate Feature Points Grid (LUT)
    # The grid size determines the "density" of cells
    # We want 'scale' cells along the largest dimension
    max_dim = max(D, H, W)
    sd = max(2, int(scale * D / max_dim))
    sh = max(2, int(scale * H / max_dim))
    sw = max(2, int(scale * W / max_dim))
    
    # Feature points: random offsets (0..1) relative to cell origin
    # Shape: (N, 3, sd, sh, sw) -> 3 channels for x,y,z offset
    points_lut = torch.rand(N, 3, sd, sh, sw, device=device, generator=rng)
    
    # 2. Coordinate Grids
    # We need to map pixel coordinates to (cell_index, fractional_pos)
    # The grid covers [0, s]
    z = torch.linspace(0, sd, D, device=device)
    y = torch.linspace(0, sh, H, device=device)
    x = torch.linspace(0, sw, W, device=device)
    
    gz, gy, gx = torch.meshgrid(z, y, x, indexing='ij')
    
    # (N, 1, D, H, W)
    gz = gz.unsqueeze(0).unsqueeze(1).expand(N, 1, D, H, W)
    gy = gy.unsqueeze(0).unsqueeze(1).expand(N, 1, D, H, W)
    gx = gx.unsqueeze(0).unsqueeze(1).expand(N, 1, D, H, W)
    
    base_z = torch.floor(gz)
    base_y = torch.floor(gy)
    base_x = torch.floor(gx)
    
    fract_z = gz - base_z
    fract_y = gy - base_y
    fract_x = gx - base_x
    
    min_dist_sq = torch.ones(N, C, D, H, W, device=device) * 100.0
    
    # 3. Neighbor Search (3x3x3)
    # We iterate neighbor offsets
    for oz in [-1, 0, 1]:
        for oy in [-1, 0, 1]:
            for ox in [-1, 0, 1]:
                # Neighbor Cell Index
                nz = base_z + oz
                ny = base_y + oy
                nx = base_x + ox
                
                # Wrap indices for periodic boundary or clamp?
                # grid_sample supports wrapping (padding_mode='border' or 'reflection')
                # But here we are doing manual lookup.
                # Let's use F.grid_sample to fetch the point from points_lut!
                
                # To use grid_sample, we need normalized coordinates [-1, 1].
                # Our cell indices nz are in [0, sd].
                # Normalized = (nz + 0.5) / sd * 2 - 1
                
                # Normalize coordinates for grid_sample
                # We need to sample the point at cell (nx, ny, nz)
                # grid_sample coord: (x, y, z)
                norm_x = (nx + 0.5) / sw * 2.0 - 1.0
                norm_y = (ny + 0.5) / sh * 2.0 - 1.0
                norm_z = (nz + 0.5) / sd * 2.0 - 1.0
                
                # Stack grid (N, D, H, W, 3)
                grid = torch.stack([norm_x, norm_y, norm_z], dim=-1).squeeze(1) # Remove C dim
                
                # Fetch the random point offset for this neighbor cell
                # mode='nearest' gives us the exact value for that cell
                # padding_mode='border' (zeros) or 'reflection'. 
                # If we go out of bounds, we get a point.
                # 'border' gives 0.0 which creates a point at the corner.
                point_offset = F.grid_sample(points_lut, grid, mode='nearest', padding_mode='border', align_corners=False)
                
                # point_offset is (N, 3, D, H, W) -> (px, py, pz)
                
                # Calculate vector to that point
                # Vector = (NeighborCell + PointOffset) - (CurrentCell + Fract)
                #        = (Current + Offset + PointOffset) - (Current + Fract)
                #        = Offset + PointOffset - Fract
                
                # But wait, if we are at the edge, 'border' returns 0.
                # That effectively places a point at (0,0,0) in that imaginary cell.
                # This is fine.
                
                vec_z = oz + point_offset[:, 2:3] * jitter - fract_z
                vec_y = oy + point_offset[:, 1:2] * jitter - fract_y
                vec_x = ox + point_offset[:, 0:1] * jitter - fract_x
                
                dist_sq = vec_x**2 + vec_y**2 + vec_z**2
                
                min_dist_sq = torch.min(min_dist_sq, dist_sq)
                
    return torch.sqrt(min_dist_sq)

def generate_anisotropic_noise_2d(
    shape: Tuple[int, int, int, int], # (N, C, H, W)
    scale_x: float = 4.0,
    scale_y: float = 4.0,
    angle_deg: float = 0.0,
    octaves: int = 3,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    device: torch.device = torch.device('cpu'),
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Generates 2D Anisotropic Fractal Noise.
    Useful for simulating directional flow/streaks.
    """
    N, C, H, W = shape
    
    if seed is not None:
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)
    else:
        rng = None
        
    noise = torch.zeros(shape, device=device)
    amplitude = 1.0
    
    # Base frequencies
    freq_x = scale_x
    freq_y = scale_y
    
    # Coordinate Grid [-1, 1]
    y = torch.linspace(-1, 1, H, device=device)
    x = torch.linspace(-1, 1, W, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    
    # Rotation
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    
    # Rotated coordinates
    # x' = x cos - y sin
    # y' = x sin + y cos
    rot_x = grid_x * cos_a - grid_y * sin_a
    rot_y = grid_x * sin_a + grid_y * cos_a
    
    # (N, H, W, 2) for grid_sample
    # We will scale these dynamically per octave
    
    for _ in range(octaves):
        # Lattice resolution
        res_h = max(2, int(freq_y))
        res_w = max(2, int(freq_x))
        
        # Generate random lattice
        lattice = torch.randn(N, C, res_h, res_w, device=device, generator=rng)
        
        # Create sampling grid
        # (N, H, W, 2)
        sx = rot_x * freq_x
        sy = rot_y * freq_y
        
        # Random offset for octave decorrelation
        off = torch.rand(N, 1, 1, 2, device=device, generator=rng) * 100.0
        
        sample_grid = torch.stack([sx, sy], dim=-1).unsqueeze(0).expand(N, -1, -1, -1) + off
        
        # To sample from a finite lattice using these potentially large coordinates, 
        # we use padding_mode='reflection'.
        # The lattice can be small, e.g. 8x8.
        # The "frequency" is effectively controlled by the coordinate scaling.
        
        lattice_res = 8 # Sufficient for linear interpolation
        lattice = torch.randn(N, C, lattice_res, lattice_res, device=device, generator=rng)
        
        sampled = F.grid_sample(lattice, sample_grid, mode='bilinear', padding_mode='reflection', align_corners=False)
        
        noise += sampled * amplitude
        
        amplitude *= persistence
        freq_x *= lacunarity
        freq_y *= lacunarity
        
    # Normalize
    noise = (noise - noise.mean()) / (noise.std() + 1e-6)
    return noise
