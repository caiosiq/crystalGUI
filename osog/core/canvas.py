import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class Canvas:
    width: int
    height: int
    image: np.ndarray = field(init=False)
    # Future expansion: depth_map, label_map
    
    def __post_init__(self):
        # Initialize as float32 for accumulation, will clip to uint8 later
        # Or init as None and let background fill it?
        # The background function usually creates the image.
        # Let's initialize with zeros or allow setting later.
        self.image = np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def set_image(self, img: np.ndarray):
        if img.shape[:2] != (self.height, self.width):
            raise ValueError(f"Image shape {img.shape} does not match canvas {(self.height, self.width)}")
        self.image = img

    def to_uint8(self) -> np.ndarray:
        return np.clip(self.image, 0, 255).astype(np.uint8)
