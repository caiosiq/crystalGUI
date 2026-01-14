from dataclasses import dataclass
from typing import Any
from .particles import RenderableObject

@dataclass
class GhostObject(RenderableObject):
    """
    A wrapper class that treats any RenderableObject as a 'Ghost'.
    It delegates geometric properties to the wrapped object but enforces
    requires_label=False and can override other properties (like Z-depth)
    to simulate being out-of-focus or 'noise'.
    """
    wrapped_obj: RenderableObject = None
    
    def __post_init__(self):
        if self.wrapped_obj is None:
            raise ValueError("GhostObject must wrap a RenderableObject")
            
        # Enforce no label for ghosts
        self.requires_label = False
        # Sync basic properties if not set explicitly during init
        # (Though dataclass init sets them from args, we might want to ensure consistency)
        if self.cx == 0.0 and self.cy == 0.0:
             self.cx = self.wrapped_obj.cx
             self.cy = self.wrapped_obj.cy
        
        # If z was passed as 0.0 (default), maybe we should inherit or offset?
        # Typically ghost logic sets Z explicitly.
        
    @property
    def bounding_box(self) -> Any:
        return self.wrapped_obj.bounding_box
        
    @property
    def corners(self) -> Any:
        if hasattr(self.wrapped_obj, 'corners'):
            return self.wrapped_obj.corners
        return None

    def __getattr__(self, name: str):
        # Delegate attribute access to the wrapped object for things like L, W, angle_deg, etc.
        return getattr(self.wrapped_obj, name)
