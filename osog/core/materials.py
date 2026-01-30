
from dataclasses import dataclass
from typing import Literal, Dict, Tuple

@dataclass
class MaterialProperties:
    name: str
    
    # --- Micro-Surface (Texture) ---
    # These modify the physical height map before rendering
    texture_type: Literal["smooth", "striated", "pitted", "granular"] = "smooth"
    roughness: float = 0.0      # Amplitude of random surface noise
    grain_size: float = 1.0     # Scale of the noise patterns
    
    # --- Optical Properties ---
    # These determine interaction with the light engine
    refractive_index: float = 1.50
    birefringence: float = 0.0  # For Polarization mode (0 = isotropic)
    opacity: float = 0.0        # 0 = transparent, 1 = opaque (absorbs light)
    
    # Phase 4.3: Technicolor Support
    reflectivity: float = 0.04 # Base reflectance at normal incidence (linear). 0.04 is typical glass/water.
    dispersion: float = 0.01   # Change in RI across visible spectrum (Abbe number proxy). 0 = no rainbows.
    absorption_color: Tuple[float, float, float] = (1.0, 1.0, 1.0) # RGB transmission filter. (1,1,1) is white/clear.
    
    # --- Advanced Interactions ---
    # Specific behaviors
    internal_inclusions: float = 0.0 # "Cloudiness" inside the particle

# ==========================================
# MATERIAL PRESETS (The Scientific Truths)
# ==========================================

MATERIALS: Dict[str, MaterialProperties] = {
    # Generic Plastic/Glass
    "standard": MaterialProperties(
        name="Standard",
        texture_type="smooth",
        roughness=0.02,
        refractive_index=1.50
    ),
    
    # Crystalline Needles (e.g., Threonine)
    # Grows in layers -> Longitudinal lines (Striations)
    "crystal_fibrous": MaterialProperties(
        name="Fibrous Crystal",
        texture_type="striated",
        roughness=0.15,
        birefringence=0.2, # Strongly lights up in polarization
        refractive_index=1.55
    ),
    
    # Blocky Salts (e.g., Adipic Acid, NaCl)
    # Cleaves cleanly -> Smooth faces, sharp edges
    "crystal_smooth": MaterialProperties(
        name="Smooth Crystal",
        texture_type="smooth",
        roughness=0.01, # Very clean faces
        birefringence=0.1,
        refractive_index=1.52
    ),
    
    # Amorphous blobs (e.g., Protein Aggregates)
    # No structure -> Pitted, bumpy surface
    "amorphous": MaterialProperties(
        name="Amorphous Aggregate",
        texture_type="pitted",
        roughness=0.6, # Very bumpy
        opacity=0.4,   # Significant light blocking
        internal_inclusions=0.8, # Very cloudy inside
        refractive_index=1.42
    ),
    
    # Opaque Powder (e.g., TiO2, insoluble salts)
    "opaque_powder": MaterialProperties(
        name="Opaque Powder",
        texture_type="granular",
        roughness=0.4,
        opacity=0.95, # Almost black
        refractive_index=2.6, # High RI
        internal_inclusions=0.0
    ),
    
    # Glass (Microbeads)
    "glass": MaterialProperties(
        name="Glass",
        texture_type="smooth",
        roughness=0.0,
        refractive_index=1.52,
        birefringence=0.0
    ),
    
    # Air Bubble (in water n=1.33) -> n=1.0
    "air": MaterialProperties(
        name="Air",
        texture_type="smooth",
        roughness=0.0,
        refractive_index=1.00,
        birefringence=0.0
    ),
    
    # Oil Droplet (in water n=1.33) -> n=1.47
    "oil": MaterialProperties(
        name="Oil",
        texture_type="smooth",
        roughness=0.0,
        refractive_index=1.47,
        birefringence=0.0
    ),
    
    # Metallic Particle (e.g., steel shaving)
    # Opaque + High reflectivity (not fully simulated yet, but high opacity/smoothness helps)
    "metal": MaterialProperties(
        name="Metallic Shaving",
        texture_type="striated", # machining marks
        roughness=0.05,
        opacity=1.0, # Fully opaque
        refractive_index=2.5, # Fake high RI for strong edges
        internal_inclusions=0.0,
        reflectivity=0.8, # Very shiny
        absorption_color=(0.8, 0.8, 0.8) # Grey
    ),
    
    # Highly Birefringent (e.g., Urea, Ascorbic Acid)
    "crystal_high_birefringence": MaterialProperties(
        name="High Birefringence Crystal",
        texture_type="smooth",
        roughness=0.01,
        refractive_index=1.65,
        birefringence=0.25, # Very high -> Colorful
        dispersion=0.03 # Rainbow edges
    )
}

def get_material(name: str) -> MaterialProperties:
    return MATERIALS.get(name, MATERIALS["standard"])
