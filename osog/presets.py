from .base import OSOGPreset, Param

# Import configs from separate modules if needed, or define here
# Ideally, "Materials" (Ettringite, Struvite) should be in their own file too?
# But for now, we keep them here as the "Registry".

# ==========================================
# THE DATABASE OF VALID "CANNY" REGIONS
# ==========================================

PRESETS = {
    # ----------------------------------------------------
    # PRESET 1: Fine Needles (e.g. Ettringite / Cement)
    # "Canny Region": High aspect ratio, but short length to avoid texture stretching
    # ----------------------------------------------------
    "ettringite": OSOGPreset(
        name="Ettringite Needles",
        description="High aspect ratio, thin needles. Smooth surface.",
        config_overrides={
            "physics": {
                "use_specific_specs": True,
                "rod_specs": {
                    "enable": True,
                    "count_range": Param([150, 300], [50, 50], [300, 600]), 
                    "length_range": Param((30.0, 90.0), (20.0, 20.0), (120.0, 120.0)),
                    "aspect_range": Param((0.05, 0.08), (0.02, 0.02), (0.1, 0.1)),
                    "shape_mode": "straight",
                    "ragged_p": Param(0.0, 0.0, 0.2),
                    "polarity_p": Param(0.0, 0.0, 0.0)
                },
                "sphere_specs": {"enable": False},
                "plate_specs": {"enable": False},
                "cube_specs": {"enable": False},
                "bubble_specs": {"enable": False},
                "droplet_specs": {"enable": False},
                "fused": {"enable": False}
            },
            "optics": {
                "mode": "dic"
            }
        }
    ),

    # ----------------------------------------------------
    # PRESET 2: Chunky Prisms (e.g. Struvite / Salts)
    # "Canny Region": Fat, short, highly distinct edges.
    # ----------------------------------------------------
    "struvite": OSOGPreset(
        name="Struvite Prisms",
        description="Blocky, coffin-like shapes with distinct edges.",
        config_overrides={
            "physics": {
                "use_specific_specs": True,
                "rod_specs": {
                    "enable": True,
                    "count_range": Param((20, 50), (10, 10), (80, 80)),
                    "length_range": Param((60.0, 150.0), (40.0, 40.0), (200.0, 200.0)),
                    "aspect_range": Param((0.2, 0.4), (0.15, 0.15), (0.5, 0.5)),
                    "shape_mode": "kink",
                    "ragged_p": Param(0.1, 0.0, 0.3),
                    "polarity_p": Param(0.2, 0.0, 0.5) 
                },
                "sphere_specs": {"enable": False},
                "plate_specs": {"enable": False}
            },
            "optics": {
                "mode": "dic",
                "shadow_gain": Param((12.0, 20.0), (10.0, 10.0), (30.0, 30.0))
            }
        }
    ),

    # ----------------------------------------------------
    # PRESET 3: Dirty Agglomerates (e.g. API Crash)
    # "Canny Region": Noisy surface, undefined shapes.
    # ----------------------------------------------------
    "agglomerate": OSOGPreset(
        name="Rough Agglomerates",
        description="Fused, rough particles. High surface noise.",
        config_overrides={
            "physics": {
                "use_specific_specs": True,
                "rod_specs": {
                    "enable": True,
                    "count_range": Param((10, 30), (5, 5), (50, 50)),
                    "length_range": Param((50.0, 150.0), (30.0, 30.0), (200.0, 200.0)),
                    "aspect_range": Param((0.2, 0.5), (0.1, 0.1), (0.8, 0.8)),
                    "ragged_p": Param(0.8, 0.5, 1.0),
                    "shape_mode": "noisy"
                },
                "fused": {
                    "enable": True,
                    "p1": Param(0.8, 0.5, 1.0)
                },
                "sphere_specs": {"enable": False}
            }
        }
    ),
    
    # ----------------------------------------------------
    # PRESET 4: Mixed Suspension (General)
    # ----------------------------------------------------
    "mixed": OSOGPreset(
        name="Mixed Suspension",
        description="A general mix of rods and spheres.",
        config_overrides={
            "physics": {
                "use_specific_specs": True,
                "rod_specs": {
                    "enable": True,
                    "count_range": (20, 40),
                    "ragged_p": 0.1,
                    "shape_mode": "straight"
                },
                "sphere_specs": {
                    "enable": True,
                    "count_range": (10, 20)
                }
            }
        }
    )
}


def load_preset(config, preset_name):
    """
    Applies the preset 'layer' on top of the current config.
    Returns the new config.
    """
    if preset_name not in PRESETS:
        # If passed None or invalid, return original
        return config
        
    # print(f"Loading Preset: {PRESETS[preset_name].name}")
    
    # Apply logic
    new_config = PRESETS[preset_name].apply(config)
    
    return new_config

def get_optimization_bounds(preset_name):
    """
    Extracts min/max bounds for all optimizable parameters in a preset.
    Useful for Inverse Design.
    """
    if preset_name not in PRESETS:
        return {}
        
    preset = PRESETS[preset_name]
    bounds = {}
    
    # Recursively find all Params with Min/Max
    def extract_bounds(d, prefix=""):
        for k, v in d.items():
            if isinstance(v, dict) and "min" in v and v["min"] is not None:
                # It's an optimizable parameter!
                bounds[prefix + k] = (v["min"], v["max"])
            elif isinstance(v, dict) and "val" not in v: # Skip if it is a Param dict
                extract_bounds(v, prefix + k + ".")
                
    extract_bounds(preset.config_overrides)
    return bounds
