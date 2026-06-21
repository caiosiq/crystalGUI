# Material-level presets previously lived here (ettringite, struvite, etc.).
# Playground presets are JSON files in data/synth_presets/ only.

PRESETS = {}


def load_preset(config, preset_name):
    """Apply a named preset layer on top of config. No built-in presets remain."""
    if preset_name not in PRESETS:
        return config
    return PRESETS[preset_name].apply(config)


def get_optimization_bounds(preset_name):
    """Extract optimizable bounds from a preset. Returns empty if unknown."""
    if preset_name not in PRESETS:
        return {}

    preset = PRESETS[preset_name]
    bounds = {}

    def extract_bounds(d, prefix=""):
        for k, v in d.items():
            if isinstance(v, dict) and "min" in v and v["min"] is not None:
                bounds[prefix + k] = (v["min"], v["max"])
            elif isinstance(v, dict) and "val" not in v:
                extract_bounds(v, prefix + k + ".")

    extract_bounds(preset.config_overrides)
    return bounds
