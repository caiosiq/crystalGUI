from __future__ import annotations
"""
Legacy wrapper for the new O-SOG (Optically-Simulated Object Generator) package.
This ensures backward compatibility with existing code that imports from data_generator.synth.
"""

from crystalGUI.osog import generate_image, default_config, SynthConfig, sample_lambda, lambda_to_t, params_for_t

# Re-export everything
__all__ = [
    "generate_image",
    "default_config",
    "SynthConfig",
    "sample_lambda",
    "lambda_to_t",
    "params_for_t"
]
