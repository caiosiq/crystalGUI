"""
Data generator module for synthetic crystal-like images, used by the Synthesis tab.

This is an analogous, self-contained generator inspired by synth_speckles/batched_full_synth.py,
but simplified and parameterized so it can be driven by GUI inputs. It does not modify or depend
on the original script, and can be extended with more parameters over time.
"""

from .synth import default_config, generate_image, params_for_t, sample_lambda, lambda_to_t

__all__ = [
    "default_config",
    "generate_image",
    "params_for_t",
    "sample_lambda",
    "lambda_to_t",
]