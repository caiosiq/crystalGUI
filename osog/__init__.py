from typing import Dict, Any, Optional
from .config import default_config, SynthConfig
from .core.pipeline import Pipeline
from .physics.distribution import sample_lambda, lambda_to_t, params_for_t

# Expose main interface
def generate_image(config: Dict[str, Any], t: float, seed: int | None = None, return_obbs: bool = False, parallel_workers: int | None = None, return_heads: bool = False):
    pipeline = Pipeline(config)
    return pipeline.generate(t, seed, return_obbs, parallel_workers, return_heads)

__all__ = [
    "generate_image",
    "default_config",
    "SynthConfig",
    "sample_lambda",
    "lambda_to_t",
    "params_for_t"
]
