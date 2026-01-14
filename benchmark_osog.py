import time
import torch
import numpy as np
from osog.core.pipeline import Pipeline
from osog.config import SynthConfig

def benchmark():
    print("="*60)
    print("OSOG PROFILE: Vectorized Pipeline")
    print("="*60)
    
    # 1. Setup Configuration for a heavy workload
    N_RODS = 2000 
    CANVAS_SIZE = 1024
    
    config = {
        "canvas": {
            "width": CANVAS_SIZE,
            "height": CANVAS_SIZE,
            "use_gpu": True, # This enables the Torch path (CPU or CUDA)
            "parallel_workers": 1
        },
        "physics": {
            "rods": {
                "enable": True,
                "n_rods_rng_lo_hi": (N_RODS, N_RODS),
                "rod_len_px_lo_hi": (30, 80),
            },
            "debris": { "rate": 0.0001 }, # Enable some debris to test that path
            "ghosts": { "fraction": 0.1 } # Enable some ghosts
        }
    }

    print(f"Workload: {N_RODS} rods + ghosts + debris on {CANVAS_SIZE}x{CANVAS_SIZE}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU (Vectorized)'}")
    print("-" * 60)

    try:
        pipeline = Pipeline(config)
        
        # Warmup
        print("Warming up...")
        pipeline.generate(t=0.5)
        print("-" * 60)
        
        # Measure
        print("Running Profile...")
        start_time = time.time()
        pipeline.generate(t=0.5)
        end_time = time.time()
        
        print("-" * 60)
        print(f"Total Wall Time: {end_time - start_time:.4f} seconds")
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

    print("="*60)

if __name__ == "__main__":
    benchmark()
