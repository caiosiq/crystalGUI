import time
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

# Add project root to path
# osog/scripts/comp_analysis.py -> crystalGUI
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Also support running from root
sys.path.append(os.getcwd())

from osog.core.pipeline import Pipeline
from osog.config import SynthConfig

def benchmark_throughput():
    print("=== OSOG Computational Throughput Benchmark ===")
    
    # Setup Config
    cfg = SynthConfig()
    cfg.canvas.width = 1024
    cfg.canvas.height = 1024
    cfg.canvas.use_gpu = True
    
    # We will vary the number of particles
    # For Rods, n_rods_rng_lo_hi controls the count
    
    particle_counts = [100, 500, 1000, 2000, 5000, 10000, 20000, 40000]
    render_times = []
    render_stds = []
    
    # Warmup
    print("Warming up GPU...")
    for i in range(100):
        cfg.physics.rods.n_rods_rng_lo_hi = (100, 100, 100)
        pipeline = Pipeline(cfg)
        pipeline.generate(t=0.5, no_detail=True)
    
    for N in particle_counts:
        print(f"Benchmarking N={N}...")
        
        # Set fixed count
        cfg.physics.rods.n_rods_rng_lo_hi = (N, N, N)
        
        # Re-init pipeline to ensure config update (though generate re-reads config usually, but let's be safe)
        pipeline = Pipeline(cfg)
        
        # Average over K runs
        K = 100
        times = []
        for _ in range(K):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.time()
            
            pipeline.generate(t=0.5, no_detail=True)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.time()
            times.append((t1 - t0) * 1000.0) # ms
            
        avg_time = np.mean(times)
        std_time = np.std(times)
        render_times.append(avg_time)
        render_stds.append(std_time)
        print(f"  -> {avg_time:.2f} ms +/- {std_time:.2f}")
        
    # --- Plotting ---
    print("Generating Plot...")
    
    # Academic Style
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'lines.linewidth': 2.5,
        'figure.figsize': (8, 6),
        'font.family': 'serif',
        'axes.grid': True,
        'grid.alpha': 0.3
    })
    
    fig, ax = plt.subplots()
    
    # 1. OSOG Data with Error Bars
    ax.errorbar(particle_counts, render_times, yerr=render_stds, 
                fmt='-o', color='#1976D2', ecolor='#546E7A', 
                capsize=5, elinewidth=2, label='OSOG (GPU)', zorder=3)
    
    ax.set_title("Computational Throughput")
    ax.set_xlabel("Number of Particles ($N$)")
    ax.set_ylabel("Render Time (ms)")
    ax.set_ylim(bottom=0)
    
    # Legend
    ax.legend(loc='upper left', frameon=True, fancybox=True)
    
    # Add a shaded region for "Real-Time" performance (< 33.3 ms or > 30 FPS)
    ax.axhspan(0, 33.3, facecolor='#4CAF50', alpha=0.15, zorder=1)
    ax.text(39000, 24, "Real-Time Generation (>30 FPS)", color='#388E3C', 
            fontsize=12, horizontalalignment='right', verticalalignment='top')
    
    # Annotate "1024x1024 Canvas"
    ax.text(0.95, 0.05, "Canvas: 1024$\\times$1024", transform=ax.transAxes, 
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
    plt.tight_layout()
    
    out_path = os.path.join(os.path.dirname(__file__), "results.png")
    plt.savefig(out_path, dpi=300)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    benchmark_throughput()
