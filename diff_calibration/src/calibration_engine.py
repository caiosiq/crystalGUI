import torch
import torch.optim as optim
import numpy as np
from typing import List, Dict, Optional, Callable

# Components
from diff_calibration.src.diff_wrapper import DiffOSOG
from diff_calibration.src.parameter_manager import ParameterManager
from diff_calibration.src.auto_scheduler import AutoScheduler
from diff_calibration.src.loss_manager import LossManager
from diff_calibration.src.convergence_guard import ConvergenceGuard
from diff_calibration.src.robustness import RobustnessGuard

# Losses
from diff_calibration.src.loss.perceptual import VGGPerceptualLoss
from diff_calibration.src.loss.spectral import SpectralLoss
from diff_calibration.src.loss.texture import GramMatrixLoss, HistogramLoss
from diff_calibration.src.loss.balancer import LossBalancer

class CalibrationEngine:
    """
    The Unified Parameter Optimization Engine.
    
    Orchestrates the entire calibration process:
    1. Parameter Management (Bounds, Normalization)
    2. Curriculum Scheduling (Geometry -> Texture -> Fine-tune)
    3. Semantic Loss Routing (VGG vs Gram vs Histogram)
    4. Robustness (Gradient Clipping, Convergence Detection)
    """
    def __init__(self, 
                 model: DiffOSOG, 
                 device: str = 'cpu',
                 base_lr: float = 0.05,
                 max_steps: int = 300,
                 rules_path: str = None):
        
        self.model = model
        self.device = device
        self.base_lr = base_lr
        self.max_steps = max_steps
        
        # --- The Brain Components ---
        self.param_manager = ParameterManager(model, rules_path)
        self.scheduler = AutoScheduler(rules_path)
        self.loss_manager = LossManager(rules_path)
        self.convergence_guard = ConvergenceGuard()
        self.robustness_guard = RobustnessGuard(self.param_manager.latent_params) # Watch latent params
        
        # --- Loss Functions ---
        self.losses = {
            'vgg': VGGPerceptualLoss(resize=False).to(device),
            'spectral': SpectralLoss().to(device),
            'gram': GramMatrixLoss(device).to(device),
            'histogram': HistogramLoss().to(device)
        }
        
        self.optimizer = None
        self.history = {'loss': [], 'stage': []}
        
    def calibrate(self, 
                  target_img: torch.Tensor, 
                  selected_params: List[str], 
                  progress_callback: Optional[Callable] = None):
        """Runs the full calibration pipeline."""
        
        # 1. Build Plan & Init Parameters
        plan = self.scheduler.build_plan(selected_params)
        print(f"[Engine] Calibration Plan: {[s['name'] for s in plan]}")
        
        self.param_manager.init_parameters(selected_params)
        
        # Fix for Pinhole Camera Effect (Zero Gradients on Focus)
        # If we optimize focus_z but aperture is 0.0 (default), DoF is infinite -> No gradients.
        if 'optics.focus_z' in selected_params:
            # Check if aperture is also being optimized
            if 'optics.aperture' not in selected_params:
                 # Check current value
                 current_ap = self.model.cfg.optics.aperture
                 if current_ap < 0.05:
                     print(f"[Engine] Warning: Aperture is {current_ap}. Forcing to 0.2 to enable focus gradients.")
                     self.model.cfg.optics.aperture = 0.2
            else:
                 # It is being optimized, but we should ensure it starts non-zero?
                 # ParamManager handles initialization based on rules, usually sets to center of bounds.
                 # Rules say bounds [0.001, 1.0], so it should be fine.
                 pass

        global_step = 0
        
        # 2. Execution Loop
        for stage_idx, stage in enumerate(plan):
            # Sync Scheduler State
            self.scheduler.current_stage_idx = stage_idx
            
            stage_name = stage['name']
            stage_steps = int(self.max_steps * stage['steps_ratio'])
            
            print(f"\n[Engine] Starting Stage {stage_idx+1}: {stage_name}")
            print(f"         Steps: {stage_steps} | Seed: {stage['seed_mode']}")
            
            # --- Stage Setup ---
            active_params = self.scheduler.get_active_params_for_stage(selected_params)
            
            # Freeze/Unfreeze Logic
            for name, param in self.param_manager.latent_params.items():
                real_name = next((p for p in selected_params if p.replace('.', '_') == name), None)
                if real_name in active_params:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                    param.grad = None
            
            # Setup Optimizer for active params
            filtered_groups = []
            for group in self.param_manager.get_parameter_groups(self.base_lr):
                active_in_group = [p for p in group['params'] if p.requires_grad]
                if active_in_group:
                    filtered_groups.append({'params': active_in_group, 'lr': group['lr']})
                    
            if not filtered_groups:
                print(f"[Engine] Warning: No active parameters for {stage_name}. Skipping.")
                continue
                
            self.optimizer = optim.Adam(filtered_groups)
            loss_weights = self.loss_manager.get_stage_weights(active_params, stage)
            self.convergence_guard.reset()
            
            # --- Optimization Loop ---
            for step in range(stage_steps):
                self.optimizer.zero_grad()
                self.param_manager.update_model_config()
                
                # Forward Pass
                seed = 42 if stage['seed_mode'] == 'locked' else None
                pred_img = self.model(seed=seed) / 255.0 
                
                # Compute Loss
                final_loss = 0.0
                for name, criterion in self.losses.items():
                    w = loss_weights.get(name, 0.0)
                    if w > 0:
                        final_loss += criterion(pred_img, target_img) * w
                
                # Backward Pass & Robustness Check
                final_loss.backward()
                if not self.robustness_guard.check_gradients():
                    print("[Engine] Bad gradients detected. Skipping step.")
                    continue
                    
                self.optimizer.step()
                
                # Convergence Check & Logging
                velocities = self.convergence_guard.update(self.param_manager, active_params)
                
                # Monitor ALL losses for debugging (Phase 3.7.1)
                monitor_losses = {}
                with torch.no_grad():
                     for name, criterion in self.losses.items():
                         monitor_losses[name] = criterion(pred_img, target_img).item()
                
                # Update progress callback every step (for CSV logging)
                if progress_callback:
                    current_vals = self.param_manager.get_physical_values()
                    progress_callback(global_step, self.max_steps, {
                        'step': global_step, 
                        'loss': final_loss.item(),
                        'stage': stage_name,
                        'params': current_vals,
                        'velocities': velocities,
                        'monitor_losses': monitor_losses # Pass full breakdown
                    })

                if step % 5 == 0:
                    current_vals = self.param_manager.get_physical_values()
                    
                    # Format params for display
                    params_str = " | ".join([f"{n.split('.')[-1]}: {v:.4f}" for n, v in current_vals.items()])
                    loss_str = " | ".join([f"{k}: {v:.4f}" for k,v in monitor_losses.items()])
                    
                    log_msg = f"Step {global_step} (S{stage_idx}:{step}): TotalLoss={final_loss.item():.4f}\n" \
                              f"         Params: {params_str}\n" \
                              f"         Losses: {loss_str}"
                    print(log_msg)
                
                if step > 20 and self.convergence_guard.check_convergence():
                    print(f"[Engine] Stage {stage_name} Converged early at step {step}!")
                    break
                    
                global_step += 1
                
        print("\n[Engine] Calibration Complete.")
        return self.param_manager.get_physical_values()

    def evaluate_loss(self, target_img: torch.Tensor, n_samples: int = 1):
        """
        Evaluate losses between current model state and target image.
        Supports multiple stochastic samples to estimate loss distribution.
        """
        results = {name: [] for name in self.losses.keys()}
        results['total'] = []
        
        with torch.no_grad():
            self.param_manager.update_model_config()
            for i in range(n_samples):
                # Random seed for robust evaluation if n_samples > 1 or seed is None
                # DiffOSOG uses seed=None for random
                pred_img = self.model(seed=None) / 255.0
                
                total_loss = 0.0
                for name, criterion in self.losses.items():
                    val = criterion(pred_img, target_img).item()
                    results[name].append(val)
                    total_loss += val # Simple sum for total metric
                results['total'].append(total_loss)
                
        # Compute stats
        stats = {}
        for k, vals in results.items():
            stats[k] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)) if len(vals) > 1 else 0.0,
                'values': vals # Return all for histogram if needed
            }
        return stats