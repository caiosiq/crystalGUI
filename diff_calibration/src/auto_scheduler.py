from typing import List, Dict, Optional
import json
import os

class AutoScheduler:
    """
    Automated Curriculum Scheduler for OSOG Calibration.
    
    Responsibilities:
    1. Analyze user-selected parameters.
    2. Build a multi-stage plan (Geometry -> Texture -> Fine-tune).
    3. Manage the active stage and transition logic.
    """
    def __init__(self, rules_path: str = None):
        # Load Rules
        if rules_path is None:
            rules_path = os.path.join(os.path.dirname(__file__), "../optimization_rules.json")
            
        with open(rules_path, 'r') as f:
            self.rules = json.load(f)
            
        self.plan = []
        self.current_stage_idx = 0
        
    def build_plan(self, selected_params: List[str]) -> List[Dict]:
        """
        Constructs a curriculum based on the active parameters.
        Returns a list of Stage Dicts.
        """
        # 1. Identify required stages
        has_geometry = False
        has_texture = False
        
        for name in selected_params:
            if name not in self.rules:
                continue # Skip unknown (or assume fine-tune only)
            
            stage = self.rules[name].get('stage', 'fine_tune')
            if stage == 'geometry':
                has_geometry = True
            elif stage == 'texture':
                has_texture = True
                
        plan = []
        
        # Helper to load stage config or defaults
        def get_stage_config(stage_key, default_name, default_seed, default_freeze):
            if "stages" in self.rules and stage_key in self.rules["stages"]:
                s = self.rules["stages"][stage_key]
                return {
                    "name": s.get("name", default_name),
                    "active_tags": [stage_key],
                    "freeze_others": s.get("freeze_others", default_freeze),
                    "seed_mode": s.get("seed_mode", default_seed),
                    "loss_weights": s.get("loss_weights", {}),
                    "steps_ratio": 0.4
                }
            else:
                # Fallback for old configs
                return {
                    "name": default_name,
                    "active_tags": [stage_key],
                    "freeze_others": default_freeze,
                    "seed_mode": default_seed,
                    "steps_ratio": 0.4
                }
        
        # 2. Stage 1: Geometry (Macro)
        if has_geometry:
            s = get_stage_config("geometry", "Geometry", "locked", True)
            s["steps_ratio"] = 0.4
            plan.append(s)
            
        # 3. Stage 2: Texture (Micro)
        if has_texture:
            s = get_stage_config("texture", "Texture", "locked", True)
            s["steps_ratio"] = 0.4
            plan.append(s)
            
        # 4. Stage 3: Fine-Tuning (Polishing)
        # CRITICAL CHANGE (Phase 3.5.4 Fix):
        # If we have both Geometry and Texture, Fine-Tuning is DANGEROUS.
        # Unfreezing Geometry (seed locked) while optimizing Texture (seed random) causes drift.
        # We only add Fine-Tuning if we are NOT mixing conflicting seed strategies.
        
        # Update: With new "stages" config, we can define Fine-Tuning seed mode explicitly.
        # But generally, mixing locked and random is still risky.
        # If Texture is locked now (which we did in optimization_rules.json), then Fine-Tuning is safe!
        
        should_finetune = True 
        # Previously we disabled it if mixed. Now if both are locked, it's fine.
        # But let's keep the logic simple: Always add fine-tune if requested, 
        # but let the stage config dictate behavior.
        
        # Check if fine-tune is implicit or explicit? 
        # Usually fine-tune runs on everything.
        
        # Logic: If we did both stages, add a fine-tune stage.
        # If we only did one, maybe we don't need a separate fine-tune stage? 
        # Actually, fine-tune is useful to relax "freeze_others".
        
        if has_geometry or has_texture:
             # Load Fine-Tune config
             s = get_stage_config("fine_tune", "Fine-Tuning", "locked", False)
             s["active_tags"] = ["geometry", "texture", "fine_tune"]
             s["steps_ratio"] = 0.2
             
             # If we are mixing seed modes (e.g. Geometry=Locked, Texture=Random), 
             # and Fine-Tune is Random, Geometry might drift.
             # But if user set Texture=Locked in config, then Fine-Tune=Locked is safe.
             plan.append(s)
            
        # 5. Normalize Step Ratios
        total_ratio = sum(s['steps_ratio'] for s in plan)
        if total_ratio > 0:
            for s in plan:
                s['steps_ratio'] /= total_ratio
        
        self.plan = plan
        self.current_stage_idx = 0
        return plan

    def get_current_stage(self) -> Dict:
        if self.current_stage_idx >= len(self.plan):
            return self.plan[-1] # Stay on last stage
        return self.plan[self.current_stage_idx]
        
    def advance_stage(self) -> bool:
        """Returns True if advanced, False if already finished."""
        if self.current_stage_idx < len(self.plan) - 1:
            self.current_stage_idx += 1
            print(f"[AutoScheduler] Advancing to Stage: {self.get_current_stage()['name']}")
            return True
        return False

    def get_active_params_for_stage(self, selected_params: List[str]) -> List[str]:
        """
        Returns the subset of selected_params that should be ACTIVE (gradient ON)
        in the current stage.
        """
        stage = self.get_current_stage()
        active_tags = stage['active_tags']
        freeze_others = stage['freeze_others']
        
        active_list = []
        
        for name in selected_params:
            if name not in self.rules:
                # Default behavior for unknown: Active if not strict freezing
                if not freeze_others:
                    active_list.append(name)
                continue
                
            param_stage = self.rules[name].get('stage', 'fine_tune')
            
            # If active tags includes 'all' or specific tag
            if param_stage in active_tags:
                active_list.append(name)
            elif not freeze_others:
                # If we are not freezing others (Fine-Tuning), include everything
                active_list.append(name)
                
        return active_list
