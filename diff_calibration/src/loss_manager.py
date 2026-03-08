import json
import os
from typing import Dict, List

class LossManager:
    """
    Semantic Loss Routing.
    
    Responsibilities:
    1. Read 'loss_weights' from optimization rules for ACTIVE parameters.
    2. Aggregate weights (Max or Sum strategy).
    3. Return a dictionary of scalar weights for the loss engine.
    """
    def __init__(self, rules_path: str = None):
        # Load Rules
        if rules_path is None:
            rules_path = os.path.join(os.path.dirname(__file__), "../optimization_rules.json")
            
        with open(rules_path, 'r') as f:
            self.rules = json.load(f)
            
    def get_stage_weights(self, active_params: List[str], stage_config: Dict = None) -> Dict[str, float]:
        """
        Calculate loss weights for the current stage.
        Strategy: 
        1. If stage_config has explicit 'loss_weights', use those (Override Mode).
        2. Else, aggregate from active parameters (Legacy Mode).
        """
        # 1. Check for Stage-Level Override (New System)
        if stage_config and stage_config.get("loss_weights"):
            return stage_config["loss_weights"]
            
        # 2. Fallback to Parameter Aggregation (Old System)
        # Base weights (always have some minimal guidance)
        weights = {
            "vgg": 0.1, 
            "spectral": 0.1,
            "gram": 0.0,
            "histogram": 0.0
        }
        
        for name in active_params:
            if name not in self.rules:
                continue
                
            param_weights = self.rules[name].get('loss_weights', {})
            
            for loss_type, weight in param_weights.items():
                if loss_type not in weights:
                    weights[loss_type] = 0.0
                # Aggregation: MAX is safer than SUM to prevent exploding weights
                weights[loss_type] = max(weights[loss_type], weight)
                
        return weights
