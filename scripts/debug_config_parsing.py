
import sys
import os
import json
from dataclasses import asdict

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from osog.config import SynthConfig

def test_config_parsing():
    # Simulate the JSON sent by playground.js
    # Note: rods.enable is True (legacy), rod_specs.enable is False (specific)
    payload = {
        "canvas": {"width": 1024, "height": 768, "use_gpu": True},
        "physics": {
            "rods": {
                "enable": True,
                "n_rods_rng_lo_hi": [50, 200, 200],
                "rod_len_px_lo_hi": [30, 380, 380],
                "rod_aspect_lo_hi": [0.02, 0.3, 0.3],
                "rod_delta_rng": [-12, 0, 0]
            },
            "use_specific_specs": True,
            "rod_specs": {
                "enable": False,
                "count_range": [50, 200],
                "length_range": [30, 380],
                "aspect_range": [0.02, 0.30],
                "delta_range": [-12, 0]
            },
            "sphere_specs": {
                "enable": False,
                "count_range": [10, 50],
                "diameter_range": [20, 100],
                "delta_range": [-12, 0]
            },
            "cube_specs": {"enable": False},
            "plate_specs": {"enable": False},
            "ghosts": {"enable": False},
            "debris": {"rate": 0.0}
        },
        "optics": {},
        "sensor": {}
    }

    print("--- Simulating Payload ---")
    print(json.dumps(payload, indent=2))

    # Parse
    cfg = SynthConfig.from_dict(payload)

    print("\n--- Parsed Config ---")
    print(f"Physics use_specific_specs: {cfg.physics.use_specific_specs}")
    print(f"Physics rods.enable (Legacy): {cfg.physics.rods.enable}")
    print(f"Physics rod_specs.enable (Specific): {cfg.physics.rod_specs.enable}")

    # Logic Check
    if cfg.physics.use_specific_specs:
        print("\n[OK] Backend WILL use specific specs.")
        if cfg.physics.rod_specs.enable:
             print("[FAIL] rod_specs.enable is True, but should be False!")
        else:
             print("[OK] rod_specs.enable is False. No rods should be generated.")
    else:
        print("\n[FAIL] Backend will use LEGACY mode (use_specific_specs is False).")
        if cfg.physics.rods.enable:
            print("       Legacy rods.enable is True -> Rods WILL be generated!")

if __name__ == "__main__":
    test_config_parsing()
