"""Tests for simplified-crystal OBB label merge."""

import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from crystalGUI.osog.labels.merge import merge_obbs, overlap_fraction_smaller_box


def _box(cx, cy, L, W, angle_deg=0.0):
    rect = ((cx, cy), (L, W), angle_deg)
    corners = cv2.boxPoints(rect)
    return {
        "cx": cx,
        "cy": cy,
        "L": L,
        "W": W,
        "angle_deg": angle_deg,
        "corners": corners.tolist(),
        "group_id": 0,
    }


def test_overlap_fraction_smaller_box_high_when_mostly_overlapping():
    a = np.array(_box(100, 100, 80, 20)["corners"], dtype=np.float32)
    b = np.array(_box(105, 100, 80, 20)["corners"], dtype=np.float32)
    frac = overlap_fraction_smaller_box(a, b)
    assert frac > 0.7


def test_overlap_fraction_smaller_box_zero_when_disjoint():
    a = np.array(_box(50, 50, 40, 10)["corners"], dtype=np.float32)
    b = np.array(_box(200, 200, 40, 10)["corners"], dtype=np.float32)
    assert overlap_fraction_smaller_box(a, b) == 0.0


def test_merge_obbs_combines_overlapping_pair():
    obbs = [_box(100, 100, 80, 20), _box(108, 100, 80, 20)]
    merged = merge_obbs(obbs, overlap_threshold=0.3)
    assert len(merged) == 1
    assert merged[0].get("merge_count") == 2


def test_merge_obbs_keeps_disjoint_separate():
    obbs = [_box(50, 50, 40, 10), _box(300, 300, 40, 10)]
    merged = merge_obbs(obbs, overlap_threshold=0.3)
    assert len(merged) == 2


def test_merge_by_group_id_blocks_cross_group():
    a = _box(100, 100, 80, 20)
    b = _box(105, 100, 80, 20)
    a["group_id"] = 1
    b["group_id"] = 2
    merged = merge_obbs([a, b], overlap_threshold=0.3, merge_by_group_id=True)
    assert len(merged) == 2


def test_synth_config_defaults_label_merge_off():
    from crystalGUI.osog.config import SynthConfig

    cfg = SynthConfig.from_dict({"canvas": {}, "physics": {}, "optics": {}, "sensor": {}})
    assert cfg.physics.label_merge.enable is False
    assert cfg.physics.label_merge.overlap_threshold == 0.4


def test_synth_config_preserves_label_merge_from_preset():
    from crystalGUI.osog.config import SynthConfig

    cfg = SynthConfig.from_dict(
        {
            "physics": {
                "label_merge": {
                    "enable": True,
                    "overlap_threshold": 0.55,
                    "merge_by_group_id": True,
                }
            }
        }
    )
    assert cfg.physics.label_merge.enable is True
    assert cfg.physics.label_merge.overlap_threshold == 0.55
    assert cfg.physics.label_merge.merge_by_group_id is True
