from app.postprocess import compute_stats


def test_compute_stats_empty():
    out = compute_stats([])
    assert out["count"] == 0
    assert out["mean_area"] == 0.0
    assert out["mean_aspect_ratio"] == 0.0
    assert out["areas"] == []
    assert out["aspect_ratios"] == []


def test_compute_stats_basic():
    dets = [
        {"w": 2.0, "h": 3.0},
        {"w": 4.0, "h": 5.0},
    ]
    out = compute_stats(dets)
    assert out["count"] == 2
    assert out["areas"] == [6.0, 20.0]
    assert abs(out["mean_area"] - 13.0) < 1e-6
    # aspect ratios: 2/3 and 4/5 -> mean ~ 0.5666...
    assert abs(out["mean_aspect_ratio"] - ((2/3)+(4/5))/2) < 1e-6