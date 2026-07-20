from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from tca_map.epoch9b_metrics import distribution_summary, nondecreasing, template_shift_at_center


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_template_metric_is_monotonic_for_deliberately_translated_object() -> None:
    initial = np.full((96, 96, 3), 170, dtype=np.uint8)
    yy, xx = np.mgrid[-7:8, -7:8]
    textured = np.clip(50 + 3 * (xx + 7) + 2 * (yy + 7), 0, 255).astype(np.uint8)
    mask = xx * xx + yy * yy <= 42
    patch = initial[41:56, 41:56].copy()
    patch[mask] = np.stack([textured[mask], textured[mask] // 2, 220 - textured[mask]], axis=1)
    initial[41:56, 41:56] = patch

    magnitudes = []
    for displacement in (0, 2, 4, 6, 8):
        translated = np.full_like(initial, 170)
        translated[41:56, 41 + displacement : 56 + displacement] = patch
        metric = template_shift_at_center(initial, translated, (48, 48), radius=7, search=12)
        magnitudes.append(metric["magnitude_pixels"])
        assert metric["dx"] == float(displacement)
        assert metric["dy"] == 0.0
        assert abs(metric["subpixel_dx"] - displacement) < 0.25
        assert abs(metric["subpixel_dy"]) < 0.25
    assert nondecreasing(magnitudes)
    assert magnitudes[-1] > magnitudes[0]


def test_preserved_front_crop_failure_is_detected() -> None:
    root = REPO_ROOT / "reports/epoch9_relational_probe_dataset/development/rotation3_demo37_diagnostic/frames"
    initial = np.asarray(
        Image.open(root / "development_demo37_front1_back8_front-first_front_initial.png").convert("RGB")
    )
    final = np.asarray(
        Image.open(root / "development_demo37_front1_back8_front-first_front_final.png").convert("RGB")
    )
    legacy = template_shift_at_center(initial, final, (95, 62), radius=8, search=12)
    audited = template_shift_at_center(initial, final, (92, 28), radius=8, search=18)
    assert legacy["magnitude_pixels"] == 0.0
    assert legacy["quality"] > 0.99
    assert audited["magnitude_pixels"] >= 5.0
    assert audited["quality"] > 0.5


def test_distribution_summary_reports_tail_and_uncertainty() -> None:
    summary = distribution_summary([0.01, 0.02, 0.04, 0.05], threshold=0.03, bootstrap_draws=200)
    assert summary["count"] == 4
    assert summary["count_above_threshold"] == 2
    assert summary["fraction_above_threshold"] == 0.5
    assert summary["median"] == 0.03
    assert summary["quantiles"]["q95"] > summary["quantiles"]["q05"]
    assert summary["mean_bootstrap_ci95"][0] <= summary["mean"] <= summary["mean_bootstrap_ci95"][1]
