"""Leakage and failure-to-improve regressions for Walters-style score research."""
import copy
import math

import pytest

from scripts.walters_score_calibration import evaluate, fit, grade


def sample(sport="nfl", effect=0.0):
    m, t = ("projected_home_margin", "projected_total") if sport == "nfl" else (
        "pred_margin_home", "pred_total")
    a, b = ("actual_home_margin", "actual_total") if sport == "nfl" else (
        "actual_margin_home", "actual_total")
    rows = []
    for season in (2023, 2024, 2025):
        for i in range(120):
            x = (i % 20) - 10
            total = 35 + (i % 20)
            rows.append({
                "game_id": f"{season}-{i}", "season": season, "week": i // 10 + 1,
                m: float(x), a: x + effect * (x / 10) + ((i % 3) - 1),
                t: float(total), b: total + effect + ((i % 3) - 1)
            })
    return rows


@pytest.mark.parametrize("sport", ["nfl", "cfb"])
def test_evaluation_outcomes_cannot_change_fitted_or_selected_coefficients(sport):
    rows = sample(sport, effect=2.0)
    baseline = evaluate(rows, sport, 2024, 2025)
    perturbed = copy.deepcopy(rows)
    for row in perturbed:
        if row["season"] == 2025:
            key = "actual_total"
            row[key] += 1000
    updated = evaluate(perturbed, sport, 2024, 2025)
    for target in ("margin", "total"):
        assert baseline["targets"][target]["selection"] == updated["targets"][target]["selection"]
    assert baseline["targets"]["total"]["evaluation_selected"] != updated["targets"]["total"]["evaluation_selected"]
    assert not updated["betting_authorized"]


def test_zero_update_when_tuning_is_adversarial():
    rows = sample(effect=3.0)
    for row in rows:
        if row["season"] == 2024:
            row["actual_total"] = row["projected_total"] - 30
            row["actual_home_margin"] = row["projected_home_margin"] - 30
    report = evaluate(rows, "nfl", 2024, 2025)
    assert all(report["targets"][target]["selection"]["family"] == "baseline"
               for target in ("margin", "total"))


def test_ridge_correction_is_finite():
    rows = sample(effect=2)
    for kind in ("intercept", "affine"):
        w = fit(rows[:120], "projected_total", "actual_total", kind, 100)
        assert all(math.isfinite(v) for v in w.values())
        assert grade(rows[120:240], "projected_total", "actual_total", w)["games"] == 120


def test_requires_independent_chronological_splits():
    with pytest.raises(ValueError, match="Evaluation"):
        evaluate(sample(), "nfl", 2025, 2024)
    with pytest.raises(ValueError, match="Insufficient"):
        evaluate(sample()[:125], "nfl", 2024, 2025)
