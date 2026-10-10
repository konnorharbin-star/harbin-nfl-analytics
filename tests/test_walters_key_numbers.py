"""Walters key-number distribution safeguards: training chronology and real pushes."""
import csv
import math
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.walters_key_number_distribution import (
    ALPHAS,
    LINES,
    bootstrap,
    distribution,
    evaluate,
    fit,
    gaussian_bins,
    load,
    outcomes,
)


def games(sport):
    first = 2022 if sport == "nfl" else 2023
    rows = []
    for year in range(first, 2026):
        for i in range(128):
            margin = (-12, -7, -3, 0, 3, 7, 13, 17)[i % 8]
            rows.append({
                "game_id": f"{sport}_{year}_{i}",
                "season": year,
                "week": 1 + i // 8,
                "pred": ((i % 9) - 4) * 1.7,
                "actual": margin + (year % 3 - 1),
            })
    return rows


def test_gaussian_discrete_bins_normalized_and_symmetric():
    p = gaussian_bins(0.0, 13.0)
    assert len(p) == 201
    assert sum(p) == pytest.approx(1.0, abs=1e-10)
    assert p[97] == pytest.approx(p[103])
    assert all(0 <= v <= 1 for v in p)


def test_integer_key_spreads_have_real_push_mass_and_no_fictitious_complements():
    model = fit(games("nfl")[:240])
    pmf = distribution(model, 3.0, 0.0)
    home_win, home_push, home_loss = outcomes(pmf, home_spread=-3.0)
    away_win, away_push, away_loss = outcomes(pmf, home_spread=3.0)
    assert home_push == pytest.approx(pmf[103])
    assert away_push == pytest.approx(pmf[97])
    assert home_win + home_push + home_loss == pytest.approx(1.0)
    assert away_win + away_push + away_loss == pytest.approx(1.0)
    assert home_push > 0 and away_push > 0
    assert outcomes(pmf, home_spread=2.5)[1] == 0
    with pytest.raises(ValueError, match="prespecified"):
        outcomes(pmf, home_spread=3.5)


def test_tilt_preserves_mass_and_key_probability_increases_when_ratio_exceeds_one():
    model = fit(games("nfl")[:240])
    model["multipliers"] = {3: 2.0, 7: 1.5}
    low = distribution(model, 0.0, 0.0)
    high = distribution(model, 0.0, 1.0)
    assert sum(high) == pytest.approx(1.0, abs=1e-10)
    assert high[97] + high[103] > low[97] + low[103]
    assert high[93] + high[107] > low[93] + low[107]
    assert distribution(model, 0.0, 0.0) == low
    with pytest.raises(ValueError, match="Unregistered"):
        distribution(model, 0.0, 0.3)


@pytest.mark.parametrize("sport", ["nfl", "cfb"])
def test_strict_season_split_and_holdout_outcome_independence(sport):
    rows = games(sport)
    baseline = evaluate(rows, sport)
    poisoned = deepcopy(rows)
    for row in poisoned:
        if row["season"] == 2025:
            row["actual"] = -row["actual"] + 2
    rerun = evaluate(poisoned, sport)
    for key in (
        "initial_training_multipliers",
        "tuning_candidates",
        "selected_strength_on_tuning",
        "refit_prior_season_multipliers",
    ):
        assert baseline[key] == rerun[key]
    assert baseline["evaluation_baseline"] != rerun["evaluation_baseline"]
    assert baseline["betting_authorized"] is False
    assert rerun["promotion_authorized"] is False
    assert baseline["price_source"] == "NO_SPORTSBOOK_PRICE_USED"
    assert baseline["pristine_holdout"] is False


def test_training_rows_cannot_be_shuffled_into_different_coefficients():
    original = games("nfl")[:240]
    permuted = list(reversed(original))
    first, second = fit(original), fit(permuted)
    assert first["multipliers"][3] == pytest.approx(second["multipliers"][3])
    assert first["multipliers"][7] == pytest.approx(second["multipliers"][7])
    assert first["sigma"] == pytest.approx(second["sigma"])


def test_csv_rejects_duplicate_games_nonnumeric_outcomes_and_missing_columns(tmp_path):
    source = Path(tmp_path) / "archive.csv"
    rows = [
        {"season": "2024", "week": "2", "game_id": "a",
         "projected_home_margin": "3.2", "actual_home_margin": "3",
         "market_line": "-3", "later_market_close": "-7"},
    ]
    def write():
        with source.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write()
    observed = load(source, "nfl")
    assert observed[0]["pred"] == 3.2
    assert "market_line" not in observed[0]
    rows.append(deepcopy(rows[0]))
    write()
    with pytest.raises(ValueError, match="Duplicate"):
        load(source, "nfl")
    rows.pop()
    rows[0]["actual_home_margin"] = "3.4"
    write()
    with pytest.raises(ValueError, match="integer"):
        load(source, "nfl")
    rows[0]["actual_home_margin"] = "nan"
    write()
    with pytest.raises(ValueError):
        load(source, "nfl")


def test_bootstrap_requires_exact_pairing():
    a = [{"game_id": str(i), "season": 2025, "week": 1 + i // 10,
          "log_loss": .45, "brier": .3} for i in range(90)]
    b = deepcopy(a)
    for row in b:
        row["log_loss"] -= .01
        row["brier"] -= .005
    result = bootstrap(a, b, repetitions=100)
    assert result["clusters"] == 9
    assert result["familywise_95_ci"]["log_loss"][0] > 0
    b[0]["game_id"] = "not-corresponding"
    with pytest.raises(ValueError, match="alignment"):
        bootstrap(a, b)


def test_single_model_reference_lines_include_key_and_half_points():
    assert 3.0 in LINES and 7.0 in LINES
    assert -3.0 in LINES and -7.0 in LINES
    assert -2.5 in LINES and 2.5 in LINES
    assert 0.0 in ALPHAS
    assert math.isfinite(fit(games("cfb")[:180])["sigma"])
