import copy

import pytest

from scripts.chronological_pricing_research import arrays, evaluate, fit, select


def season_rows(season, *, useful=True):
    return [
        {
            "game_id": f"{season}-{i}",
            "season": season,
            "week": i // 10 + 1,
            "outcome": i % 2,
            "predictions": {
                "market": 0.5,
                "independent_model": (0.8 if i % 2 else 0.2) if useful else 0.5,
            },
        }
        for i in range(120)
    ]


def test_evaluation_outcomes_cannot_change_selected_parameters():
    rows = sum((season_rows(s) for s in (2022, 2023, 2024)), [])
    original = evaluate(rows)
    changed = copy.deepcopy(rows)
    for row in changed:
        if row["season"] == 2024:
            row["outcome"] = 1 - row["outcome"]
    perturbed = evaluate(changed)
    assert original["folds"][0]["selections"] == perturbed["folds"][0]["selections"]
    assert original["metrics"] != perturbed["metrics"]
    assert original["folds"][0]["training_seasons"] == [2022]
    assert original["folds"][0]["tuning_season"] == 2023
    assert not original["promotion_eligible"]
    assert not original["betting_authorized"]


def test_zero_weight_when_tuning_does_not_improve():
    train = season_rows(2022)
    tune = season_rows(2023)
    for r in tune:
        r["outcome"] = 1 - r["outcome"]
    selection = select(train, tune, "football_residual")
    assert selection["weights"] == [0, 0]
    assert selection["reason"] == "REJECTED_ON_TUNING"


def test_insufficient_development_keeps_market():
    assert select(season_rows(2022)[:10], season_rows(2023), "football_residual")["weights"] == [
        0,
        0,
    ]


def test_market_only_fit_never_uses_model():
    rows = season_rows(2022)
    assert fit(rows, 10, residual=False) == [0, 0]
    assert 0 < fit(rows, 10, residual=True)[1] <= 1


def test_duplicate_game_rejected_across_seasons():
    rows = season_rows(2022)
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate(rows + [rows[0]])


@pytest.mark.parametrize("invalid", [0, 1, float("nan")])
def test_invalid_probabilities_rejected(invalid):
    rows = season_rows(2022)
    rows[0]["predictions"]["market"] = invalid
    with pytest.raises(ValueError):
        arrays(rows)
