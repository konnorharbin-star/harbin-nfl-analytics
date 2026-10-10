"""Chronological and selection-invariance tests for matchup residual research."""
import copy
import csv

import pytest

from scripts.walters_matchup_residual import evaluate, load_rows, pregame_features, metrics


def example():
    rows = []
    for season in (2023, 2024, 2025):
        for week in range(1, 13):
            for index in range(10):
                a = f"Team{index*2}"
                b = f"Team{index*2+1}"
                baseline = 40.0 + (week % 3)
                margin = float((index % 5)-2)
                actual_total = baseline + (2 if index % 2 else -2) + (week % 2)
                actual_margin = margin + ((index % 3)-1)
                rows.append({"game_id": f"{season}-{week}-{index}",
                             "season": season, "week": week,
                             "home_team": a, "away_team": b,
                             "pred_margin": margin, "pred_total": baseline,
                             "actual_margin": actual_margin,
                             "actual_total": actual_total})
    return rows


def test_same_week_outcomes_cannot_leak_into_any_pregame_feature():
    rows = example()
    before = pregame_features(rows, 3)
    changed = copy.deepcopy(rows)
    changed[0]["actual_total"] += 900
    changed[0]["actual_margin"] -= 350
    after = pregame_features(changed, 3)
    same_week = [(a, b) for a, b in zip(before, after)
                 if a["season"] == 2023 and a["week"] == 1]
    assert same_week
    assert all(a["delta_total"] == b["delta_total"]
               and a["delta_margin"] == b["delta_margin"]
               for a, b in same_week)
    # The next week is allowed to incorporate the newly completed result.
    assert any(a["delta_total"] != b["delta_total"]
               for a, b in zip(before, after) if a["season"] == 2023 and a["week"] == 2)


@pytest.mark.parametrize("sport", ["nfl", "cfb"])
def test_evaluation_outcomes_cannot_select_weights(sport):
    data = example()
    original = evaluate(data, sport)
    changed = copy.deepcopy(data)
    for r in changed:
        if r["season"] == 2025:
            r["actual_margin"] += 1000
            r["actual_total"] -= 1000
    revised = evaluate(changed, sport)
    for target in ("margin", "total"):
        assert original["targets"][target]["selected"] == revised["targets"][target]["selected"]
    assert revised["model_promotion_authorized"] is False
    assert revised["betting_authorized"] is False


def test_future_season_changes_do_not_change_prior_year_features():
    data = example()
    original = pregame_features(data, 6)
    modified = copy.deepcopy(data)
    for r in modified:
        if r["season"] == 2025:
            r["actual_total"] += 999
    other = pregame_features(modified, 6)
    assert [(r["delta_total"],r["delta_margin"]) for r in original if r["season"] < 2025] == [
        (r["delta_total"],r["delta_margin"]) for r in other if r["season"] < 2025]


def test_shuffled_input_and_no_market_data_affect_projection():
    data = example()
    base = sorted(pregame_features(data, 10), key=lambda r: r["game_id"])
    shuffled = sorted(pregame_features(list(reversed(data)), 10), key=lambda r: r["game_id"])
    assert [(r["delta_total"],r["delta_margin"]) for r in base] == [
        (r["delta_total"],r["delta_margin"]) for r in shuffled]
    injected = [{**r, "market_total": 10000, "closing_spread": -999} for r in data]
    assert [(r["delta_total"],r["delta_margin"]) for r in base] == [
        (r["delta_total"],r["delta_margin"]) for r in sorted(
            pregame_features(injected, 10), key=lambda r: r["game_id"])]


def test_load_is_strict_about_missing_or_duplicate_ids(tmp_path):
    source = tmp_path / "sample.csv"
    mapping = [dict(game_id=x["game_id"], season=x["season"], week=x["week"],
                    home_team=x["home_team"], away_team=x["away_team"],
                    pred_margin_home=x["pred_margin"], pred_total=x["pred_total"],
                    actual_margin_home=x["actual_margin"], actual_total=x["actual_total"])
               for x in example()[:2]]
    with source.open("w", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=list(mapping[0]))
        writer.writeheader()
        writer.writerows(mapping)
    assert len(load_rows(source, "cfb")) == 2
    mapping[1]["game_id"] = mapping[0]["game_id"]
    with source.open("w", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=list(mapping[0]))
        writer.writeheader()
        writer.writerows(mapping)
    with pytest.raises(ValueError, match="Duplicate"):
        load_rows(source, "cfb")


def test_zero_weight_does_not_create_fictitious_gain():
    rows = pregame_features(example(), 3)
    s = metrics(rows, "total", 0.0)
    assert s["mae"] > 0 and s["games"] == len(rows)
    with pytest.raises(ValueError, match="Insufficient"):
        evaluate(example()[:1], "nfl")
