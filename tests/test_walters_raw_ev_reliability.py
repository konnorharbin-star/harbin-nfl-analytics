"""Historical raw-spread overconfidence research must remain non-actionable."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.walters_raw_ev_reliability import (
    ALPHAS,
    blend,
    economic,
    load,
    proper_scores,
    report,
)

COLUMNS = [
    "market_type", "game_id", "season", "week", "result",
    "model_probability", "no_vig_probability",
    "expected_value_per_unit", "net_units", "historical_provenance",
    "entry_timestamp_verified", "probability_model_family",
]


def synthetic():
    rows = []
    for year in (2024, 2025):
        for week in range(1, 18):
            for i in range(8):
                winner = (i + week) % 2 == 0
                rows.append({
                    "season": year, "week": week,
                    "game_id": f"{year}_{week}_{i}",
                    "result": "win" if winner else "loss",
                    "model_p": 0.75, "market_p": 0.50,
                    "raw_ev": 0.35, "net_units": 0.91 if winner else -1.0,
                    "provenance": "provider_labeled_open_close",
                    "entry_timestamp_verified": False,
                    "family": "older_model_not_2026_key_number",
                })
    return rows


def test_market_dominates_synthetic_overconfident_forecast():
    rows = synthetic()
    r = report(rows)
    assert r["chosen_alpha_2024_only"] == 0.0
    assert r["2025_probability_scores"]["market_only"]["brier"] == pytest.approx(.25)
    assert r["2025_probability_scores"]["full_raw_model"]["brier"] > .30
    assert r["2025_spread_archive_large_ev_tails"][
        "raw_ev_at_least_30_pct"
    ]["observations"] == 136
    assert r["model_promotion_allowed"] is False
    assert r["betting_authorized"] is False
    assert r["2025_paired_week_bootstrap"]["week_clusters"] == 17
    assert r["2025_paired_week_bootstrap"]["simultaneous_95_ci"]


def test_no_2025_data_in_locked_tuning_choice():
    rows = synthetic()
    original = report(rows)
    altered = [dict(x) for x in rows]
    for x in altered:
        if x["season"] == 2025:
            x["model_p"] = 0.01
            x["result"] = "loss"
            x["net_units"] = -1.0
    later = report(altered)
    assert later["chosen_alpha_2024_only"] == original[
        "chosen_alpha_2024_only"
    ]
    assert later["tuning_brier_by_alpha"] == original[
        "tuning_brier_by_alpha"
    ]
    assert later["2025_probability_scores"] != original[
        "2025_probability_scores"
    ]


def test_push_does_not_count_as_win_or_loss_in_probabilities():
    rows = synthetic()[:20]
    rows[0] = {**rows[0], "result": "push", "net_units": 0.0}
    s = proper_scores(rows, 0)
    assert s["resolved"] == 19
    assert s["pushes_excluded"] == 1
    assert economic(rows)["pushes"] == 1


def test_market_log_odds_weight_endpoints():
    row = synthetic()[0]
    assert blend(row, 0) == pytest.approx(row["market_p"])
    assert blend(row, 1) == pytest.approx(row["model_p"])
    assert ALPHAS[0] == 0 and ALPHAS[-1] == 1


def test_refuse_archival_time_as_verified_book_execution():
    rows = synthetic()
    rows[0] = {**rows[0], "entry_timestamp_verified": True}
    with pytest.raises(ValueError, match="unverified historical snapshots"):
        report(rows)
    rows[0] = {**rows[0], "entry_timestamp_verified": False}
    rows[0]["provenance"] = ""
    with pytest.raises(ValueError, match="unverified historical snapshots"):
        report(rows)


def test_require_separate_2024_and_2025_sets():
    with pytest.raises(ValueError, match="both 2024 tune and 2025"):
        report([x for x in synthetic() if x["season"] == 2025])


def test_archive_csv_loader_rejects_duplicate_and_bogus_push(tmp_path: Path):
    p = tmp_path / "archived.csv"
    record = {
        "market_type": "spread", "game_id": "2025_05_A_B",
        "season": "2025", "week": "5",
        "result": "push", "model_probability": ".61",
        "no_vig_probability": ".50", "expected_value_per_unit": ".2",
        "net_units": "0", "historical_provenance": "provider_labeled_open_close",
        "entry_timestamp_verified": "false",
        "probability_model_family": "older_model",
    }
    with p.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerow(record)
    assert len(load(p)) == 1
    with p.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writerow(record)
    with pytest.raises(ValueError, match="duplicate"):
        load(p)
    with p.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerow({**record, "net_units": "-1"})
    with pytest.raises(ValueError, match="Push did not settle"):
        load(p)


def test_archive_loader_refuses_new_year(tmp_path: Path):
    path = tmp_path / "future.csv"
    row = {
        "market_type": "spread", "game_id": "2026_05_A_B",
        "season": "2026", "week": "5",
        "result": "win", "model_probability": ".61",
        "no_vig_probability": ".50", "expected_value_per_unit": ".2",
        "net_units": ".9", "historical_provenance": "provider_labeled_open_close",
        "entry_timestamp_verified": "false", "probability_model_family": "older",
    }
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerow(row)
    with pytest.raises(ValueError, match="2024/2025 only"):
        load(path)
