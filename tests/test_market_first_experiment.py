import csv
import json
from datetime import UTC, datetime, timedelta

import pytest

from scripts.market_first_experiment import (
    capture,
    decimal,
    forecasts,
    immutable,
    nfl_history,
    scores,
    settle,
)

NOW = datetime(2026, 10, 10, 17, tzinfo=UTC)


def test_power_calibration_is_complementary_and_fixed():
    a, b = forecasts(0.7, 0.6), forecasts(0.3, 0.4)
    for key in a:
        assert a[key] + b[key] == pytest.approx(1)
    assert a["market_power_1_1"] > 0.7
    assert a["market_model_25pct"] == pytest.approx(0.675)
    with pytest.raises(ValueError):
        forecasts(float("nan"), 0.5)


@pytest.mark.parametrize("price", [0, 99, -99, 110.5, float("inf"), 20000])
def test_invalid_prices_are_not_reconstructed(price):
    assert decimal(price) is None


def test_duplicate_games_cannot_inflate_confidence():
    row = {
        "game_id": "x",
        "season": 2026,
        "week": 6,
        "outcome": 1,
        "predictions": forecasts(0.5, 0.6),
    }
    with pytest.raises(ValueError, match="Duplicate"):
        scores([row, row.copy()])


def test_sparse_evidence_has_no_interval():
    row = {
        "game_id": "x",
        "season": 2026,
        "week": 6,
        "outcome": 1,
        "predictions": forecasts(0.5, 0.6),
    }
    result = scores([row])
    assert not result["variants"]["market_power_1_1"]["interval_supported"]
    assert result["variants"]["market_power_1_1"]["paired_familywise_95_brier_ci"] is None


def publication(tmp_path, monkeypatch, *, quoted_at=None):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/audit_snapshot.json").write_text(
        json.dumps({"generated_at": NOW.isoformat(), "identity": {"model_version": "test"}})
    )
    pair = {
        "home_ml": -110,
        "away_ml": 100,
        "provider": "DraftKings",
        "source": "espn_core",
        "captured_at": (quoted_at or NOW).isoformat(),
    }
    row = {
        "game_id": "x",
        "season": 2026,
        "week": 6,
        "home_team": "A",
        "away_team": "B",
        "date": (NOW + timedelta(hours=3)).isoformat(),
        "calibrated_home_probability": 0.6,
        "market_quotes_json": json.dumps([pair]),
    }
    write_publication(tmp_path, row)
    return row


def write_publication(tmp_path, row):
    with (tmp_path / "docs/latest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def test_all_games_freeze_even_without_a_selected_bet(tmp_path, monkeypatch):
    row = publication(tmp_path, monkeypatch)
    root = tmp_path / "history/market_first_v1"
    assert capture("cfb", root, NOW)["new_frozen"] == 1
    path = next((root / "forecasts").glob("*.json"))
    original = path.read_bytes()
    row["calibrated_home_probability"] = 0.8
    write_publication(tmp_path, row)
    assert capture("cfb", root, NOW + timedelta(minutes=1))["already_frozen"] == 1
    assert path.read_bytes() == original
    record = json.loads(original)
    assert record["stake_units"] == 0 and record["betting_authorized"] is False
    assert record["source_quote_origin_verified"] is False


@pytest.mark.parametrize("minutes", [-1, 121])
def test_future_and_stale_observations_excluded(tmp_path, monkeypatch, minutes):
    publication(tmp_path, monkeypatch, quoted_at=NOW - timedelta(minutes=minutes))
    counts = capture("cfb", tmp_path / "history", NOW)
    assert counts["missing_named_paired_market"] == 1
    assert counts.get("new_frozen", 0) == 0


def test_atomic_records_cannot_change(tmp_path):
    path = tmp_path / "frozen.json"
    assert immutable(path, {"p": 0.6})
    assert not immutable(path, {"p": 0.9})
    assert json.loads(path.read_text()) == {"p": 0.6}


def test_historical_side_is_restored_before_scoring(tmp_path):
    path = tmp_path / "bets.csv"
    path.write_text(
        "market_type,side,result,no_vig_probability,model_probability,game_id,season,week\n"
        "moneyline,away,loss,0.3,0.4,x,2025,1\n"
    )
    row = nfl_history(path)[0]
    assert row["outcome"] == 1
    assert row["predictions"]["market"] == pytest.approx(0.7)
    assert row["predictions"]["independent_model"] == pytest.approx(0.6)


def test_late_publication_cannot_be_graded(tmp_path, monkeypatch):
    publication(tmp_path, monkeypatch)
    root = tmp_path / "history"
    capture("cfb", root, NOW)
    from types import SimpleNamespace

    monkeypatch.setattr(
        "scripts.market_first_experiment.subprocess.run",
        lambda *a, **k: SimpleNamespace(stdout="sha " + (NOW + timedelta(hours=4)).isoformat()),
    )
    monkeypatch.setattr(
        "scripts.market_first_experiment.fetch",
        lambda *a: pytest.fail("Late forecast must not retrieve a final"),
    )
    errors = settle(root, NOW + timedelta(days=1))
    assert list(errors.values()) == ["Publication not before kickoff"]
    assert not list((root / "grades").glob("*.json"))


def test_edited_forecast_cannot_be_graded(tmp_path, monkeypatch):
    from types import SimpleNamespace

    publication(tmp_path, monkeypatch)
    root = tmp_path / "history"
    capture("cfb", root, NOW)
    monkeypatch.setattr(
        "scripts.market_first_experiment.subprocess.run",
        lambda *a, **k: SimpleNamespace(
            stdout="first " + NOW.isoformat() + "\nedit " + NOW.isoformat()
        ),
    )
    assert list(settle(root, NOW + timedelta(days=1)).values()) == [
        "Missing unique publication commit"
    ]


def test_verified_final_grades_original_forecast(tmp_path, monkeypatch):
    from types import SimpleNamespace

    publication(tmp_path, monkeypatch)
    root = tmp_path / "history"
    capture("cfb", root, NOW)
    monkeypatch.setattr(
        "scripts.market_first_experiment.subprocess.run",
        lambda *a, **k: SimpleNamespace(stdout="sha " + NOW.isoformat()),
    )
    monkeypatch.setattr("scripts.market_first_experiment.fetch", lambda *a: ({}, "source-url"))
    monkeypatch.setattr(
        "scripts.market_first_experiment.final_result",
        lambda *a: {"home_score": 20, "away_score": 10, "event_id": "x"},
    )
    assert settle(root, NOW + timedelta(days=1)) == {}
    path = next((root / "grades").glob("*.json"))
    grade = json.loads(path.read_text())
    assert grade["outcome"] == 1
    assert grade["pregame_publication_commit"] == "sha"
    assert grade["forecast_sha256"]
    original = path.read_bytes()
    settle(root, NOW + timedelta(days=2))
    assert path.read_bytes() == original
