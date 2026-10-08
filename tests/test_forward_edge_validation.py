"""First-snapshot NFL edge grading invariants: no hindsight, no implicit fills."""
from __future__ import annotations

import csv
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from nfl.forward_edge_validation import (
    SEASON,
    _ci,
    append_forward_candidates,
    grade_forward_candidates,
    load_forward_candidates,
    write_forward_validation,
)

NOW = datetime(2026, 10, 7, 18, tzinfo=UTC)
KICKOFF = NOW + timedelta(days=4)


def candidate(**kwargs):
    row = {
        "season": SEASON, "week": 5,
        "game_id": "2026_05_NYG_WAS",
        "home_team": "WAS", "away_team": "NYG",
        "kickoff": KICKOFF.isoformat(),
        "quant_market": "spread", "quant_side": "home",
        "quant_book": "DraftKings Sportsbook", "quant_price": -3.5,
        "quant_odds": -110, "quant_quote_at": (NOW - timedelta(minutes=2)).isoformat(),
        "quant_probability": 0.62, "quant_market_probability": 0.50,
        "quant_edge": 0.12,
        "quant_ev": 0.62 * (1 + 100 / 110) - 1,
        "market_execution_verified": True,
        "market_quote_timestamp_verified": True,
        "market_quote_sanity_ok": True,
        "execution_ready": True,
        "portfolio_action": "PASS",
        "portfolio_stake_units": 0.0,
        "edge_discovery_tier": "EVIDENCE_OR_CONTEXT_BLOCKED",
        "edge_timing_action": "PASS",
        "probability_model_family": "nfl_gaussian",
    }
    row.update(kwargs)
    return row


def completed(**kwargs):
    row = {
        "game_id": "2026_05_NYG_WAS", "season": SEASON,
        "home_team": "WAS", "away_team": "NYG",
        "home_score": 28.0, "away_score": 21.0,
        "game_type": "REG",
    }
    row.update(kwargs)
    return row


def snapshot(**kwargs):
    stamp = KICKOFF - timedelta(minutes=20)
    row = {
        "game_id": "2026_05_NYG_WAS",
        "market_type": "spread",
        "book": "Draft Kings",
        "captured_at": stamp.isoformat(),
        "kickoff": KICKOFF.isoformat(),
        "first_side": "home", "first_line": -3.5,
        "first_american_odds": -125,
        "second_side": "away", "second_line": 3.5,
        "second_american_odds": 105,
    }
    row.update(kwargs)
    return row


def record(tmp_path, row=None):
    path = tmp_path / "future.csv"
    capture = append_forward_candidates(
        pl.DataFrame([row or candidate()]), path=path, captured_at=NOW
    )
    frame = load_forward_candidates(path)
    return frame, capture, path


def test_first_snapshot_frozen_and_later_shopping_cannot_replace(tmp_path):
    frame, summary, path = record(tmp_path)
    before = path.read_bytes()
    later = candidate(
        quant_price=-1.5,
        quant_probability=0.81,
        quant_ev=0.81 * (1 + 100 / 110) - 1,
        quant_edge=0.31,
    )
    updated = append_forward_candidates(
        pl.DataFrame([later]),
        path=path, captured_at=NOW + timedelta(minutes=40),
    )
    assert summary["appended_rows"] == 1
    assert updated["appended_rows"] == 0
    assert updated["capture_status_counts"]["previously_frozen"] == 1
    assert path.read_bytes() == before
    assert frame.height == 1


def test_capture_has_no_hindsight_backfill_or_2025_season(tmp_path):
    path = tmp_path / "edge.csv"
    result = append_forward_candidates(
        pl.DataFrame([candidate()]),
        captured_at=KICKOFF + timedelta(seconds=1), path=path,
    )
    assert result["appended_rows"] == 0
    assert not path.exists()
    result = append_forward_candidates(
        pl.DataFrame([candidate(season=2025)]),
        captured_at=NOW, path=path,
    )
    assert result["capture_status_counts"]["wrong_season"] == 1
    assert not path.exists()


def test_capture_retains_unverified_selection_as_denominator_not_valid_edge(tmp_path):
    frame, capture, _ = record(
        tmp_path, candidate(market_execution_verified=False)
    )
    assert capture["appended_rows"] == 1
    assert frame["snapshot_status"][0] == "UNVERIFIED_BOOK_OR_MARKET_PRICE"
    graded, result = grade_forward_candidates(frame, pl.DataFrame([completed()]))
    assert result["summary"]["frozen"] == 1
    assert result["summary"]["decided"] == 0
    assert graded["observation_status"][0] == "INVALID_FROZEN_QUOTE"


def test_first_frozen_quote_grades_only_matched_final_game(tmp_path):
    ledger, _, _ = record(tmp_path)
    graded, report = grade_forward_candidates(
        ledger, pl.DataFrame([completed()]), snapshots=pl.DataFrame([snapshot()])
    )
    row = graded.to_dicts()[0]
    assert row["observation_status"] == "GRADED_PROBABILITY"
    assert row["result"] == "win"
    assert row["simulated_net_units"] == pytest.approx(100 / 110)
    assert row["model_brier"] == pytest.approx((0.62 - 1) ** 2)
    assert row["market_brier"] == pytest.approx(0.25)
    assert report["summary"]["brier_lift_vs_market"] > 0
    assert report["summary"]["hypothetical_archive_free_quote_roi"] > 0
    assert report["staking_authorized"] is False
    assert report["production_model_change_enabled"] is False
    assert row["portfolio_stake_units_at_capture"] == 0


def test_grade_skips_no_final_score_and_never_uses_live_predictions(tmp_path):
    ledger, _, _ = record(tmp_path)
    for empty in (pl.DataFrame([]), pl.DataFrame([completed(home_score=None)])):
        if empty.is_empty():
            continue
        graded, report = grade_forward_candidates(ledger, empty)
        assert graded["observation_status"][0] == "PENDING_RESULT"
        assert report["summary"]["decided"] == 0
        assert report["status"] == "PENDING_FORWARD"


def test_team_identity_and_season_mismatch_are_blocked(tmp_path):
    ledger, _, _ = record(tmp_path)
    wrong_team, _ = grade_forward_candidates(
        ledger, pl.DataFrame([completed(home_team="GB")])
    )
    assert wrong_team["observation_status"][0] == "TEAM_IDENTITY_MISMATCH"
    wrong_year, _ = grade_forward_candidates(
        ledger, pl.DataFrame([completed(season=2025)])
    )
    assert wrong_year["observation_status"][0] == "SEASON_MISMATCH"


def test_push_excluded_from_brier_not_simulated_payoff(tmp_path):
    ledger, _, _ = record(
        tmp_path,
        candidate(quant_price=-7.0),
    )
    grade, report = grade_forward_candidates(ledger, pl.DataFrame([completed()]))
    assert grade["result"][0] == "push"
    assert grade["simulated_net_units"][0] == 0.0
    assert grade["observation_status"][0] == "GRADED_PUSH_UNSCORED_PROBABILITY"
    assert report["summary"]["decided"] == 0
    assert report["summary"]["graded"] == 1
    assert report["summary"]["mean_model_brier"] is None


def test_no_cross_book_cross_game_or_post_kickoff_quote(tmp_path):
    ledger, _, _ = record(tmp_path)
    prices = [
        snapshot(book="FanDuel"),
        snapshot(game_id="2026_05_CHI_GB"),
        snapshot(captured_at=(KICKOFF + timedelta(minutes=1)).isoformat()),
        snapshot(captured_at=(KICKOFF - timedelta(hours=2)).isoformat()),
    ]
    grade, _ = grade_forward_candidates(
        ledger, pl.DataFrame([completed()]), snapshots=pl.DataFrame(prices)
    )
    assert grade["near_kickoff_status"][0] == "NOT_OBSERVED"


def test_near_kickoff_quote_not_official_close_and_same_line_only(tmp_path):
    ledger, _, _ = record(tmp_path)
    grade, summary = grade_forward_candidates(
        ledger, pl.DataFrame([completed()]),
        snapshots=pl.DataFrame([snapshot()]),
    )
    assert grade["near_kickoff_status"][0] == "SAME_HANDICAP_PRICE_OBSERVED"
    assert grade["near_kickoff_price_change_pp"][0] > 0
    assert summary["summary"]["near_kickoff_same_line"] == 1
    assert "not official close" in summary["late_quote_provenance"]


def test_handicap_change_never_reuses_same_line_probability(tmp_path):
    ledger, _, _ = record(tmp_path)
    changed = snapshot(first_line=-4.5, second_line=4.5)
    grade, _ = grade_forward_candidates(
        ledger, pl.DataFrame([completed()]),
        snapshots=pl.DataFrame([changed]),
    )
    assert grade["near_kickoff_status"][0] == "HANDICAP_CHANGED_PRICE_UTILITY_UNPRICED"
    assert grade["near_kickoff_line_change"][0] == 1.0
    assert grade["near_kickoff_price_change_pp"][0] is None


def test_market_no_vig_frozen_and_ev_arithmetic_reconciled(tmp_path):
    ledger, _, _ = record(
        tmp_path,
        candidate(quant_edge=0.25),
    )
    assert ledger["snapshot_status"][0] == "UNRECONCILED_PROBABILITY_OR_EV"
    grade, result = grade_forward_candidates(
        ledger, pl.DataFrame([completed()])
    )
    assert result["summary"]["graded"] == 0
    assert grade["observation_status"][0] == "INVALID_FROZEN_QUOTE"


def test_quote_time_freshness_missing_book_and_future_quote_fail_closed(tmp_path):
    for i, changes in enumerate((
        {"quant_quote_at": (NOW - timedelta(hours=3)).isoformat()},
        {"quant_quote_at": (NOW + timedelta(minutes=1)).isoformat()},
        {"quant_quote_at": "2026-10-07T18:00:00"},
        {"quant_book": ""},
        {"quant_odds": -95},
        {"market_quote_timestamp_verified": False},
        {"market_quote_sanity_ok": False},
        {"quant_probability": 1.3},
    )):
        frame, summary, _ = record(tmp_path / str(i), candidate(**changes))
        assert summary["appended_rows"] == 1
        assert frame["snapshot_status"][0] != "VALID_POINT_IN_TIME_RESEARCH_QUOTE"


def test_freeze_home_side_moneyline_and_under_market_validity(tmp_path):
    rows = [
        candidate(quant_market="moneyline", quant_price=None, quant_odds=140,
                  quant_probability=0.48, quant_market_probability=0.43,
                  quant_edge=0.05, quant_ev=0.48 * 2.4 - 1),
        candidate(quant_market="total", quant_side="under", quant_price=48.5),
    ]
    path = tmp_path / "forward.csv"
    summary = append_forward_candidates(
        pl.DataFrame(rows), path=path, captured_at=NOW
    )
    assert summary["appended_rows"] == 2
    assert set(load_forward_candidates(path)["snapshot_status"]) == {
        "VALID_POINT_IN_TIME_RESEARCH_QUOTE"
    }


def test_bootstrap_week_blocks_only_after_minimum_new_sample():
    rows = [
        {"game_id": f"g{i}", "season": 2026, "week": i // 10 + 1,
         "lift": 0.02 if i % 2 else 0.04}
        for i in range(120)
    ]
    assert _ci(rows, "lift") == _ci(rows, "lift")
    assert _ci(rows, "lift")[0] > 0
    assert _ci(rows[:99], "lift") is None
    collapse = [dict(row, week=row["week"] % 7 + 1) for row in rows]
    assert _ci(collapse, "lift") is None
    same_game = [dict(row, game_id=f"g{j // 4}") for j, row in enumerate(rows)]
    assert _ci(same_game, "lift") is None


def test_duplicate_ledger_key_cannot_create_double_forward_success(tmp_path):
    ledger, _, _ = record(tmp_path)
    dupes = pl.concat([ledger, ledger])
    graded, report = grade_forward_candidates(dupes, pl.DataFrame([completed()]))
    assert report["status"] == "INVALID_LEDGER_DUPLICATES_FAIL_CLOSED"
    assert report["duplicate_frozen_keys"] == 1
    assert report["summary"]["decided"] == 1
    assert report["promotion_eligible"] is False


def test_empty_ledger_outputs_and_research_policy_unchanged(tmp_path):
    empty, status = grade_forward_candidates(pl.DataFrame(), pl.DataFrame())
    assert empty.is_empty()
    assert status["status"] == "PENDING_FORWARD"
    write_forward_validation(
        empty, status,
        report_dir=tmp_path / "reports",
        docs_dir=tmp_path / "docs",
        outputs_dir=tmp_path / "outputs",
    )
    for filename in ("forward_edge_validation.json", "forward_edge_graded.csv"):
        b = (tmp_path / "reports" / filename).read_bytes()
        assert b == (tmp_path / "docs" / filename).read_bytes()
        assert b == (tmp_path / "outputs" / filename).read_bytes()
    with (tmp_path / "reports" / "forward_edge_graded.csv").open() as handle:
        assert list(csv.DictReader(handle)) == []
    payload = json.loads(
        (tmp_path / "reports" / "forward_edge_validation.json").read_text()
    )
    assert payload["staking_authorized"] is False

def test_grading_follows_completed_main_model_run_and_reads_latest_commit():
    workflow = (
        Path(__file__).resolve().parents[1]
        / ".github/workflows/live-grade.yml"
    ).read_text(encoding="utf-8")
    assert 'workflows: ["NFL Model + Operations"]' in workflow
    assert "types: [completed]" in workflow
    assert "github.event.workflow_run.conclusion == 'success'" in workflow
    assert "github.event.workflow_run.head_branch == 'main'" in workflow
    assert "ref: ${{ github.event_name == 'workflow_run' && 'main' || github.ref }}" in workflow
    assert "run: python grade_forward_edge.py" in workflow
