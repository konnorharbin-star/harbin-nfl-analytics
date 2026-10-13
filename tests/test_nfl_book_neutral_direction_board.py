"""Synthetic matchup research tests: no book lines, payouts, or result backfills."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from scripts.nfl_book_neutral_direction_board import build_board

NOW = datetime(2026, 10, 12, 18, tzinfo=UTC)


def forecasts():
    kickoff = (NOW + timedelta(days=6)).isoformat()
    e = {
        "game_id": "2026_06_WAS_SF", "season": 2026, "week": 6,
        "home_team": "SF", "away_team": "WAS", "kickoff_utc": kickoff,
        "first_observed_at_utc": (NOW - timedelta(hours=2)).isoformat(),
        "p_home_win": 0.724, "p_away_win": 0.272, "p_tie_push": .004,
        "bookmaker_account_acceptance": False,
    }
    q = {
        "game_id": "2026_06_WAS_SF", "season": 2026, "week": 6,
        "home": "SF", "away": "WAS", "kickoff_utc": kickoff,
        "first_observed_at_utc": (NOW - timedelta(hours=1)).isoformat(),
        "p_home_win": .764, "p_away_win": .232, "p_tie_push": .004,
        "QB_proxy_status": "LAST_OBSERVED_NOT_VERIFIED_STARTER",
        "wager": False,
    }
    return e, q


def test_agreement_reports_premarket_pick_without_bet_claim():
    e, q = forecasts()
    out = build_board([e], [q], now=NOW)
    row = out["games"][0]
    assert out["top_directional_research_game"] == "2026_06_WAS_SF"
    assert row["consensus_pick"] == "SF"
    assert row["descriptive_min_directional_probability"] == .724
    assert row["p_win_as_bet"] is None
    assert row["estimated_ev"] is None
    assert row["market_quote"] is None
    assert row["wager_authorized"] is False
    assert out["approved_bets"] == 0
    assert out["all_books_optional_for_direction"] is True


def test_model_disagreement_does_not_force_pick():
    e, q = forecasts()
    q["p_home_win"], q["p_away_win"] = .37, .626
    out = build_board([e], [q], now=NOW)
    assert out["games"][0]["consensus_pick"] is None
    assert out["top_directional_research_game"] is None
    assert out["approved_bets"] == 0


def test_rejects_changed_identity_and_late_predictions():
    for mutation in (
        {"home": "OTHER"},
        {"kickoff_utc": (NOW + timedelta(days=2)).isoformat()},
        {"first_observed_at_utc": (NOW + timedelta(minutes=1)).isoformat()},
        {"p_home_win": 1.2},
        {"p_home_win": .999},
        {"wager": True},
    ):
        e, q = forecasts()
        q.update(mutation)
        report = build_board([e], [q], now=NOW)
        assert not report["games"], mutation
        assert report["rejected"], mutation


def test_no_future_matchup_is_invented_from_historical_games():
    e, q = forecasts()
    later = NOW + timedelta(days=20)
    out = build_board([e], [q], now=later)
    assert not out["games"]
    assert out["top_directional_research_game"] is None


def test_missing_original_match_and_duplicates_are_blocked():
    e, q = forecasts()
    out = build_board([e, e], [], now=NOW)
    assert out["games"] == []
    assert len(out["rejected"]) == 2


def test_invalid_reference_clock_rejected():
    e, q = forecasts()
    with pytest.raises(ValueError, match="timezone-aware"):
        build_board([e], [q], now=NOW.replace(tzinfo=None))


def test_actual_pregame_freeze_source_can_build_research_board():
    from pathlib import Path
    from scripts.nfl_book_neutral_direction_board import read_jsonl
    elo = read_jsonl(Path("history/moneyline_elo_v1_2026.jsonl"))
    qb = read_jsonl(Path("history/moneyline_weekly_qb_challenger_v2_2026.jsonl"))
    assert elo and qb
    out = build_board(elo, qb, now=NOW)
    assert out["games"] and len(out["games"]) <= 32
    assert all(r["estimated_ev"] is None and r["action"] == "PASS_RESEARCH_ONLY" for r in out["games"])
