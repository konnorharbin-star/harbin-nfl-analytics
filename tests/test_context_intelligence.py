"""NFL Step 6: injury source integrity, context stress and frozen game state."""
from __future__ import annotations

import csv
import json
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from nfl.context_intelligence import (
    append_first_context_snapshots,
    build_context_intelligence,
    write_context_intelligence,
)
from nfl.contracts import DataContractError
from nfl.injuries import injury_severity

NOW = datetime(2026, 10, 7, 20, tzinfo=UTC)
KICKOFF = (NOW + timedelta(days=4)).isoformat()


def _candidate(market="spread", **kwargs):
    row = {
        "season": 2026, "week": 5,
        "game_id": "2026_05_NYG_WAS",
        "kickoff": KICKOFF, "home_team": "WAS", "away_team": "NYG",
        "quant_market": market,
        "quant_side": "home" if market != "total" else "over",
        "quant_book": "DraftKings", "quant_odds": -110,
        "quant_probability": .65, "quant_ev": .24,
        "quant_signal": "PASS", "portfolio_action": "PASS",
        "portfolio_stake_units": 0.0,
        "context_as_of": (NOW - timedelta(minutes=2)).isoformat(),
        "context_injuries_personnel_fresh": True,
        "context_rest_travel_available": True,
        "context_weather_stadium_available": True,
        "home_expected_qb_id": "qb-was",
        "away_expected_qb_id": "qb-nyg",
        "home_expected_qb_decision_ready": True,
        "away_expected_qb_decision_ready": True,
        "home_expected_qb_changed_from_last_observed": False,
        "away_expected_qb_changed_from_last_observed": False,
        "home_expected_qb_source": "depth_chart",
        "away_expected_qb_source": "depth_chart",
        "home_expected_qb_confidence": 0.9,
        "away_expected_qb_confidence": 0.9,
        "injury_feed_freshness_status": "FRESH",
        "injury_feed_latest_reported_at": (
            NOW - timedelta(hours=2)
        ).isoformat(),
        "home_injury_freshness_status": "FRESH",
        "away_injury_freshness_status": "FRESH",
        "home_depth_freshness_status": "FRESH",
        "away_depth_freshness_status": "FRESH",
        "home_roster_freshness_status": "FRESH",
        "away_roster_freshness_status": "FRESH",
        "home_injury_risk": .0, "away_injury_risk": 0.0,
        "home_starter_injury_risk": 0.0,
        "away_starter_injury_risk": 0.0,
        "home_ol_injury_risk": 0.0, "away_ol_injury_risk": .0,
        "home_skill_injury_risk": 0.0, "away_skill_injury_risk": 0.0,
        "home_defense_injury_risk": 0.0, "away_defense_injury_risk": 0.0,
        "home_rest_days": 7.0, "away_rest_days": 7.0,
        "away_travel_miles": 200.0, "home_travel_miles": 0.0,
        "away_timezone_shift_hours": 0.0,
        "neutral_site": False,
        "weather_error": None,
    }
    row.update(kwargs)
    return row


def _build(*rows):
    return build_context_intelligence(
        pl.DataFrame(list(rows)), as_of=NOW,
        source_meta={"status": "PARTIAL", "injury_feed_freshness": {"status": "FRESH"}},
    )


@pytest.mark.parametrize(
    ("status", "practice", "expected"),
    [
        ("Inactive", "Full Participation", 1.0),
        ("Out", "Full Participation", 1.0),
        ("Out for season", "Available", 1.0),
        ("Injured Reserve", "Full Participation", 1.0),
        ("Suspended", "Full Participation", 1.0),
        ("Questionable", "Full Participation", 0.45),
        ("Doubtful", "Full Participation", 0.8),
        ("Probable", "Full Participation", 0.1),
        ("Active", "Full Participation", 0.0),
        ("", "Did Not Participate", 0.4),
        ("", "Limited Participation", 0.25),
        ("", "Full Participation", 0.0),
    ],
)
def test_injury_status_precedence_never_downgrades_official_out(
    status, practice, expected
):
    assert injury_severity(status, practice) == pytest.approx(expected)


def test_one_row_per_game_across_all_markets_without_stake_changes():
    candidates = [
        _candidate(market) for market in ("moneyline", "spread", "total")
    ]
    rows, report = _build(*candidates)
    assert len(rows) == 1
    assert rows[0]["matched_market_candidates"] == 3
    assert rows[0]["status"] == "OBSERVED_RESEARCH_ONLY"
    assert rows[0]["qb_decision_ready"]
    assert rows[0]["personnel_fresh"]
    assert rows[0]["rest_travel_available"]
    assert rows[0]["research_staking_authorized"] is False
    assert report["market_candidate_rows"] == 3
    assert report["ready_research_games"] == 1
    assert report["score_adjustment_enabled"] is not True if (
        "score_adjustment_enabled" in report
    ) else report["new_moneyline_spread_total_adjustment_enabled"] is False
    assert report["betting_authorized"] is False
    for row in candidates:
        assert row["quant_signal"] == "PASS"
        assert row["portfolio_stake_units"] == 0


def test_missing_team_injury_state_cannot_be_hidden_by_fresh_league_feed():
    rows, report = _build(
        _candidate(
            away_injury_freshness_status="UNKNOWN",
            context_injuries_personnel_fresh=False,
            context_freshness_reason="NYG injury state unknown",
        )
    )
    row = rows[0]
    assert row["status"] == "BLOCKED_CONTEXT_PROVENANCE"
    assert not row["personnel_fresh"]
    assert "AWAY_INJURY_REPORT_UNVERIFIED" in row["readiness_blockers"]
    assert "PERSONNEL_RISK_SCORES_NOT_CURRENT" in row["flags"]
    assert row["home_ol_vs_away_defense_availability"] is None
    assert report["blocked_games"] == 1
    assert report["source_injury_feed_status"] == "FRESH"


def test_live_untimed_injuries_and_partial_qb_context_explain_zero_readiness():
    rows, report = _build(
        _candidate(
            injury_feed_freshness_status="UNKNOWN",
            home_injury_freshness_status="UNKNOWN",
            away_injury_freshness_status="UNKNOWN",
            context_injuries_personnel_fresh=False,
            home_expected_qb_decision_ready=False,
            home_expected_qb_changed_from_last_observed=True,
        )
    )
    row = rows[0]
    assert "INJURY_FEED_UNTIMED_OR_STALE" in row["readiness_blockers"]
    assert "QB_IDENTITY_OR_STARTER_CERTAINTY_UNRESOLVED" in row["readiness_blockers"]
    assert "HOME_QB_CHANGE_UNREPRICED" in row["flags"]
    assert not row["qb_decision_ready"]
    assert report["ready_research_games"] == 0
    assert not report["betting_authorized"]


def test_descriptive_ol_matchup_qb_change_rest_travel_stack_not_score_adj():
    rows, report = _build(
        _candidate(
            home_expected_qb_changed_from_last_observed=True,
            home_ol_injury_risk=0.40,
            away_ol_injury_risk=0.2,
            away_defense_injury_risk=0.1,
            home_defense_injury_risk=0.0,
            home_skill_injury_risk=0.3,
            away_rest_days=5.0, away_travel_miles=1800.0,
            away_timezone_shift_hours=2.0,
        )
    )
    row = rows[0]
    assert row["home_ol_vs_away_defense_availability"] == pytest.approx(.3)
    assert row["away_ol_vs_home_defense_availability"] == pytest.approx(.2)
    assert row["stacked_qb_ol"]
    assert row["stacked_rest_travel"]
    assert "QB_CHANGE_PLUS_OL_STRESS" in row["flags"]
    assert "AWAY_SHORT_REST_PLUS_LONG_TRAVEL" in row["flags"]
    assert "HOME_SKILL_HEALTH_STRESS" in row["flags"]
    assert "AWAY_TIMEZONE_SHIFT" in row["flags"]
    assert report["new_moneyline_spread_total_adjustment_enabled"] is False


def test_weather_unknown_is_visible_but_does_not_fabricate_wind_effect():
    rows, report = _build(
        _candidate(
            context_weather_stadium_available=False,
            weather_error="unverified forecast",
        )
    )
    row = rows[0]
    assert row["status"] == "OBSERVED_RESEARCH_ONLY"
    assert "WEATHER_OR_VENUE_CONTEXT_UNKNOWN" in row["flags"]
    assert row["weather_available"] is False
    assert report["weather_venue_ready_games"] == 0


def test_market_inconsistency_causes_forensic_block_not_best_row_selection():
    rows, report = _build(
        _candidate("moneyline"),
        _candidate("spread", home_ol_injury_risk=.55),
    )
    assert rows[0]["status"] == "BLOCKED_CONTEXT_PROVENANCE"
    assert "CONTEXT_DISAGREEMENT_BETWEEN_MARKETS" in rows[0]["readiness_blockers"]
    assert report["blocked_games"] == 1


def test_missing_context_join_timestamp_and_kickoff_are_not_presented_current():
    rows, _ = _build(
        _candidate(
            context_as_of=(NOW + timedelta(minutes=5)).isoformat(),
        )
    )
    assert rows[0]["source_status"] == "UNKNOWN"
    assert "UNKNOWN_OR_FUTURE_CONTEXT_TIMESTAMP" in rows[0]["readiness_blockers"]
    after, _ = _build(
        _candidate(kickoff=(NOW - timedelta(minutes=1)).isoformat())
    )
    assert "NO_VALID_FUTURE_KICKOFF" in after[0]["readiness_blockers"]


def test_first_game_context_snapshot_frozen_despite_later_changed_qb(tmp_path):
    path = tmp_path / "history" / "context.csv"
    rows, _ = _build(_candidate())
    first = append_first_context_snapshots(rows, path=path, as_of=NOW)
    assert first["appended"] == 1
    before = path.read_bytes()
    later, _ = build_context_intelligence(
        pl.DataFrame([_candidate(
            home_expected_qb_id="replacement",
            home_expected_qb_changed_from_last_observed=True,
        )]), as_of=NOW + timedelta(minutes=30),
    )
    second = append_first_context_snapshots(
        later, path=path, as_of=NOW + timedelta(minutes=30)
    )
    assert second["appended"] == 0
    assert second["skipped"]["previously_frozen"] == 1
    assert path.read_bytes() == before
    with path.open() as stream:
        frozen = list(csv.DictReader(stream))
    assert frozen[0]["home_qb_id"] == "qb-was"
    assert frozen[0]["status"] == "OBSERVED_RESEARCH_ONLY"


def test_no_after_kickoff_or_untrusted_future_context_is_backfilled(tmp_path):
    path = tmp_path / "contexts.csv"
    rows, _ = _build(_candidate())
    results = append_first_context_snapshots(
        rows, path=path, as_of=NOW + timedelta(days=5),
    )
    assert results["appended"] == 0
    assert not path.exists()
    future, _ = _build(_candidate(
        context_as_of=(NOW + timedelta(days=1)).isoformat()
    ))
    blocked = append_first_context_snapshots(future, path=path, as_of=NOW)
    assert blocked["appended"] == 0
    assert blocked["skipped"]["context_time_unknown_or_future"] == 1


def test_duplicate_frozen_game_keys_fail_closed(tmp_path):
    path = tmp_path / "frozen.csv"
    rows, _ = _build(_candidate())
    append_first_context_snapshots(rows, path=path, as_of=NOW)
    content = path.read_text()
    path.write_text(content + content.splitlines()[1] + "\n")
    with pytest.raises(DataContractError, match="duplicate"):
        append_first_context_snapshots(rows, path=path, as_of=NOW)


def test_empty_context_and_identical_output_artifacts(tmp_path):
    rows, report = build_context_intelligence(pl.DataFrame(), as_of=NOW)
    assert report["status"] == "NO_GAME_CONTEXT"
    assert report["games"] == 0
    write_context_intelligence(
        rows, report,
        outputs_dir=tmp_path / "outputs",
        docs_dir=tmp_path / "docs",
    )
    for name in ("context_intelligence_report.json", "context_risk_register.csv"):
        assert (tmp_path / "outputs" / name).read_bytes() == (
            tmp_path / "docs" / name
        ).read_bytes()
    with (tmp_path / "outputs" / "context_risk_register.csv").open() as stream:
        assert list(csv.DictReader(stream)) == []
    assert json.loads(
        (tmp_path / "outputs" / "context_intelligence_report.json").read_text()
    )["betting_authorized"] is False


def test_malformed_frozen_snapshot_as_of_never_passes_validation():
    with pytest.raises(ValueError, match="timezone"):
        build_context_intelligence(
            pl.DataFrame([_candidate()]),
            as_of=datetime(2026, 10, 7, 20),
        )
