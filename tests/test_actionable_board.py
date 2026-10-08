"""The public NFL board must never present unvalidated raw EV as actionable."""
import polars as pl

from nfl.actionable_board import build_actionable_board


def _row(**overrides):
    row = {
        "season": 2026, "week": 5, "game_id": "2026_05_TB_DAL",
        "away_team": "TB", "home_team": "DAL",
        "model_margin_home": 2.74, "model_total": 50.57,
        "quant_market": "spread", "quant_side": "away",
        "quant_odds": 100, "quant_book": "BetRivers",
        "quant_quote_at": "2026-10-08T22:54:51Z",
        "quant_ev": 0.337, "quant_edge": 0.19,
        "execution_ready": True,
        "market_execution_verified": True,
        "market_quote_timestamp_verified": True,
        "market_quote_sanity_ok": True,
        "context_veto": False, "context_freshness_veto": False,
        "context_injuries_personnel_fresh": True,
        "qb_context_ready": True, "qb_certainty_veto": False,
        "probability_reliability_ready": True,
        "probability_reliability_veto": False,
        "regime_reliability_ready": True,
        "regime_reliability_status": "RELIABLE",
        "edge_shrunk_ev": 0.053, "edge_discovery_tier": "SUPPORTED_RESEARCH",
        "edge_market_shrinkage_status": "VALIDATED_SHRINKAGE",
        "market_disagreement_severity": "LOW",
        "market_dispersion_high": False,
        "recommendation_status": "ACTIVE",
        "portfolio_stake_units": 0.0,
    }
    row.update(overrides)
    return row


def test_all_research_checks_can_produce_review_only_never_bet():
    row = build_actionable_board(pl.DataFrame([_row()])).to_dicts()[0]
    assert row["projected_winner"] == "DAL"
    assert abs(row["predicted_home_score"] - 26.655) < 1e-9
    assert abs(row["predicted_away_score"] - 23.915) < 1e-9
    assert row["betting_action"] == "REVIEW_ONLY"
    assert row["quote_quality"] == "VERIFIED_QUOTE_RESEARCH_ONLY"
    assert row["evidence_status"] == "SUPPORTED_RESEARCH_NOT_PRODUCTION"
    assert row["conservative_ev"] == 0.053


def test_tonights_large_raw_ev_is_blocked_when_shrunk_ev_negative():
    tonight = _row(
        quant_ev=0.337,
        edge_shrunk_ev=-0.04347,
        edge_market_shrinkage_status="MARKET_ONLY_PREFERRED",
        regime_reliability_ready=False,
        regime_reliability_status="BLOCKED",
        context_injuries_personnel_fresh=False,
        context_freshness_veto=True,
        market_disagreement_severity="HIGH",
        edge_discovery_tier="EVIDENCE_OR_CONTEXT_BLOCKED",
    )
    entry = build_actionable_board(pl.DataFrame([tonight])).to_dicts()[0]
    assert entry["betting_action"] == "PASS"
    assert entry["quant_ev"] == 0.337   # raw historical projection retained for auditing
    assert entry["conservative_ev"] is None
    for reason in (
        "STALE_OR_UNKNOWN_INJURY_CONTEXT",
        "HISTORICAL_REGIME_NOT_VALIDATED",
        "HOLDOUT_PREFERS_MARKET",
        "NO_POSITIVE_HOLDOUT_SHRUNK_EV",
        "UNUSUAL_MARKET_DISAGREEMENT",
    ):
        assert reason in entry["evidence_blockers"]


def test_absent_quote_or_veto_never_promotes_review():
    candidates = [
        _row(game_id="noquote", quant_quote_at=None),
        _row(game_id="veto", context_veto=True),
        _row(game_id="qb", qb_context_ready=False),
        _row(game_id="unknown", context_injuries_personnel_fresh=None),
        _row(game_id="unpriced", quant_odds=None),
        _row(game_id="inactive", recommendation_status="EXPIRED"),
    ]
    board = build_actionable_board(pl.DataFrame(candidates))
    assert board["betting_action"].to_list() == ["PASS"] * 6
    assert board.filter(pl.col("game_id")=="noquote")["quote_quality"].item() == "NO_VERIFIED_QUOTE"


def test_nonzero_portfolio_stake_cannot_bypass_regime_or_injury_veto():
    entry = build_actionable_board(pl.DataFrame([
        _row(portfolio_stake_units=2.0, regime_reliability_ready=False)
    ])).to_dicts()[0]
    assert entry["betting_action"] == "PASS"
    assert entry["conservative_ev"] is None


def test_missing_evidence_columns_fail_closed_and_keep_score():
    basic = {k: v for k, v in _row().items() if k not in (
        "edge_shrunk_ev", "regime_reliability_ready", "regime_reliability_status",
        "edge_discovery_tier", "qb_context_ready",
    )}
    entry = build_actionable_board(pl.DataFrame([basic])).to_dicts()[0]
    assert entry["betting_action"] == "PASS"
    assert entry["predicted_home_score"] > entry["predicted_away_score"]


def test_without_market_keeps_game_projection():
    basic = {k: v for k, v in _row().items() if not k.startswith("quant_")}
    entry = build_actionable_board(pl.DataFrame([basic])).to_dicts()[0]
    assert entry["betting_action"] == "NO_VERIFIED_MARKET"
    assert entry["evidence_blockers"] == "no model market offer"
    assert entry["projected_winner"] == "DAL"
