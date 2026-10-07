"""Policy-driven NFL signal classification and conservative stake sizing.

The structure intentionally mirrors the CFB production-policy layer, but every NFL
threshold remains NFL-specific. Sportsbook prices are downstream inputs only.
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path

from .market import american_to_decimal

DEFAULT_POLICY: dict[str, object] = {
    "version": 1,
    "deployment_mode": "paper",
    "markets": {
        "moneyline": {
            "enabled": True,
            "excluded_weeks": [],
            "lean": {"min_ev": 0.02, "min_edge": 0.015, "min_prob": 0.52},
            "bet": {"min_ev": 0.04, "min_edge": 0.025, "min_prob": 0.54},
            "strong": {"min_ev": 0.07, "min_edge": 0.040, "min_prob": 0.56},
        },
        "spread": {
            "enabled": True,
            "excluded_weeks": [],
            "lean": {"min_ev": 0.02, "min_edge": 0.020, "min_prob": 0.52},
            "bet": {"min_ev": 0.04, "min_edge": 0.030, "min_prob": 0.54},
            "strong": {"min_ev": 0.07, "min_edge": 0.050, "min_prob": 0.57},
        },
        "total": {
            "enabled": True,
            "excluded_weeks": [],
            "lean": {"min_ev": 0.02, "min_edge": 0.025, "min_prob": 0.52},
            "bet": {"min_ev": 0.04, "min_edge": 0.040, "min_prob": 0.54},
            "strong": {"min_ev": 0.07, "min_edge": 0.060, "min_prob": 0.57},
        },
    },
    "decision_intelligence": {
        "enabled": True,
        "enforce_execution_timing": False,
        "spread_disagreement_warn_points": 2.0,
        "spread_disagreement_high_points": 3.5,
        "total_disagreement_warn_points": 2.5,
        "total_disagreement_high_points": 4.5,
        "moneyline_disagreement_warn_probability": 0.05,
        "moneyline_disagreement_high_probability": 0.09,
        "spread_dispersion_warn_points": 0.75,
        "total_dispersion_warn_points": 1.0,
        "moneyline_dispersion_warn_probability": 0.035,
        "meaningful_line_move_points": 0.5,
        "meaningful_moneyline_decimal_move": 0.05,
        "spread_quote_outlier_points": 2.0,
        "total_quote_outlier_points": 3.0,
        "moneyline_quote_outlier_probability": 0.10,
    },
    "portfolio": {
        "max_slate_units": 5.0,
        "max_game_units": 1.0,
        "max_team_units": 1.5,
        "max_market_units": 2.5,
        "max_kickoff_window_units": 2.0,
        "kickoff_window_hours": 3,
        "max_book_units": 2.0,
        "max_bets": 20,
        "min_allocation_units": 0.05,
        "kelly_fraction": 0.20,
        "max_single_bet_units": 1.0,
        "drawdown_soft_stop_units": 8.0,
        "drawdown_hard_stop_units": 15.0,
        "drawdown_floor_multiplier": 0.25,
        "trailing_window_bets": 50,
        "min_trailing_bets_for_throttle": 30,
        "trailing_roi_throttle": -0.10,
        "trailing_clv_throttle": 0.0,
        "adverse_run_multiplier": 0.50,
        "enable_performance_feedback": True,
        "feedback_min_segment_bets": 20,
        "feedback_min_clv_coverage": 0.60,
        "feedback_weak_multiplier": 0.75,
        "feedback_severe_multiplier": 0.50,
        "feedback_severe_min_bets": 40,
        "feedback_severe_roi": -0.05,
        "feedback_severe_positive_clv_rate": 0.45,
        "require_executable_book": True,
        "min_market_book_count_for_execution": 1,
        "require_quote_timestamp_for_execution": True,
        "max_quote_age_minutes": 60,
        "max_public_recommendation_age_minutes": 45,
        "min_minutes_to_kickoff_for_execution": 5,
        "require_live_history_for_production": True,
        "require_open_exposure_ledger_for_production": True,
    },
    "source": "conservative NFL defaults; not promoted production thresholds",
}


def _default_copy() -> dict[str, object]:
    return deepcopy(DEFAULT_POLICY)


def load_policy(path: str | Path = "reports/production_policy.json") -> dict[str, object]:
    """Load an override policy while preserving all conservative defaults."""

    source = Path(path)
    if not source.exists():
        return _default_copy()
    try:
        incoming = json.loads(source.read_text())
    except (OSError, json.JSONDecodeError):
        return _default_copy()
    if not isinstance(incoming, dict):
        return _default_copy()

    policy = _default_copy()
    for key, value in incoming.items():
        if key not in {"markets", "portfolio", "decision_intelligence"}:
            policy[key] = value

    decision_intelligence = incoming.get("decision_intelligence")
    if isinstance(decision_intelligence, dict):
        assert isinstance(policy["decision_intelligence"], dict)
        policy["decision_intelligence"].update(decision_intelligence)

    portfolio = incoming.get("portfolio")
    if isinstance(portfolio, dict):
        assert isinstance(policy["portfolio"], dict)
        policy["portfolio"].update(portfolio)

    markets = incoming.get("markets")
    if isinstance(markets, dict):
        assert isinstance(policy["markets"], dict)
        for market_name, market_override in markets.items():
            if market_name not in policy["markets"] or not isinstance(market_override, dict):
                continue
            config = policy["markets"][market_name]
            assert isinstance(config, dict)
            for key, value in market_override.items():
                if key not in {"lean", "bet", "strong"}:
                    config[key] = value
            for tier in ("lean", "bet", "strong"):
                tier_override = market_override.get(tier)
                if isinstance(tier_override, dict):
                    assert isinstance(config[tier], dict)
                    config[tier].update(tier_override)
    return policy


def market_allowed(
    market: str,
    *,
    week: int | None = None,
    policy: dict[str, object] | None = None,
) -> tuple[bool, str]:
    active = policy or _default_copy()
    markets = active.get("markets")
    if not isinstance(markets, dict) or market not in markets:
        return False, "unsupported market"
    config = markets[market]
    if not isinstance(config, dict):
        return False, "invalid market policy"
    if not bool(config.get("enabled", True)):
        return False, str(config.get("disabled_reason") or "market disabled by policy")
    if week is not None:
        excluded = {int(value) for value in config.get("excluded_weeks", [])}
        if int(week) in excluded:
            return False, f"week {int(week)} excluded by validated policy"
    return True, "market enabled"


def signal_from_policy(
    expected_value: float,
    probability_edge: float,
    probability: float,
    market: str,
    *,
    week: int | None = None,
    policy: dict[str, object] | None = None,
) -> str:
    """Classify a downstream market opportunity as PASS/LEAN/BET/STRONG."""

    active = policy or _default_copy()
    allowed, _ = market_allowed(market, week=week, policy=active)
    if not allowed:
        return "PASS"
    finite_inputs = (expected_value, probability_edge, probability)
    if not all(math.isfinite(float(value)) for value in finite_inputs):
        return "PASS"
    if expected_value <= 0 or not 0.0 <= probability <= 1.0:
        return "PASS"

    markets = active["markets"]
    assert isinstance(markets, dict)
    config = markets[market]
    assert isinstance(config, dict)
    for tier in ("strong", "bet", "lean"):
        threshold = config[tier]
        assert isinstance(threshold, dict)
        if (
            expected_value >= float(threshold["min_ev"])
            and probability_edge >= float(threshold["min_edge"])
            and probability >= float(threshold["min_prob"])
        ):
            return tier.upper()
    return "PASS"


def fractional_kelly_units(
    probability: float,
    american_odds: int,
    *,
    kelly_fraction: float = 0.20,
    max_units: float = 1.0,
) -> float:
    """Return capped fractional-Kelly units; never return a negative stake."""

    if not 0.0 <= probability <= 1.0:
        return 0.0
    if not 0.0 <= kelly_fraction <= 1.0 or max_units < 0:
        raise ValueError("invalid Kelly sizing configuration")
    decimal = american_to_decimal(american_odds)
    b = decimal - 1.0
    if b <= 0:
        return 0.0
    full_kelly = (b * probability - (1.0 - probability)) / b
    return round(min(max_units, max(0.0, full_kelly * kelly_fraction)), 6)
