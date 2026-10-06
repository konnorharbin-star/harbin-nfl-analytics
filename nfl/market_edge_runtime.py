"""Verified downstream NFL market-edge calibration.

The independent football model is never changed by sportsbook prices. This module
operates only after a raw model probability and a two-way no-vig market probability
exist.

For the current 2026 decision layer, alpha is selected on verified 2024 opening
quotes and evaluated once on verified 2025 opening quotes. The operational decision
probability is:

    logit(p_decision) = logit(p_market)
                        + alpha * (logit(p_model) - logit(p_market))

Alpha=0 means the market is preferred and the model contributes no betting edge.
Alpha=1 means the raw model probability is retained. Missing or failed evidence
falls back to alpha=0 for executable decisions while raw model values remain
available as research diagnostics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite

import polars as pl

from .contracts import DataContractError, require_columns
from .market_shrinkage import (
    DEFAULT_ALPHA_GRID,
    fit_alpha,
    probability_comparison,
    shrink_probability,
)

MARKETS = ("moneyline", "spread", "total")
VERIFIED_REQUIRED = {
    "season",
    "market_type",
    "model_probability",
    "no_vig_probability",
    "result",
}
DEFAULT_TUNE_SEASON = 2024
DEFAULT_HOLDOUT_SEASON = 2025
MIN_TUNE_ROWS = 100
MIN_HOLDOUT_ROWS = 100


@dataclass(frozen=True)
class MarketEdgeDecision:
    market_type: str
    status: str
    ready: bool
    selected_alpha: float | None
    operational_alpha: float
    raw_model_probability: float
    no_vig_probability: float
    decision_probability: float
    raw_probability_edge: float
    decision_probability_edge: float
    raw_expected_value_per_unit: float
    decision_expected_value_per_unit: float
    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _verified_sample(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    require_columns(frame, VERIFIED_REQUIRED, "verified_market_edge_history")
    sample = frame
    if "entry_price_verified" in sample.columns:
        sample = sample.filter(
            pl.col("entry_price_verified")
            .cast(pl.Boolean, strict=False)
            .fill_null(False)
        )
    if "entry_quote_verified" in sample.columns:
        sample = sample.filter(
            pl.col("entry_quote_verified")
            .cast(pl.Boolean, strict=False)
            .fill_null(False)
        )
    return sample


def _metric_values(value: object) -> tuple[float | None, float | None]:
    if not isinstance(value, dict):
        return None, None
    try:
        brier = float(value["brier"])
        log_loss = float(value["log_loss"])
    except (KeyError, TypeError, ValueError):
        return None, None
    if not isfinite(brier) or not isfinite(log_loss):
        return None, None
    return brier, log_loss


def _not_worse(
    candidate: object,
    baseline: object,
    *,
    tolerance: float = 1e-12,
) -> bool:
    candidate_brier, candidate_log = _metric_values(candidate)
    baseline_brier, baseline_log = _metric_values(baseline)
    return (
        candidate_brier is not None
        and candidate_log is not None
        and baseline_brier is not None
        and baseline_log is not None
        and candidate_brier <= baseline_brier + tolerance
        and candidate_log <= baseline_log + tolerance
    )


def _strictly_better(candidate: object, baseline: object) -> bool:
    if not _not_worse(candidate, baseline):
        return False
    candidate_brier, candidate_log = _metric_values(candidate)
    baseline_brier, baseline_log = _metric_values(baseline)
    assert candidate_brier is not None
    assert candidate_log is not None
    assert baseline_brier is not None
    assert baseline_log is not None
    return (
        candidate_brier < baseline_brier - 1e-12
        or candidate_log < baseline_log - 1e-12
    )


def _empty_registry(reason: str) -> dict[str, object]:
    return {
        "version": 1,
        "status": "INSUFFICIENT_DATA",
        "fail_closed": True,
        "operational_ready": False,
        "reason": reason,
        "tune_season": None,
        "holdout_season": None,
        "verified_rows": 0,
        "validated_incremental_markets": [],
        "market_only_markets": [],
        "markets": {
            market: {
                "status": "INSUFFICIENT_DATA",
                "selected_alpha": None,
                "operational_alpha": 0.0,
                "incremental_model_value": False,
                "decision_enabled": False,
            }
            for market in MARKETS
        },
    }


def build_verified_market_edge_registry(
    bets: pl.DataFrame,
    *,
    tune_season: int = DEFAULT_TUNE_SEASON,
    holdout_season: int = DEFAULT_HOLDOUT_SEASON,
    alpha_grid: tuple[float, ...] = DEFAULT_ALPHA_GRID,
    min_tune_rows: int = MIN_TUNE_ROWS,
    min_holdout_rows: int = MIN_HOLDOUT_ROWS,
) -> dict[str, object]:
    """Select a market anchor on one verified season and test the next season."""

    if tune_season >= holdout_season:
        raise ValueError("tune_season must be earlier than holdout_season")
    if min_tune_rows < 1 or min_holdout_rows < 1:
        raise ValueError("minimum row requirements must be positive")
    if bets.is_empty():
        return _empty_registry("no historical market rows")

    try:
        sample = _verified_sample(bets)
    except DataContractError as exc:
        return _empty_registry(str(exc))
    if sample.is_empty():
        return _empty_registry("no verified two-way entry quotes")

    markets: dict[str, object] = {}
    validated: list[str] = []
    market_only: list[str] = []

    for market in MARKETS:
        market_rows = sample.filter(
            pl.col("market_type").cast(pl.String).str.to_lowercase() == market
        )
        tune = market_rows.filter(
            pl.col("season").cast(pl.Int64, strict=False) == tune_season
        )
        holdout = market_rows.filter(
            pl.col("season").cast(pl.Int64, strict=False) == holdout_season
        )

        if tune.height < min_tune_rows or holdout.height < min_holdout_rows:
            markets[market] = {
                "status": "INSUFFICIENT_DATA",
                "selected_alpha": None,
                "operational_alpha": 0.0,
                "incremental_model_value": False,
                "decision_enabled": False,
                "tune_rows": tune.height,
                "holdout_rows": holdout.height,
                "reason": (
                    f"requires >= {min_tune_rows} tune and >= "
                    f"{min_holdout_rows} holdout rows"
                ),
            }
            continue

        alpha, tune_grid = fit_alpha(
            tune,
            alpha_grid=alpha_grid,
            minimum_rows=min_tune_rows,
        )
        if alpha is None:
            markets[market] = {
                "status": "INSUFFICIENT_DATA",
                "selected_alpha": None,
                "operational_alpha": 0.0,
                "incremental_model_value": False,
                "decision_enabled": False,
                "tune_rows": tune.height,
                "holdout_rows": holdout.height,
                "tune_grid": tune_grid,
                "reason": "no alpha satisfied tune sample requirements",
            }
            continue

        holdout_probability = probability_comparison(
            holdout,
            alpha=float(alpha),
        )
        anchored = holdout_probability["shrunk_model"]
        raw = holdout_probability["raw_model"]
        market_probability = holdout_probability["market_only"]

        improves_raw = _strictly_better(anchored, raw)
        not_worse_market = _not_worse(anchored, market_probability)
        beats_market = _strictly_better(anchored, market_probability)
        incremental = (
            float(alpha) > 0.0
            and improves_raw
            and not_worse_market
            and beats_market
        )

        market_beats_raw = _strictly_better(market_probability, raw)
        if incremental:
            status = "VALIDATED_INCREMENTAL"
            operational_alpha = float(alpha)
            decision_enabled = True
            validated.append(market)
            reason = (
                "verified holdout supports model contribution beyond the no-vig market"
            )
        elif float(alpha) == 0.0 and market_beats_raw:
            status = "MARKET_ONLY_PREFERRED"
            operational_alpha = 0.0
            decision_enabled = False
            market_only.append(market)
            reason = (
                "verified tune selected alpha=0 and holdout market probability "
                "outperformed the raw model"
            )
        else:
            status = "NO_VALIDATED_INCREMENTAL_VALUE"
            operational_alpha = 0.0
            decision_enabled = False
            reason = (
                "selected blend did not prove incremental value versus both the "
                "raw model and the no-vig market on untouched holdout"
            )

        markets[market] = {
            "status": status,
            "selected_alpha": float(alpha),
            "operational_alpha": operational_alpha,
            "incremental_model_value": incremental,
            "decision_enabled": decision_enabled,
            "tune_rows": tune.height,
            "holdout_rows": holdout.height,
            "tune_grid": tune_grid,
            "holdout_probability": holdout_probability,
            "improves_raw_model": improves_raw,
            "not_worse_than_market": not_worse_market,
            "beats_market": beats_market,
            "reason": reason,
        }

    operational_ready = len(validated) >= 2
    return {
        "version": 1,
        "status": "READY",
        "fail_closed": True,
        "operational_ready": operational_ready,
        "tune_season": tune_season,
        "holdout_season": holdout_season,
        "verified_rows": sample.height,
        "alpha_grid": list(alpha_grid),
        "selection_objective": "tune-season log loss; Brier tie-breaker",
        "validation_rule": (
            "operational alpha must improve raw model and strictly beat the "
            "no-vig market on untouched holdout"
        ),
        "validated_incremental_markets": validated,
        "market_only_markets": market_only,
        "markets": markets,
        "meaning": (
            "Sportsbook prices remain downstream. The registry controls only how much "
            "of a model-vs-market probability difference is trusted for execution."
        ),
    }


def assess_market_edge_candidate(
    registry: dict[str, object] | None,
    *,
    market_type: str,
    model_probability: float,
    no_vig_probability: float,
    decimal_odds: float,
) -> MarketEdgeDecision:
    """Return the execution probability after verified market-edge calibration."""

    market = str(market_type).strip().lower()
    raw = float(model_probability)
    market_probability = float(no_vig_probability)
    odds = float(decimal_odds)
    if not (
        market in MARKETS
        and isfinite(raw)
        and isfinite(market_probability)
        and isfinite(odds)
        and 0.0 < raw < 1.0
        and 0.0 < market_probability < 1.0
        and odds > 1.0
    ):
        raise ValueError("invalid market-edge candidate inputs")

    raw_edge = raw - market_probability
    raw_ev = raw * odds - 1.0

    if not isinstance(registry, dict):
        decision_probability = market_probability
        return MarketEdgeDecision(
            market_type=market,
            status="BLOCKED",
            ready=False,
            selected_alpha=None,
            operational_alpha=0.0,
            raw_model_probability=raw,
            no_vig_probability=market_probability,
            decision_probability=decision_probability,
            raw_probability_edge=raw_edge,
            decision_probability_edge=0.0,
            raw_expected_value_per_unit=raw_ev,
            decision_expected_value_per_unit=decision_probability * odds - 1.0,
            reason="market-edge registry is not configured",
        )

    if not bool(registry.get("fail_closed", True)):
        return MarketEdgeDecision(
            market_type=market,
            status="NOT_ENFORCED",
            ready=True,
            selected_alpha=None,
            operational_alpha=1.0,
            raw_model_probability=raw,
            no_vig_probability=market_probability,
            decision_probability=raw,
            raw_probability_edge=raw_edge,
            decision_probability_edge=raw_edge,
            raw_expected_value_per_unit=raw_ev,
            decision_expected_value_per_unit=raw_ev,
            reason="market-edge enforcement is disabled",
        )

    market_map = registry.get("markets")
    entry = market_map.get(market) if isinstance(market_map, dict) else None
    if not isinstance(entry, dict):
        status = "INSUFFICIENT_DATA"
        selected_alpha = None
        operational_alpha = 0.0
        ready = False
        reason = "market calibration is missing"
    else:
        status = str(entry.get("status") or "INSUFFICIENT_DATA").upper()
        selected = entry.get("selected_alpha")
        selected_alpha = None if selected is None else float(selected)
        ready = bool(entry.get("decision_enabled", False)) and (
            status == "VALIDATED_INCREMENTAL"
        )
        operational_alpha = (
            float(entry.get("operational_alpha", 0.0) or 0.0)
            if ready
            else 0.0
        )
        reason = str(entry.get("reason") or status.lower().replace("_", " "))

    decision_probability = shrink_probability(
        raw,
        market_probability,
        operational_alpha,
    )
    decision_edge = decision_probability - market_probability
    decision_ev = decision_probability * odds - 1.0
    return MarketEdgeDecision(
        market_type=market,
        status=status,
        ready=ready,
        selected_alpha=selected_alpha,
        operational_alpha=operational_alpha,
        raw_model_probability=raw,
        no_vig_probability=market_probability,
        decision_probability=decision_probability,
        raw_probability_edge=raw_edge,
        decision_probability_edge=decision_edge,
        raw_expected_value_per_unit=raw_ev,
        decision_expected_value_per_unit=decision_ev,
        reason=reason,
    )



def apply_market_edge_registry_to_frame(
    frame: pl.DataFrame,
    registry: dict[str, object] | None,
) -> pl.DataFrame:
    """Attach executable market-anchored probability fields to historical rows."""

    required = {
        "market_type",
        "model_probability",
        "no_vig_probability",
        "decimal_odds",
    }
    require_columns(frame, required, "market_edge_application")
    if frame.is_empty():
        return frame

    rows: list[dict[str, object]] = []
    for source in frame.iter_rows(named=True):
        try:
            decision = assess_market_edge_candidate(
                registry,
                market_type=str(source["market_type"]),
                model_probability=float(source["model_probability"]),
                no_vig_probability=float(source["no_vig_probability"]),
                decimal_odds=float(source["decimal_odds"]),
            )
        except (TypeError, ValueError):
            item = dict(source)
            item.update(
                {
                    "raw_model_probability": source.get("model_probability"),
                    "raw_probability_edge": source.get("probability_edge"),
                    "raw_expected_value_per_unit": source.get(
                        "expected_value_per_unit"
                    ),
                    "decision_probability": None,
                    "decision_probability_edge": None,
                    "decision_expected_value_per_unit": None,
                    "market_edge_status": "INVALID",
                    "market_edge_ready": False,
                    "market_edge_alpha": 0.0,
                    "market_edge_reason": "invalid historical market row",
                }
            )
            rows.append(item)
            continue

        item = dict(source)
        item.update(
            {
                "raw_model_probability": decision.raw_model_probability,
                "raw_probability_edge": decision.raw_probability_edge,
                "raw_expected_value_per_unit": (
                    decision.raw_expected_value_per_unit
                ),
                "decision_probability": decision.decision_probability,
                "decision_probability_edge": (
                    decision.decision_probability_edge
                ),
                "decision_expected_value_per_unit": (
                    decision.decision_expected_value_per_unit
                ),
                "market_edge_status": decision.status,
                "market_edge_ready": decision.ready,
                "market_edge_alpha": decision.operational_alpha,
                "market_edge_reason": decision.reason,
            }
        )
        rows.append(item)
    return pl.DataFrame(rows)
