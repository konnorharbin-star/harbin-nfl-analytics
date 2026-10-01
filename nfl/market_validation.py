"""Chronological validation for sportsbook qualification rules.

Thresholds are selected on one validation season only, then frozen before a later
holdout season is scored. Each game/market contributes at most one executable quote:
the best model EV available across books at the simulated decision time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import polars as pl

from .contracts import DataContractError, require_columns

MarketType = Literal["moneyline", "spread", "total"]

RULE_REQUIRED = {
    "game_id",
    "season",
    "market_type",
    "side",
    "book",
    "probability_edge",
    "expected_value_per_unit",
    "result",
    "net_units",
    "probability_clv",
}


@dataclass(frozen=True)
class MarketRule:
    min_probability_edge: float
    min_expected_value_per_unit: float


@dataclass(frozen=True)
class MarketRuleMetrics:
    bets: int
    wins: int
    losses: int
    pushes: int
    net_units: float
    roi_per_unit_staked: float
    average_probability_edge: float
    average_expected_value_per_unit: float
    average_probability_clv: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MarketRuleHoldoutEvaluation:
    market_type: MarketType
    validation_season: int
    holdout_season: int
    rule: MarketRule | None
    validation: MarketRuleMetrics
    holdout: MarketRuleMetrics
    candidate_pass: bool

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["rule"] = None if self.rule is None else asdict(self.rule)
        result["validation"] = asdict(self.validation)
        result["holdout"] = asdict(self.holdout)
        return result


def select_best_market_opportunities(frame: pl.DataFrame) -> pl.DataFrame:
    """Keep one line-shopped executable quote per game and market.

    This prevents a backtest from counting the same modeled game edge once per book.
    The chosen row is the highest expected-value quote available at decision time;
    deterministic tie breakers preserve reproducibility.
    """

    require_columns(
        frame,
        {
            "game_id",
            "market_type",
            "side",
            "book",
            "expected_value_per_unit",
            "probability_edge",
            "american_odds",
        },
        "market_opportunities",
    )
    if frame.is_empty():
        return frame
    return (
        frame.sort(
            [
                "game_id",
                "market_type",
                "expected_value_per_unit",
                "probability_edge",
                "american_odds",
                "book",
                "side",
            ],
            descending=[False, False, True, True, True, False, False],
        )
        .unique(subset=["game_id", "market_type"], keep="first", maintain_order=True)
        .sort(["game_id", "market_type"])
    )


def _empty_metrics() -> MarketRuleMetrics:
    return MarketRuleMetrics(
        bets=0,
        wins=0,
        losses=0,
        pushes=0,
        net_units=0.0,
        roi_per_unit_staked=0.0,
        average_probability_edge=0.0,
        average_expected_value_per_unit=0.0,
        average_probability_clv=0.0,
    )


def _apply_rule(frame: pl.DataFrame, rule: MarketRule) -> pl.DataFrame:
    return frame.filter(
        (pl.col("probability_edge") >= rule.min_probability_edge)
        & (pl.col("expected_value_per_unit") >= rule.min_expected_value_per_unit)
    )


def _metrics(frame: pl.DataFrame) -> MarketRuleMetrics:
    if frame.is_empty():
        return _empty_metrics()
    bets = frame.height
    net_units = float(frame.get_column("net_units").sum())
    return MarketRuleMetrics(
        bets=bets,
        wins=frame.filter(pl.col("result") == "win").height,
        losses=frame.filter(pl.col("result") == "loss").height,
        pushes=frame.filter(pl.col("result") == "push").height,
        net_units=net_units,
        roi_per_unit_staked=net_units / bets,
        average_probability_edge=float(frame.get_column("probability_edge").mean()),
        average_expected_value_per_unit=float(
            frame.get_column("expected_value_per_unit").mean()
        ),
        average_probability_clv=float(frame.get_column("probability_clv").mean()),
    )


def _select_rule(
    validation: pl.DataFrame,
    *,
    probability_edge_grid: tuple[float, ...],
    expected_value_grid: tuple[float, ...],
    min_validation_bets: int,
) -> tuple[MarketRule | None, MarketRuleMetrics]:
    if min_validation_bets < 1:
        raise ValueError("min_validation_bets must be >= 1")
    if not probability_edge_grid or any(value < 0 for value in probability_edge_grid):
        raise ValueError("probability_edge_grid must contain non-negative values")
    if not expected_value_grid or any(value < 0 for value in expected_value_grid):
        raise ValueError("expected_value_grid must contain non-negative values")

    candidates: list[tuple[tuple[float, ...], MarketRule, MarketRuleMetrics]] = []
    for edge in probability_edge_grid:
        for expected_value in expected_value_grid:
            rule = MarketRule(float(edge), float(expected_value))
            metrics = _metrics(_apply_rule(validation, rule))
            if metrics.bets < min_validation_bets:
                continue
            # Rule selection happens on validation only. Favor total net units first so
            # tiny-sample high-ROI rules do not dominate, then ROI and observed CLV.
            score = (
                metrics.net_units,
                metrics.roi_per_unit_staked,
                metrics.average_probability_clv,
                -rule.min_probability_edge,
                -rule.min_expected_value_per_unit,
            )
            candidates.append((score, rule, metrics))

    if not candidates:
        return None, _empty_metrics()
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, rule, metrics = candidates[0]
    if metrics.net_units <= 0 or metrics.average_probability_clv <= 0:
        return None, metrics
    return rule, metrics


def evaluate_market_rule_holdout(
    graded_with_clv: pl.DataFrame,
    *,
    market_type: MarketType,
    validation_season: int,
    holdout_season: int,
    probability_edge_grid: tuple[float, ...] = (0.02, 0.03, 0.04, 0.05, 0.06),
    expected_value_grid: tuple[float, ...] = (0.01, 0.02, 0.03, 0.04, 0.05),
    min_validation_bets: int = 25,
    min_holdout_bets: int = 25,
) -> MarketRuleHoldoutEvaluation:
    """Tune one market's betting rule on validation, then score a later holdout once."""

    require_columns(graded_with_clv, RULE_REQUIRED, "market_rule_dataset")
    if market_type not in {"moneyline", "spread", "total"}:
        raise ValueError(f"unsupported market_type: {market_type}")
    if validation_season >= holdout_season:
        raise ValueError("validation_season must be earlier than holdout_season")
    if min_holdout_bets < 1:
        raise ValueError("min_holdout_bets must be >= 1")

    opportunities = select_best_market_opportunities(graded_with_clv).filter(
        pl.col("market_type") == market_type
    )
    validation = opportunities.filter(pl.col("season") == validation_season)
    holdout = opportunities.filter(pl.col("season") == holdout_season)
    if validation.is_empty() or holdout.is_empty():
        raise DataContractError("validation and holdout seasons must both contain market rows")

    rule, validation_metrics = _select_rule(
        validation,
        probability_edge_grid=probability_edge_grid,
        expected_value_grid=expected_value_grid,
        min_validation_bets=min_validation_bets,
    )
    if rule is None:
        return MarketRuleHoldoutEvaluation(
            market_type=market_type,
            validation_season=validation_season,
            holdout_season=holdout_season,
            rule=None,
            validation=validation_metrics,
            holdout=_empty_metrics(),
            candidate_pass=False,
        )

    holdout_metrics = _metrics(_apply_rule(holdout, rule))
    candidate_pass = (
        holdout_metrics.bets >= min_holdout_bets
        and holdout_metrics.net_units > 0
        and holdout_metrics.roi_per_unit_staked > 0
        and holdout_metrics.average_probability_clv > 0
    )
    return MarketRuleHoldoutEvaluation(
        market_type=market_type,
        validation_season=validation_season,
        holdout_season=holdout_season,
        rule=rule,
        validation=validation_metrics,
        holdout=holdout_metrics,
        candidate_pass=candidate_pass,
    )
