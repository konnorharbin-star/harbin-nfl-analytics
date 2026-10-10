"""Nested chronological calibration for NFL market-decision policy.

This module mirrors the NCAA production-policy evidence contract while keeping NFL
thresholds, units, samples, and release evidence independent. Sportsbook information
is used only in the downstream decision layer; it never enters the fair-score model.

Only rows with an independently verified entry price may calibrate a production
policy. Unverified archive-final fallbacks remain useful research diagnostics but are
excluded from threshold selection and release decisions.
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path

import polars as pl

from .policy import DEFAULT_POLICY
from .regime_reliability import (
    build_regime_reliability_report,
    compact_registry,
)

CALIBRATION_REQUIRED = {
    "season",
    "week",
    "market_type",
    "model_probability",
    "probability_edge",
    "expected_value_per_unit",
    "net_units",
    "entry_price_verified",
}


def _empty_stats() -> dict[str, object]:
    return {
        "n": 0,
        "roi": None,
        "lcb": None,
        "avg_clv": None,
    }


def _profit_stats(frame: pl.DataFrame) -> dict[str, object]:
    if frame.is_empty() or "net_units" not in frame.columns:
        return _empty_stats()
    profits = frame.get_column("net_units").cast(pl.Float64, strict=False).drop_nulls()
    if profits.len() == 0:
        return _empty_stats()
    n = profits.len()
    roi = float(profits.mean())
    if n > 1:
        std = float(profits.std(ddof=1) or 0.0)
        lcb = roi - 1.28 * std / math.sqrt(n)
    else:
        lcb = roi - 1.28

    avg_clv: float | None = None
    if "clv_proxy" in frame.columns:
        clv = frame.get_column("clv_proxy").cast(pl.Float64, strict=False).drop_nulls()
        if clv.len():
            avg_clv = float(clv.mean())

    return {
        "n": int(n),
        "roi": roi,
        "lcb": float(lcb),
        "avg_clv": avg_clv,
    }


def _promotion_sample(frame: pl.DataFrame) -> pl.DataFrame:
    from .entry_integrity import timestamped_promotion_sample

    return timestamped_promotion_sample(frame)


def _ordered(frame: pl.DataFrame) -> pl.DataFrame:
    columns = [name for name in ("season", "week", "game_id") if name in frame.columns]
    return frame.sort(columns) if columns else frame


def _filter_keys(
    frame: pl.DataFrame,
    keys: list[tuple[int, int]],
) -> pl.DataFrame:
    if not keys:
        return frame.head(0)
    expr = pl.lit(False)
    for season, week in keys:
        expr = expr | (
            (pl.col("season").cast(pl.Int64, strict=False) == season)
            & (pl.col("week").cast(pl.Int64, strict=False) == week)
        )
    return frame.filter(expr)


def _nested_split(
    frame: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, str]:
    ordered = _ordered(frame)
    if ordered.is_empty():
        empty = ordered.head(0)
        return empty, empty, empty, "empty verified-entry sample"

    seasons = (
        ordered.get_column("season")
        .cast(pl.Int64, strict=False)
        .drop_nulls()
        .unique()
        .sort()
        .to_list()
    )
    if len(seasons) >= 3:
        tune_season = int(seasons[-2])
        evaluation_season = int(seasons[-1])
        season_col = pl.col("season").cast(pl.Int64, strict=False)
        development = ordered.filter(season_col < tune_season)
        tune = ordered.filter(season_col == tune_season)
        evaluation = ordered.filter(season_col == evaluation_season)
        return (
            development,
            tune,
            evaluation,
            (
                f"season<{tune_season} / tune={tune_season} / "
                f"untouched={evaluation_season}"
            ),
        )

    if {"season", "week"}.issubset(ordered.columns):
        keys = [
            (int(row["season"]), int(row["week"]))
            for row in (
                ordered.select("season", "week")
                .drop_nulls()
                .unique()
                .sort(["season", "week"])
                .iter_rows(named=True)
            )
        ]
        if len(keys) >= 8:
            first = max(1, int(round(0.50 * len(keys))))
            second = max(first + 1, int(round(0.75 * len(keys))))
            second = min(second, len(keys) - 1)
            development = _filter_keys(ordered, keys[:first])
            tune = _filter_keys(ordered, keys[first:second])
            evaluation = _filter_keys(ordered, keys[second:])
            return development, tune, evaluation, "whole-week 50/25/25 nested chronology"

    n = ordered.height
    if n < 3:
        empty = ordered.head(0)
        return ordered, empty, empty, "insufficient chronological sample"

    first = max(1, int(0.50 * n))
    second = min(n - 1, max(first + 1, int(0.75 * n)))
    return (
        ordered.slice(0, first),
        ordered.slice(first, second - first),
        ordered.slice(second),
        "chronological 50/25/25 fallback",
    )


def _filtered(
    frame: pl.DataFrame,
    *,
    min_ev: float,
    min_edge: float,
    min_probability: float,
) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    return frame.filter(
        (pl.col("expected_value_per_unit").cast(pl.Float64, strict=False) >= min_ev)
        & (
            pl.col("probability_edge")
            .cast(pl.Float64, strict=False)
            .abs()
            >= min_edge
        )
        & (pl.col("model_probability").cast(pl.Float64, strict=False) >= min_probability)
    )


def _candidate_grid(
    market: str,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    expected_values = (0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10)
    probabilities = (0.52, 0.54, 0.56, 0.58, 0.60)
    if market == "moneyline":
        edges = (0.015, 0.020, 0.025, 0.030, 0.040, 0.050, 0.060)
    elif market == "spread":
        edges = (0.020, 0.025, 0.030, 0.035, 0.040, 0.050, 0.060)
    else:
        edges = (0.025, 0.030, 0.035, 0.040, 0.050, 0.060, 0.070)
    return expected_values, edges, probabilities


def _weak_weeks(
    development: pl.DataFrame,
    tune: pl.DataFrame,
    *,
    min_ev: float,
    min_edge: float,
    min_probability: float,
) -> list[int]:
    if "week" not in development.columns or "week" not in tune.columns:
        return []

    development = _filtered(
        development,
        min_ev=min_ev,
        min_edge=min_edge,
        min_probability=min_probability,
    )
    tune = _filtered(
        tune,
        min_ev=min_ev,
        min_edge=min_edge,
        min_probability=min_probability,
    )
    if development.is_empty() or tune.is_empty():
        return []

    dev_weeks = set(
        int(value)
        for value in development.get_column("week")
        .cast(pl.Int64, strict=False)
        .drop_nulls()
        .to_list()
    )
    tune_weeks = set(
        int(value)
        for value in tune.get_column("week")
        .cast(pl.Int64, strict=False)
        .drop_nulls()
        .to_list()
    )

    excluded: list[int] = []
    for week in sorted(dev_weeks & tune_weeks):
        dev_stats = _profit_stats(
            development.filter(pl.col("week").cast(pl.Int64, strict=False) == week)
        )
        tune_stats = _profit_stats(
            tune.filter(pl.col("week").cast(pl.Int64, strict=False) == week)
        )
        if (
            int(dev_stats["n"]) >= 40
            and int(tune_stats["n"]) >= 15
            and dev_stats["roi"] is not None
            and tune_stats["roi"] is not None
            and float(dev_stats["roi"]) < 0
            and float(tune_stats["roi"]) < 0
        ):
            excluded.append(week)
    return excluded


def _tiered_thresholds(
    market: str,
    *,
    min_ev: float,
    min_edge: float,
    min_probability: float,
) -> dict[str, dict[str, float]]:
    default_market = DEFAULT_POLICY["markets"][market]
    assert isinstance(default_market, dict)

    lean_default = default_market["lean"]
    bet_default = default_market["bet"]
    strong_default = default_market["strong"]
    assert isinstance(lean_default, dict)
    assert isinstance(bet_default, dict)
    assert isinstance(strong_default, dict)

    lean = {
        "min_ev": max(float(lean_default["min_ev"]), min_ev),
        "min_edge": max(float(lean_default["min_edge"]), min_edge),
        "min_prob": max(float(lean_default["min_prob"]), min_probability),
    }
    bet = {
        "min_ev": max(float(bet_default["min_ev"]), lean["min_ev"] + 0.015),
        "min_edge": max(float(bet_default["min_edge"]), lean["min_edge"] + 0.010),
        "min_prob": max(float(bet_default["min_prob"]), lean["min_prob"] + 0.015),
    }
    strong = {
        "min_ev": max(float(strong_default["min_ev"]), lean["min_ev"] + 0.040),
        "min_edge": max(float(strong_default["min_edge"]), lean["min_edge"] + 0.025),
        "min_prob": max(float(strong_default["min_prob"]), lean["min_prob"] + 0.030),
    }
    return {"lean": lean, "bet": bet, "strong": strong}


def derive_policy_from_frame(
    raw_bets: pl.DataFrame,
    *,
    evidence: dict[str, object] | None = None,
) -> dict[str, object]:
    """Derive a frozen NFL policy without allowing evaluation data to select thresholds."""

    policy = deepcopy(DEFAULT_POLICY)
    regime_report = build_regime_reliability_report(raw_bets)
    regime_registry = compact_registry(regime_report)
    policy["regime_reliability"] = regime_registry

    markets = policy.get("markets")
    regime_registry_ready = (
        str(regime_registry.get("status") or "").upper() == "READY"
    )
    if isinstance(markets, dict) and regime_registry_ready:
        market_status = regime_registry.get("market_status")
        if not isinstance(market_status, dict):
            market_status = {}
        for market_name, config in markets.items():
            if not isinstance(config, dict):
                continue
            status = str(
                market_status.get(str(market_name), "INSUFFICIENT")
            ).upper()
            if status != "RELIABLE":
                config["enabled"] = False
                config["disabled_reason"] = (
                    "market-level probability/edge regime reliability "
                    f"is {status.lower()}"
                )

    promotion = _promotion_sample(raw_bets)
    diagnostics: dict[str, object] = {
        "selection_uses_evaluation": False,
        "raw_archive_rows": int(raw_bets.height),
        "promotion_rows": int(promotion.height),
        "regime_reliability": regime_report.get("summary", {}),
    }

    if promotion.is_empty():
        policy["deployment_mode"] = "paper"
        policy["source"] = "paper only; no verified archived opening-entry sample"
        policy["split"] = "unavailable"
        policy["diagnostics"] = diagnostics
        return policy

    missing = CALIBRATION_REQUIRED - set(promotion.columns)
    if missing:
        policy["deployment_mode"] = "paper"
        policy["source"] = (
            "paper only; verified sample missing calibration columns: "
            + ", ".join(sorted(missing))
        )
        policy["split"] = "unavailable"
        diagnostics["missing_columns"] = sorted(missing)
        policy["diagnostics"] = diagnostics
        return policy

    development, tune, evaluation, split_description = _nested_split(promotion)
    passed_markets = 0

    for market in ("moneyline", "spread", "total"):
        dev = development.filter(pl.col("market_type").cast(pl.String).str.to_lowercase() == market)
        val = tune.filter(pl.col("market_type").cast(pl.String).str.to_lowercase() == market)
        evl = evaluation.filter(
            pl.col("market_type").cast(pl.String).str.to_lowercase() == market
        )

        candidates: list[
            tuple[float, float, int, float, float, float]
        ] = []
        expected_values, edges, probabilities = _candidate_grid(market)
        for min_ev in expected_values:
            for min_edge in edges:
                for min_probability in probabilities:
                    stats = _profit_stats(
                        _filtered(
                            dev,
                            min_ev=min_ev,
                            min_edge=min_edge,
                            min_probability=min_probability,
                        )
                    )
                    if int(stats["n"]) < 40 or stats["roi"] is None:
                        continue
                    candidates.append(
                        (
                            float(stats["lcb"]),
                            float(stats["roi"]),
                            int(stats["n"]),
                            min_ev,
                            min_edge,
                            min_probability,
                        )
                    )
        candidates.sort(reverse=True)

        chosen: tuple[float, float, float, dict[str, object]] | None = None
        for _, _, _, min_ev, min_edge, min_probability in candidates[:25]:
            stats = _profit_stats(
                _filtered(
                    val,
                    min_ev=min_ev,
                    min_edge=min_edge,
                    min_probability=min_probability,
                )
            )
            if (
                int(stats["n"]) >= 20
                and stats["roi"] is not None
                and float(stats["roi"]) >= 0
                and stats["avg_clv"] is not None
                and float(stats["avg_clv"]) >= 0
            ):
                chosen = (min_ev, min_edge, min_probability, stats)
                break

        markets = policy["markets"]
        assert isinstance(markets, dict)
        config = markets[market]
        assert isinstance(config, dict)
        market_diag: dict[str, object] = {
            "development_bets": int(dev.height),
            "tune_bets": int(val.height),
            "evaluation_bets": int(evl.height),
            "selected": chosen,
        }

        if chosen is None:
            config["enabled"] = False
            config["disabled_reason"] = (
                "no threshold passed pre-evaluation development/tuning validation"
            )
            config["excluded_weeks"] = []
            market_diag["evaluation"] = None
            market_diag["evaluation_passed"] = False
            diagnostics[market] = market_diag
            continue

        min_ev, min_edge, min_probability, _ = chosen
        thresholds = _tiered_thresholds(
            market,
            min_ev=min_ev,
            min_edge=min_edge,
            min_probability=min_probability,
        )
        config.update(thresholds)
        excluded_weeks = _weak_weeks(
            dev,
            val,
            min_ev=thresholds["lean"]["min_ev"],
            min_edge=thresholds["lean"]["min_edge"],
            min_probability=thresholds["lean"]["min_prob"],
        )
        config["excluded_weeks"] = excluded_weeks

        evaluation_sample = _filtered(
            evl,
            min_ev=thresholds["lean"]["min_ev"],
            min_edge=thresholds["lean"]["min_edge"],
            min_probability=thresholds["lean"]["min_prob"],
        )
        if excluded_weeks and "week" in evaluation_sample.columns:
            evaluation_sample = evaluation_sample.filter(
                ~pl.col("week")
                .cast(pl.Int64, strict=False)
                .is_in(excluded_weeks)
            )
        evaluation_stats = _profit_stats(evaluation_sample)
        passed = (
            int(evaluation_stats["n"]) >= 20
            and evaluation_stats["roi"] is not None
            and float(evaluation_stats["roi"]) >= 0
            and evaluation_stats["avg_clv"] is not None
            and float(evaluation_stats["avg_clv"]) >= 0
        )
        config["enabled"] = bool(passed)
        if passed:
            config.pop("disabled_reason", None)
            passed_markets += 1
        else:
            config["disabled_reason"] = (
                "frozen threshold failed untouched chronological evaluation"
            )

        market_diag.update(
            {
                "excluded_weeks": excluded_weeks,
                "evaluation": evaluation_stats,
                "evaluation_passed": bool(passed),
            }
        )
        diagnostics[market] = market_diag

    evidence = evidence or {}
    promotion_evidence = evidence.get("promotion_sample")
    if not isinstance(promotion_evidence, dict):
        promotion_evidence = {}
    robust = (
        str(evidence.get("status") or "").upper() == "ROBUST"
        and bool(promotion_evidence.get("entry_quote_verified", False))
    )
    policy["deployment_mode"] = (
        "production" if robust and passed_markets >= 2 else "paper"
    )
    policy["source"] = (
        "nested chronological NFL policy calibration on verified archived opening "
        "entries; untouched evaluation is release-only"
    )
    policy["split"] = split_description
    diagnostics["passed_markets"] = passed_markets
    diagnostics["robust_evidence"] = robust
    policy["diagnostics"] = diagnostics
    return policy


def derive_production_policy(
    *,
    bets_path: str | Path = "reports/free_market_bets.csv",
    verified_bets_path: str | Path = "reports/verified_market_bets.csv",
    evidence_path: str | Path = "reports/evidence_report.json",
    output_path: str | Path = "reports/production_policy.json",
) -> dict[str, object]:
    """Read canonical evidence, derive policy, and persist the frozen policy snapshot."""

    frames: list[pl.DataFrame] = []
    bets_file = Path(bets_path)
    if bets_file.exists():
        frames.append(pl.read_csv(bets_file, try_parse_dates=True))

    verified_file = Path(verified_bets_path)
    if verified_file.exists():
        frames.append(pl.read_csv(verified_file, try_parse_dates=True))

    raw_bets = (
        pl.concat(frames, how="diagonal_relaxed")
        if frames
        else pl.DataFrame()
    )
    if not raw_bets.is_empty():
        dedupe = [
            name
            for name in ("season", "week", "game_id", "market_type", "entry_price_verified")
            if name in raw_bets.columns
        ]
        if len(dedupe) >= 4:
            raw_bets = raw_bets.unique(subset=dedupe, keep="last")

    evidence_file = Path(evidence_path)
    try:
        evidence = json.loads(evidence_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        evidence = {}

    policy = derive_policy_from_frame(raw_bets, evidence=evidence)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(policy, indent=2, sort_keys=True), encoding="utf-8")
    return policy
