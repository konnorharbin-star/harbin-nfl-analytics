"""Rigorous NFL archive diagnostics aligned with the NCAA backtest runtime.

The base archive backtest remains the evidence generator. This module adds week-block
uncertainty, probability-equivalent CLV, and stable market/role/location/time segments
without changing which historical bets were selected.
"""

from __future__ import annotations

import json
from math import erf, isfinite, sqrt
from pathlib import Path

import numpy as np
import polars as pl

from .contracts import DataContractError, require_columns
from .free_market_backtest import BET_REQUIRED

SQRT_TWO = sqrt(2.0)


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / SQRT_TWO))


def normalize_clv_proxy(
    raw_clv: object,
    market_type: object,
    *,
    margin_sigma: float,
    total_sigma: float,
) -> float | None:
    """Convert archive CLV into probability-equivalent favorable movement."""

    try:
        raw = float(raw_clv)
    except (TypeError, ValueError):
        return None
    if not isfinite(raw):
        return None
    market = str(market_type).lower()
    if market == "moneyline":
        return raw
    if market not in {"spread", "total"}:
        return None
    sigma = margin_sigma if market == "spread" else total_sigma
    if sigma <= 0:
        raise ValueError("residual sigma must be positive")
    return float(_normal_cdf(raw / max(6.0, sigma)) - 0.5)


def _week_block_roi_ci(
    frame: pl.DataFrame,
    *,
    seed: int = 26,
    samples: int = 3000,
) -> tuple[float | None, float | None]:
    if frame.height < 30:
        return None, None
    require_columns(frame, {"season", "week", "net_units"}, "archive_block_bootstrap")
    grouped = frame.group_by(["season", "week"]).agg(
        pl.col("net_units").sum().alias("block_units"),
        pl.len().alias("block_bets"),
    )
    if grouped.height < 8:
        profits = np.asarray(frame.get_column("net_units"), dtype=float)
        rng = np.random.default_rng(seed)
        values = np.empty(samples, dtype=float)
        for index in range(samples):
            values[index] = float(rng.choice(profits, size=profits.size, replace=True).mean())
        low, high = np.quantile(values, [0.025, 0.975])
        return float(low), float(high)

    blocks = [
        (float(row["block_units"]), int(row["block_bets"]))
        for row in grouped.iter_rows(named=True)
    ]
    rng = np.random.default_rng(seed)
    values = np.empty(samples, dtype=float)
    count = len(blocks)
    for index in range(samples):
        picked = rng.integers(0, count, size=count)
        units = sum(blocks[item][0] for item in picked)
        bets = sum(blocks[item][1] for item in picked)
        values[index] = units / max(1, bets)
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def _max_drawdown(values: list[float]) -> float:
    cumulative = 0.0
    peak = 0.0
    drawdown = 0.0
    for value in values:
        cumulative += value
        peak = max(peak, cumulative)
        drawdown = max(drawdown, peak - cumulative)
    return drawdown


def _role(market: str, side: str, line: object, odds: object) -> str:
    if market == "total":
        return side if side in {"over", "under"} else "total"
    try:
        numeric = float(line if market == "spread" else odds)
    except (TypeError, ValueError):
        return "unknown"
    if numeric < 0:
        return "favorite"
    if numeric > 0:
        return "underdog"
    return "pickem"


def _location(market: str, side: str) -> str:
    if market == "total":
        return side if side in {"over", "under"} else "total"
    return side if side in {"home", "away"} else "other"


def attach_runtime_columns(
    bets: pl.DataFrame,
    *,
    margin_sigma: float,
    total_sigma: float,
) -> pl.DataFrame:
    """Attach normalized CLV and NCAA-style market-role diagnostics."""

    require_columns(bets, BET_REQUIRED, "free_archive_bets")
    if bets.is_empty():
        return bets
    rows: list[dict[str, object]] = []
    for source in bets.iter_rows(named=True):
        row = dict(source)
        market = str(source.get("market_type") or "")
        side = str(source.get("side") or "")
        row["clv_raw"] = source.get("clv_proxy")
        row["clv_probability_equivalent"] = normalize_clv_proxy(
            source.get("clv_proxy"),
            market,
            margin_sigma=margin_sigma,
            total_sigma=total_sigma,
        )
        row["market_role"] = _role(
            market,
            side,
            source.get("line"),
            source.get("american_odds"),
        )
        row["side_location"] = _location(market, side)
        rows.append(row)
    return pl.DataFrame(rows).sort(["season", "week", "game_id", "market_type"])


def _summary(frame: pl.DataFrame) -> dict[str, object]:
    if frame.is_empty():
        return {
            "bets": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "win_rate": None,
            "units": 0.0,
            "roi": None,
            "max_drawdown": 0.0,
            "avg_clv": None,
            "positive_clv_rate": None,
            "clv_samples": 0,
            "roi_ci_95": [None, None],
            "profitable_week_rate": None,
        }
    profits = [float(value) for value in frame.get_column("net_units").to_list()]
    wins = frame.filter(pl.col("result") == "win").height
    losses = frame.filter(pl.col("result") == "loss").height
    pushes = frame.filter(pl.col("result") == "push").height
    decided = wins + losses
    clv = (
        frame.get_column("clv_probability_equivalent").drop_nulls().to_list()
        if "clv_probability_equivalent" in frame.columns
        else []
    )
    low, high = _week_block_roi_ci(frame)
    weeks = frame.group_by(["season", "week"]).agg(
        pl.col("net_units").sum().alias("week_units")
    )
    profitable_week_rate = None
    if weeks.height:
        profitable_week_rate = weeks.filter(pl.col("week_units") > 0).height / weeks.height
    return {
        "bets": frame.height,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": wins / decided if decided else None,
        "units": float(sum(profits)),
        "roi": float(sum(profits) / frame.height),
        "max_drawdown": _max_drawdown(profits),
        "avg_clv": None if not clv else float(sum(float(value) for value in clv) / len(clv)),
        "positive_clv_rate": (
            None if not clv else sum(float(value) > 0 for value in clv) / len(clv)
        ),
        "clv_samples": len(clv),
        "roi_ci_95": [low, high],
        "profitable_week_rate": profitable_week_rate,
    }


def _grouped(frame: pl.DataFrame, column: str) -> dict[str, object]:
    if frame.is_empty() or column not in frame.columns:
        return {}
    output: dict[str, object] = {}
    for value in frame.get_column(column).unique().sort().to_list():
        output[str(value)] = _summary(frame.filter(pl.col(column) == value))
    return output


def build_backtest_runtime_report(
    bets: pl.DataFrame,
    projections: pl.DataFrame,
) -> tuple[pl.DataFrame, dict[str, object]]:
    """Build diagnostics without changing the underlying archive-bet selection."""

    require_columns(
        projections,
        {
            "projected_home_margin",
            "actual_home_margin",
            "projected_total",
            "actual_total",
        },
        "archive_projections",
    )
    margin_error = np.asarray(
        projections.get_column("actual_home_margin")
        - projections.get_column("projected_home_margin"),
        dtype=float,
    )
    total_error = np.asarray(
        projections.get_column("actual_total") - projections.get_column("projected_total"),
        dtype=float,
    )
    if margin_error.size < 2 or total_error.size < 2:
        raise DataContractError("backtest runtime requires at least two projection residuals")
    margin_sigma = max(6.0, float(np.std(margin_error, ddof=1)))
    total_sigma = max(6.0, float(np.std(total_error, ddof=1)))
    enriched = attach_runtime_columns(
        bets,
        margin_sigma=margin_sigma,
        total_sigma=total_sigma,
    )
    report = {
        "overall": _summary(enriched),
        "by_market": _grouped(enriched, "market_type"),
        "by_season": _grouped(enriched, "season"),
        "by_week": _grouped(enriched, "week"),
        "by_role": _grouped(enriched, "market_role"),
        "by_side_location": _grouped(enriched, "side_location"),
        "residual_sigma": {
            "margin": margin_sigma,
            "total": total_sigma,
        },
        "methodology": {
            "confidence_interval": (
                "95% week-block bootstrap preserving within-week wager correlation"
            ),
            "clv": (
                "moneyline keeps no-vig probability movement; spread/total line movement "
                "is converted to probability-equivalent CLV using projection residual sigma"
            ),
            "selection": (
                "diagnostic layer only; underlying chronological archive opportunities "
                "are unchanged"
            ),
        },
    }
    return enriched, report


def write_backtest_runtime(
    bets: pl.DataFrame,
    projections: pl.DataFrame,
    *,
    reports_dir: str | Path = "reports",
) -> dict[str, object]:
    enriched, report = build_backtest_runtime_report(bets, projections)
    directory = Path(reports_dir)
    directory.mkdir(parents=True, exist_ok=True)
    enriched.write_csv(directory / "free_market_bets_runtime.csv")
    (directory / "free_market_runtime.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str)
    )

    segment_specs = (
        ("market_type", "free_market_by_market.csv"),
        ("season", "free_market_by_season.csv"),
        ("week", "free_market_by_week.csv"),
        ("market_role", "free_market_by_role.csv"),
        ("side_location", "free_market_by_side_location.csv"),
    )
    for column, name in segment_specs:
        rows: list[dict[str, object]] = []
        if not enriched.is_empty() and column in enriched.columns:
            for value in enriched.get_column(column).unique().sort().to_list():
                rows.append(
                    {
                        column: value,
                        **_summary(enriched.filter(pl.col(column) == value)),
                    }
                )
        if rows:
            pl.DataFrame(rows).write_csv(directory / name)
    return report
