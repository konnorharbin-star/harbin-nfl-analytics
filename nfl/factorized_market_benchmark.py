"""Frozen possession × scoring efficiency market-relative NFL audit.

Reuses the existing historically investigated Stage 19 architecture without
reselecting ridge settings on 2024/2025 test outcomes. Archived market lines
are unverified close proxies and never enter model fitting.
"""
from __future__ import annotations

import polars as pl

from .contracts import DataContractError, require_columns
from .factorized_score import build_factorized_walkforward

KEYS = ("season", "week", "game_id")
SEASONS = (2024, 2025)
FIXED_RIDGE = 8.0
MIN_GAMES_PER_SEASON = 150


def evaluate_factorized_market(
    schedules: pl.DataFrame,
    pbp: pl.DataFrame,
    *,
    seasons: tuple[int, ...] = SEASONS,
) -> dict[str, object]:
    require_columns(schedules, set(KEYS) | {"spread_line", "total_line"},
                    "factorized_market_schedules")
    if schedules.select(KEYS).unique().height != schedules.height:
        raise DataContractError("duplicate schedule keys")
    if not seasons or tuple(sorted(set(seasons))) != seasons:
        raise ValueError("evaluation seasons must be increasing and unique")
    by_season: dict[str, object] = {}
    for season in seasons:
        forecasts = build_factorized_walkforward(
            schedules, pbp, season, start_week=5, end_week=18,
            factorized_ridge=FIXED_RIDGE, baseline_ridge=8.0,
        )
        if forecasts.select(KEYS).unique().height != forecasts.height:
            raise DataContractError("duplicate factorized forecasts")
        paired = forecasts.join(
            schedules.select(*KEYS, "spread_line", "total_line"),
            on=list(KEYS), how="left", validate="1:1",
        )
        results: dict[str, object] = {}
        for target, truth, base, alt, market in (
            ("margin", "actual_home_margin", "baseline_home_margin",
             "factorized_home_margin", "spread_line"),
            ("total", "actual_total", "baseline_total",
             "factorized_total", "total_line"),
        ):
            valid = paired.filter(pl.all_horizontal([
                pl.col(c).cast(pl.Float64, strict=False).is_finite()
                for c in (truth, base, alt, market)
            ]))
            metrics = {
                key: float(valid.select(
                    (pl.col(column) - pl.col(truth)).abs().mean()
                ).item()) if valid.height else None
                for key, column in (
                    ("baseline_mae", base), ("factorized_mae", alt),
                    ("market_archive_mae", market),
                )
            }
            enough = valid.height >= MIN_GAMES_PER_SEASON
            results[target] = {
                "paired_games": valid.height,
                "missing_market_games": paired.height - valid.height,
                **metrics,
                "beats_baseline": (
                    enough and metrics["factorized_mae"] < metrics["baseline_mae"]
                ),
                "beats_archived_market": (
                    enough
                    and metrics["factorized_mae"] < metrics["market_archive_mae"]
                ),
                "status": "HISTORICAL_RESEARCH" if enough
                else "INSUFFICIENT_PAIRED_GAMES",
            }
        by_season[str(season)] = results
    return {
        "status": "FIXED_FACTORIZED_MARKET_RESEARCH_ONLY",
        "specification": "preexisting Stage19 possessions x PPD; fixed ridge=8",
        "selection_on_evaluation_seasons": False,
        "architecture_previously_explored": True,
        "market_stage": "NFLVERSE_ARCHIVE_FINAL_UNVERIFIED_CLOSE_PROXY",
        "market_timestamp_verified": False,
        "sportsbook_inputs_to_model": False,
        "production_changed": False,
        "staking_authorized": False,
        "by_season": by_season,
        "repeatable_margin_market_win": all(
            by_season[str(s)]["margin"]["beats_archived_market"] for s in seasons
        ),
        "repeatable_total_market_win": all(
            by_season[str(s)]["total"]["beats_archived_market"] for s in seasons
        ),
    }
