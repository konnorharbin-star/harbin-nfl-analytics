"""Distinguish research-only NFL bet simulations from actual price evidence.

Never promote nflverse archive-final fallback odds as playable pregame entries.
Track game/market outcomes and ROI only with explicit observation provenance.
"""
from __future__ import annotations

import polars as pl

from .contracts import DataContractError, require_columns

REQUIRED = {
    "season", "week", "game_id", "market_type", "side", "result",
    "net_units", "probability_edge", "expected_value_per_unit",
    "entry_line_observed", "entry_price_verified", "entry_price_stage",
}
EDGE_LEVELS = (0.0, 0.03, 0.06, 0.10)
MIN_BETS_PER_SEASON = 30


def audit_bet_evidence(bets: pl.DataFrame) -> dict[str, object]:
    """Report by-season evidence, with execution provenance as hard gate."""
    require_columns(bets, REQUIRED, "nfl_bet_evidence")
    if bets.select("season", "week", "game_id", "market_type").unique().height != bets.height:
        raise DataContractError("expected one selected side per game and market")
    valid = bets.filter(
        pl.col("net_units").is_finite()
        & pl.col("probability_edge").is_finite()
        & pl.col("expected_value_per_unit").is_finite()
    )
    groups = {}
    for market in ("spread", "total", "moneyline"):
        market_bets = valid.filter(pl.col("market_type") == market)
        groups[market] = {}
        for threshold in EDGE_LEVELS:
            selected = market_bets.filter(
                (pl.col("probability_edge") >= threshold)
                & (pl.col("expected_value_per_unit") > 0)
            )
            cohorts = {}
            for season in sorted(selected["season"].unique().to_list()):
                year = selected.filter(pl.col("season") == season)
                executable = year.filter(
                    pl.col("entry_line_observed")
                    & pl.col("entry_price_verified")
                    & ~pl.col("entry_price_stage").str.contains("fallback")
                )
                wins = year.filter(pl.col("result") == "win").height
                pushes = year.filter(pl.col("result") == "push").height
                net = float(year["net_units"].sum())
                cohorts[str(season)] = {
                    "research_bets": year.height,
                    "research_wins": wins,
                    "research_pushes": pushes,
                    "research_roi": net / year.height if year.height else None,
                    "verified_entry_bets": executable.height,
                    "verified_entry_net_units": float(executable["net_units"].sum())
                    if executable.height else 0.0,
                    "research_status": (
                        "DESCRIPTIVE_ONLY" if year.height >= MIN_BETS_PER_SEASON
                        else "INSUFFICIENT_SAMPLE"
                    ),
                }
            groups[market][str(threshold)] = {
                "by_season": cohorts,
                "research_bets": selected.height,
                "verified_entry_bets": selected.filter(
                    pl.col("entry_line_observed")
                    & pl.col("entry_price_verified")
                    & ~pl.col("entry_price_stage").str.contains("fallback")
                ).height,
            }
    return {
        "status": "BET_ENTRY_PROVENANCE_AUDIT_ONLY",
        "source": "FREE_NFL_MARKET_ARCHIVE",
        "archive_fallback_is_not_executable_entry": True,
        "retrospective_thresholds_not_a_validated_strategy": True,
        "independent_forward_testing_required": True,
        "bet_recommendation_approved": False,
        "targets": groups,
    }
