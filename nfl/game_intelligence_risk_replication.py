"""Frozen QB/OL risk-combination replication checks across historical seasons.

Outcome labels only measure misses. No combination becomes an approved betting
feature; pregame signals are observational proxies, not verified starters.
"""
from __future__ import annotations

import polars as pl

from .contracts import DataContractError, require_columns
from .game_intelligence_risk import MIN_GROUP_GAMES, THRESHOLDS

RISK_COLUMNS = (
    "last_observed_qb_switch",
    "historical_qb_sack_exposure",
    "pregame_ol_injury_stress",
)
COMBINATIONS = {
    "qb_and_sack": RISK_COLUMNS[:2],
    "qb_and_ol": (RISK_COLUMNS[0], RISK_COLUMNS[2]),
    "sack_and_ol": RISK_COLUMNS[1:],
    "qb_sack_and_ol": RISK_COLUMNS,
}
EVALUATION_SEASONS = (2024, 2025)


def _risk_rate(frame: pl.DataFrame, residual: str, threshold: float) -> dict[str, object]:
    n = frame.height
    misses = frame.filter(pl.col(residual).abs() >= threshold).height
    return {
        "games": n,
        "extreme_misses": misses,
        "extreme_rate": misses / n if n >= MIN_GROUP_GAMES else None,
        "status": "DESCRIPTIVE_ONLY" if n >= MIN_GROUP_GAMES
        else "INSUFFICIENT_SAMPLE",
    }


def audit_risk_combinations(
    games: pl.DataFrame, *, seasons: tuple[int, ...] = EVALUATION_SEASONS,
) -> dict[str, object]:
    """Compare joint risk-present versus risk-absent with frozen definitions."""
    require_columns(
        games,
        {"season", "week", "game_id", *RISK_COLUMNS,
         "margin_residual", "total_residual"},
        "risk_combinations",
    )
    if games.select("season", "week", "game_id").unique().height != games.height:
        raise DataContractError("duplicate game in risk combinations")
    if not seasons or tuple(sorted(set(seasons))) != seasons:
        raise ValueError("seasons must be unique and increasing")
    results: dict[str, object] = {}
    for target, threshold in THRESHOLDS.items():
        residual = f"{target}_residual"
        by_combination: dict[str, object] = {}
        for label, columns in COMBINATIONS.items():
            by_season: dict[str, object] = {}
            for season in seasons:
                subset = games.filter(pl.col("season") == season)
                known = subset.filter(
                    pl.all_horizontal([pl.col(c).is_not_null() for c in columns])
                    & pl.col(residual).is_finite()
                )
                exposed = known.filter(
                    pl.all_horizontal([pl.col(c) for c in columns])
                )
                not_exposed = known.filter(
                    ~pl.all_horizontal([pl.col(c) for c in columns])
                )
                a = _risk_rate(exposed, residual, threshold)
                b = _risk_rate(not_exposed, residual, threshold)
                difference = (
                    a["extreme_rate"] - b["extreme_rate"]
                    if a["extreme_rate"] is not None
                    and b["extreme_rate"] is not None else None
                )
                by_season[str(season)] = {
                    "covered_games": known.height,
                    "missing_games": subset.height - known.height,
                    "joint_risk_present": a,
                    "other_known_games": b,
                    "extreme_rate_difference": difference,
                }
            replicated = all(
                value["extreme_rate_difference"] is not None
                and value["extreme_rate_difference"] > 0
                for value in by_season.values()
            )
            by_combination[label] = {
                "status": (
                    "REPLICATED_DESCRIPTIVE_ASSOCIATION"
                    if replicated else "NOT_REPLICATED_OR_UNDERPOWERED"
                ),
                "by_season": by_season,
                "requires_independent_forward_test": True,
                "staking_authorized": False,
            }
        results[target] = by_combination
    return {
        "status": "HISTORICAL_REPLICATION_RESEARCH_ONLY",
        "seasons": list(seasons),
        "risk_definitions_frozen": True,
        "minimum_each_arm_per_season": MIN_GROUP_GAMES,
        "overlapping_cohorts": True,
        "not_multiple_testing_adjusted": True,
        "pregame_qb_not_confirmed_starter": True,
        "ol_weekly_unverified_provenance_possible": True,
        "never_use_postgame_turnovers_as_features": True,
        "independent_2026_forward_validation_required": True,
        "canonical_projection_changed": False,
        "staking_authorized": False,
        "targets": results,
    }
