"""Pregame QB/OL exposure versus postgame NFL forecast-error tails.

The QB is the last-observed primary passer, NOT a verified next-game starter.
Historical sack rate is a *pressure exposure proxy*, NOT measured pressure.
Personnel rows are only used when a temporal source mode is recorded.
Same-game turnovers are outcome explanations only and never model features.
"""
from __future__ import annotations

import polars as pl

from .contracts import DataContractError, require_columns

KEYS = ("season", "week", "game_id")
QB_COLUMNS = (
    "home_qb_proxy_last_week", "away_qb_proxy_last_week",
    "qb_change_total_signal", "qb_sack_rate_total_signal",
)
OL_COLUMNS = (
    "home_ol_injury_risk", "away_ol_injury_risk",
    "home_depth_temporal_mode", "away_depth_temporal_mode",
)
THRESHOLDS = {"margin": 14.0, "total": 17.0}
MIN_GROUP_GAMES = 30
OL_RISK_THRESHOLD = 0.20
QB_SACK_EXCESS_THRESHOLD = -0.02


def _unique(frame: pl.DataFrame, source: str) -> None:
    require_columns(frame, set(KEYS), source)
    if frame.select(KEYS).unique().height != frame.height:
        raise DataContractError(f"duplicate game keys in {source}")


def assemble_pregame_risk(
    baseline: pl.DataFrame,
    qb_snapshot: pl.DataFrame,
    personnel_snapshot: pl.DataFrame | None,
) -> pl.DataFrame:
    """Whitelist pregame predictors and leave uncovered games explicitly null."""
    needed = set(KEYS) | {
        "home_team", "away_team", "margin_residual", "total_residual",
        "baseline_home_margin", "baseline_total",
    }
    require_columns(baseline, needed, "game_intelligence_baseline")
    require_columns(qb_snapshot, set(KEYS) | set(QB_COLUMNS), "pregame_qb")
    _unique(baseline, "game_intelligence_baseline")
    _unique(qb_snapshot, "pregame_qb")

    # This is a hard temporal invariant: a purported prior QB observation
    # from the target week/future cannot silently enter the research board.
    invalid = qb_snapshot.filter(
        pl.col("home_qb_proxy_last_week").is_null()
        | pl.col("away_qb_proxy_last_week").is_null()
        | (pl.col("home_qb_proxy_last_week") >= pl.col("week"))
        | (pl.col("away_qb_proxy_last_week") >= pl.col("week"))
    )
    if invalid.height:
        raise DataContractError("QB proxy observations are not strictly pregame")

    result = baseline.join(
        qb_snapshot.select(*KEYS, *QB_COLUMNS), on=list(KEYS),
        how="left", validate="1:1",
    )
    if personnel_snapshot is not None and not personnel_snapshot.is_empty():
        require_columns(
            personnel_snapshot, set(KEYS) | set(OL_COLUMNS), "pregame_personnel"
        )
        _unique(personnel_snapshot, "pregame_personnel")
        result = result.join(
            personnel_snapshot.select(*KEYS, *OL_COLUMNS),
            on=list(KEYS), how="left", validate="1:1",
        )
    else:
        result = result.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("home_ol_injury_risk"),
            pl.lit(None, dtype=pl.Float64).alias("away_ol_injury_risk"),
            pl.lit(None, dtype=pl.String).alias("home_depth_temporal_mode"),
            pl.lit(None, dtype=pl.String).alias("away_depth_temporal_mode"),
        )

    # Untimestamped season-only depth charts cannot be used to infer
    # a particular week's OL injury state.
    valid_ol = (
        pl.col("home_depth_temporal_mode").is_in(["timestamped", "weekly"])
        & pl.col("away_depth_temporal_mode").is_in(["timestamped", "weekly"])
        & pl.col("home_ol_injury_risk").is_finite()
        & pl.col("away_ol_injury_risk").is_finite()
    )
    return result.with_columns(
        pl.when(pl.col("qb_change_total_signal").is_not_null())
        .then(pl.col("qb_change_total_signal") < 0)
        .otherwise(None).alias("last_observed_qb_switch"),
        pl.when(pl.col("qb_sack_rate_total_signal").is_finite())
        .then(pl.col("qb_sack_rate_total_signal") <= QB_SACK_EXCESS_THRESHOLD)
        .otherwise(None).alias("historical_qb_sack_exposure"),
        pl.when(valid_ol)
        .then(pl.max_horizontal("home_ol_injury_risk", "away_ol_injury_risk")
              >= OL_RISK_THRESHOLD)
        .otherwise(None).alias("pregame_ol_injury_stress"),
        pl.when(valid_ol).then(
            pl.when(
                (pl.col("home_depth_temporal_mode") == "timestamped")
                & (pl.col("away_depth_temporal_mode") == "timestamped")
            ).then(pl.lit("TIMESTAMPED"))
            .otherwise(pl.lit("WEEKLY_UNTIMESTAMPED"))
        ).otherwise(pl.lit("MISSING_OR_SEASON_ONLY"))
        .alias("ol_provenance"),
    ).sort(KEYS)


def attach_postgame_turnovers(
    games: pl.DataFrame, pbp: pl.DataFrame,
) -> tuple[pl.DataFrame, str]:
    """Postgame explanatory count, not a prospective football feature."""
    required = set(KEYS) | {"posteam", "interception", "fumble_lost"}
    if not required.issubset(pbp.columns):
        return games.with_columns(
            pl.lit(None, dtype=pl.Int64).alias("turnover_net_home_advantage")
        ), "MISSING_TURNOVER_PLAY_FIELDS"

    plays = pbp.select(
        *KEYS, "posteam", "interception", "fumble_lost"
    ).join(
        games.select(*KEYS, "home_team", "away_team"),
        on=list(KEYS), how="inner", validate="m:1",
    ).with_columns(
        (
            (pl.col("interception").cast(pl.Float64, strict=False).fill_null(0) > 0)
            | (pl.col("fumble_lost").cast(pl.Float64, strict=False).fill_null(0) > 0)
        ).cast(pl.Int64).alias("_turnover")
    ).filter(
        (pl.col("posteam") == pl.col("home_team"))
        | (pl.col("posteam") == pl.col("away_team"))
    )
    if plays.is_empty():
        return games.with_columns(
            pl.lit(None, dtype=pl.Int64).alias("turnover_net_home_advantage")
        ), "NO_MATCHED_TURNOVER_PLAYS"
    totals = plays.group_by(*KEYS).agg(
        pl.when(pl.col("posteam") == pl.col("away_team"))
        .then(pl.col("_turnover")).otherwise(0).sum().alias("_away_lost"),
        pl.when(pl.col("posteam") == pl.col("home_team"))
        .then(pl.col("_turnover")).otherwise(0).sum().alias("_home_lost"),
    ).with_columns(
        (pl.col("_away_lost") - pl.col("_home_lost"))
        .alias("turnover_net_home_advantage")
    ).select(*KEYS, "turnover_net_home_advantage")
    return games.join(totals, on=list(KEYS), how="left", validate="1:1"), (
        "POSTGAME_EXPLANATION_ONLY"
    )


def _segment(data: pl.DataFrame, target: str) -> dict[str, object]:
    threshold = THRESHOLDS[target]
    residual = f"{target}_residual"
    valid = data.filter(pl.col(residual).is_finite())
    count = valid.height
    extremes = valid.filter(pl.col(residual).abs() >= threshold).height
    return {
        "games": count,
        "extreme_errors": extremes,
        "extreme_error_rate": (
            extremes / count if count >= MIN_GROUP_GAMES else None
        ),
        "status": "DESCRIPTIVE_ONLY" if count >= MIN_GROUP_GAMES
        else "INSUFFICIENT_SAMPLE",
    }


def summarize_risk_attribution(
    games: pl.DataFrame, *, turnover_status: str,
) -> dict[str, object]:
    """Predeclared risk cohorts; never optimize thresholds on outcomes."""
    require_columns(
        games, { *KEYS, "margin_residual", "total_residual",
                 "last_observed_qb_switch", "historical_qb_sack_exposure",
                 "pregame_ol_injury_stress", "ol_provenance",
                 "turnover_net_home_advantage" },
        "qb_ol_error_intelligence",
    )
    groups = {
        "qb_last_observed_switch": "last_observed_qb_switch",
        "historical_qb_sack_exposure": "historical_qb_sack_exposure",
        "ol_injury_stress": "pregame_ol_injury_stress",
    }
    results: dict[str, object] = {}
    for target in THRESHOLDS:
        cohorts = {}
        for name, column in groups.items():
            known = games.filter(pl.col(column).is_not_null())
            cohorts[name] = {
                "covered_games": known.height,
                "missing_games": games.height - known.height,
                "risk_present": _segment(
                    known.filter(pl.col(column)), target
                ),
                "risk_absent": _segment(
                    known.filter(~pl.col(column)), target
                ),
                "by_season": {
                    str(season): {
                        "risk_present": _segment(
                            subset.filter(pl.col(column)), target
                        ),
                        "risk_absent": _segment(
                            subset.filter(~pl.col(column)), target
                        ),
                    }
                    for (season,), subset in known.group_by("season")
                },
            }
        # Same-game turnovers are labels only. They are not present in
        # any predictor set or live model and cannot justify a bet.
        turnover_known = games.filter(
            pl.col("turnover_net_home_advantage").is_not_null()
        )
        cohorts["postgame_turnover_swing"] = {
            "status": "POSTGAME_EXPLANATION_ONLY",
            "covered_games": turnover_known.height,
            "two_plus_net_turnovers": _segment(
                turnover_known.filter(
                    pl.col("turnover_net_home_advantage").abs() >= 2
                ), target
            ),
            "less_than_two": _segment(
                turnover_known.filter(
                    pl.col("turnover_net_home_advantage").abs() < 2
                ), target
            ),
        }
        results[target] = {
            "extreme_threshold_points": THRESHOLDS[target],
            "overall": _segment(games, target),
            "risk_cohorts": cohorts,
        }
    modes = games.group_by("ol_provenance").len()
    return {
        "status": "RESEARCH_ONLY",
        "games": games.height,
        "turnover_source_status": turnover_status,
        "ol_provenance_counts": {
            str(row["ol_provenance"]): int(row["len"])
            for row in modes.iter_rows(named=True)
        },
        "qb_source": "LAST_OBSERVED_PRIMARY_QB_NOT_CONFIRMED_STARTER",
        "pressure_source": "HISTORICAL_QB_SACK_RATE_PROXY_NOT_PRESSURE_MEASURE",
        "ol_source": "HISTORICAL_RECONSTRUCTION_TIMESTAMP_QUALITY_VARIES",
        "postgame_turnovers_as_features": False,
        "multiple_testing_adjusted": False,
        "independent_2026_forward_validation_required": True,
        "targets": results,
        "canonical_score_changed": False,
        "betting_policy_changed": False,
        "staking_authorized": False,
    }
