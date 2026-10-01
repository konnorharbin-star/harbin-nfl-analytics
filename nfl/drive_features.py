"""Leakage-safe NFL drive-level scoring opportunity and finishing features.

The drive layer deliberately works at a lower frequency than the existing per-play EPA
features. It measures possession outcomes, red-zone access, ball security, field
position, drive sustain, and possession volume using football play-by-play only.
Sportsbook prices are not accepted by this module.
"""

from __future__ import annotations

import math

import polars as pl

from .contracts import DataContractError, require_columns

DRIVE_PBP_REQUIRED = {
    "season",
    "week",
    "game_id",
    "play_id",
    "drive",
    "posteam",
    "defteam",
    "play_type",
    "yardline_100",
    "first_down",
    "interception",
    "fumble_lost",
    "posteam_score",
    "posteam_score_post",
}

DRIVE_METRICS = (
    "points_per_drive",
    "scoring_drive_rate",
    "red_zone_entry_rate",
    "red_zone_points_per_entry",
    "drive_security",
    "start_field_position",
    "first_downs_per_drive",
    "drives_per_game",
)


def _drive_table(pbp: pl.DataFrame) -> pl.DataFrame:
    """Collapse eligible PBP rows to one row per offensive drive."""

    require_columns(pbp, DRIVE_PBP_REQUIRED, "drive_pbp")
    plays = pbp.filter(
        pl.col("game_id").is_not_null()
        & pl.col("drive").is_not_null()
        & pl.col("posteam").is_not_null()
        & pl.col("defteam").is_not_null()
        & pl.col("play_id").is_not_null()
        & (pl.col("play_type") != "kickoff")
    )
    if plays.is_empty():
        raise DataContractError("no eligible drive rows are available")

    score_delta = (
        pl.col("posteam_score_post").drop_nulls().max()
        - pl.col("posteam_score").drop_nulls().min()
    ).fill_null(0.0)
    drives = (
        plays.group_by(["game_id", "drive", "posteam", "defteam"])
        .agg(
            score_delta.cast(pl.Float64).clip(lower_bound=0.0).alias("drive_points"),
            pl.col("yardline_100").drop_nulls().min().alias("best_yardline_100"),
            pl.col("yardline_100")
            .sort_by("play_id")
            .drop_nulls()
            .first()
            .alias("start_yardline_100"),
            pl.col("first_down")
            .fill_null(0)
            .cast(pl.Float64)
            .sum()
            .alias("first_downs"),
            (
                (pl.col("interception").fill_null(0) == 1)
                | (pl.col("fumble_lost").fill_null(0) == 1)
            )
            .any()
            .alias("turnover"),
        )
        .with_columns(
            (pl.col("drive_points") >= 3.0).cast(pl.Float64).alias("scoring_drive"),
            (pl.col("best_yardline_100") <= 20.0)
            .fill_null(False)
            .cast(pl.Float64)
            .alias("red_zone_entry"),
            (~pl.col("turnover")).cast(pl.Float64).alias("drive_security"),
            (100.0 - pl.col("start_yardline_100"))
            .cast(pl.Float64)
            .alias("start_field_position"),
        )
    )
    return drives


def _finite_mean(frame: pl.DataFrame, column: str, label: str) -> float:
    values = frame.get_column(column).drop_nulls().cast(pl.Float64)
    if values.len() == 0:
        raise DataContractError(f"drive table has no eligible {label} observations")
    value = float(values.mean())
    if not math.isfinite(value):
        raise DataContractError(f"league {label} mean is non-finite")
    return value


def _league_priors(drives: pl.DataFrame) -> dict[str, float]:
    red_zone = drives.filter(pl.col("red_zone_entry") == 1.0)
    team_games = drives.select(["game_id", "posteam"]).unique().height
    if team_games <= 0:
        raise DataContractError("drive table has no offense team-games")
    return {
        "points_per_drive": _finite_mean(drives, "drive_points", "points per drive"),
        "scoring_drive_rate": _finite_mean(drives, "scoring_drive", "scoring drive rate"),
        "red_zone_entry_rate": _finite_mean(
            drives, "red_zone_entry", "red-zone entry rate"
        ),
        "red_zone_points_per_entry": _finite_mean(
            red_zone, "drive_points", "red-zone points per entry"
        ),
        "drive_security": _finite_mean(drives, "drive_security", "drive security"),
        "start_field_position": _finite_mean(
            drives, "start_field_position", "start field position"
        ),
        "first_downs_per_drive": _finite_mean(
            drives, "first_downs", "first downs per drive"
        ),
        "drives_per_game": float(drives.height / team_games),
    }


def _role_aggregate(
    drives: pl.DataFrame,
    *,
    role: str,
    priors: dict[str, float],
) -> pl.DataFrame:
    if role == "offense":
        team_col = "posteam"
        prefix = "off"
        suffix = ""
    elif role == "defense":
        team_col = "defteam"
        prefix = "def"
        suffix = "_allowed"
    else:
        raise ValueError("role must be offense or defense")

    grouped = drives.group_by(team_col).agg(
        pl.len().alias("_drives"),
        pl.col("game_id").n_unique().alias("_games"),
        pl.col("drive_points").sum().alias("_points"),
        pl.col("scoring_drive").sum().alias("_scoring_drives"),
        pl.col("red_zone_entry").sum().alias("_red_zone_entries"),
        pl.when(pl.col("red_zone_entry") == 1.0)
        .then(pl.col("drive_points"))
        .otherwise(None)
        .sum()
        .fill_null(0.0)
        .alias("_red_zone_points"),
        pl.col("drive_security").sum().alias("_secure_drives"),
        pl.col("start_field_position").drop_nulls().sum().alias("_start_field_sum"),
        pl.col("start_field_position").is_not_null().sum().alias("_start_field_n"),
        pl.col("first_downs").sum().alias("_first_downs"),
    )

    drive_prior = 16.0
    red_zone_prior = 8.0
    field_prior = 12.0
    game_prior = 2.0
    output = grouped.with_columns(
        (
            (pl.col("_points") + drive_prior * priors["points_per_drive"])
            / (pl.col("_drives") + drive_prior)
        ).alias(f"{prefix}_points_per_drive{suffix}"),
        (
            (pl.col("_scoring_drives") + drive_prior * priors["scoring_drive_rate"])
            / (pl.col("_drives") + drive_prior)
        ).alias(f"{prefix}_scoring_drive_rate{suffix}"),
        (
            (pl.col("_red_zone_entries") + drive_prior * priors["red_zone_entry_rate"])
            / (pl.col("_drives") + drive_prior)
        ).alias(f"{prefix}_red_zone_entry_rate{suffix}"),
        (
            (
                pl.col("_red_zone_points")
                + red_zone_prior * priors["red_zone_points_per_entry"]
            )
            / (pl.col("_red_zone_entries") + red_zone_prior)
        ).alias(f"{prefix}_red_zone_points_per_entry{suffix}"),
        (
            (pl.col("_secure_drives") + drive_prior * priors["drive_security"])
            / (pl.col("_drives") + drive_prior)
        ).alias(f"{prefix}_drive_security{suffix}"),
        (
            (
                pl.col("_start_field_sum")
                + field_prior * priors["start_field_position"]
            )
            / (pl.col("_start_field_n") + field_prior)
        ).alias(f"{prefix}_start_field_position{suffix}"),
        (
            (pl.col("_first_downs") + drive_prior * priors["first_downs_per_drive"])
            / (pl.col("_drives") + drive_prior)
        ).alias(f"{prefix}_first_downs_per_drive{suffix}"),
        (
            (pl.col("_drives") + game_prior * priors["drives_per_game"])
            / (pl.col("_games") + game_prior)
        ).alias(f"{prefix}_drives_per_game{suffix}"),
    )

    keep = [team_col]
    keep.extend(f"{prefix}_{metric}{suffix}" for metric in DRIVE_METRICS)
    return output.select(keep).rename({team_col: "team"})


def team_drive_features(pbp: pl.DataFrame) -> pl.DataFrame:
    """Return smoothed offense/defense drive state for every observed team."""

    drives = _drive_table(pbp)
    priors = _league_priors(drives)
    offense = _role_aggregate(drives, role="offense", priors=priors)
    defense = _role_aggregate(drives, role="defense", priors=priors)
    result = offense.join(defense, on="team", how="full", coalesce=True).sort("team")
    numeric = [column for column in result.columns if column != "team"]
    checks = result.select([pl.col(column).is_null().any() for column in numeric]).row(0)
    if any(checks):
        raise DataContractError("drive team features contain null values")
    return result


def drive_matchup_signals(
    features: pl.DataFrame,
    home_team: str,
    away_team: str,
) -> dict[str, float]:
    """Build symmetric margin/total signals from smoothed drive state."""

    require_columns(features, {"team"}, "drive_team_features")
    rows = {str(row["team"]): row for row in features.iter_rows(named=True)}
    if home_team not in rows or away_team not in rows:
        raise DataContractError(f"missing drive state for {away_team} at {home_team}")
    home = rows[home_team]
    away = rows[away_team]

    signals: dict[str, float] = {}
    for metric in DRIVE_METRICS:
        off_col = f"off_{metric}"
        def_col = f"def_{metric}_allowed"
        home_expected = (float(home[off_col]) + float(away[def_col])) / 2.0
        away_expected = (float(away[off_col]) + float(home[def_col])) / 2.0
        signals[f"drive_{metric}_margin_signal"] = home_expected - away_expected
        signals[f"drive_{metric}_total_signal"] = (home_expected + away_expected) / 2.0
    return signals


def drive_feature_columns(metrics: tuple[str, ...], *, target: str) -> tuple[str, ...]:
    """Return model columns for a named drive metric subset and target."""

    if target == "margin_residual":
        suffix = "margin_signal"
    elif target == "total_residual":
        suffix = "total_signal"
    else:
        raise ValueError(f"unsupported residual target: {target}")
    unknown = set(metrics) - set(DRIVE_METRICS)
    if unknown:
        raise ValueError(f"unknown drive metrics: {sorted(unknown)}")
    return tuple(f"drive_{metric}_{suffix}" for metric in metrics)
