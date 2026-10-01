"""Leakage-safe NFL situational and possession-efficiency features.

The feature layer uses football play-by-play only. It deliberately isolates situations
that the generic EPA bundle does not model explicitly: red zone, money downs,
short-yardage execution, sacks, early-down pass tendency, and play volume. Sparse
situations are shrunk toward the contemporaneous league mean before matchup signals
are formed.
"""

from __future__ import annotations

import math

import polars as pl

from .contracts import DataContractError, require_columns

SITUATIONAL_PBP_REQUIRED = {
    "season",
    "week",
    "game_id",
    "posteam",
    "defteam",
    "epa",
    "success",
    "pass_attempt",
    "rush_attempt",
    "qb_dropback",
    "sack",
    "down",
    "ydstogo",
    "yardline_100",
}

SITUATIONAL_METRICS = (
    "red_zone_epa",
    "third_down_success",
    "short_yardage_success",
    "sack_rate",
    "early_down_pass_rate",
    "plays_per_game",
)


def _eligible_plays(pbp: pl.DataFrame) -> pl.DataFrame:
    require_columns(pbp, SITUATIONAL_PBP_REQUIRED, "situational_pbp")
    return pbp.filter(
        pl.col("posteam").is_not_null()
        & pl.col("defteam").is_not_null()
        & pl.col("game_id").is_not_null()
        & pl.col("epa").is_not_null()
        & ((pl.col("pass_attempt") == 1) | (pl.col("rush_attempt") == 1))
    ).with_columns(
        pl.col("success").fill_null(0).cast(pl.Float64).alias("_success"),
        (pl.col("qb_dropback").fill_null(0) == 1).alias("_dropback"),
        pl.col("sack").fill_null(0).cast(pl.Float64).alias("_sack"),
        pl.col("down").is_in([1, 2]).fill_null(False).alias("_early_down"),
        (pl.col("down") == 3).fill_null(False).alias("_third_down"),
        (
            pl.col("down").is_in([3, 4])
            & (pl.col("ydstogo").fill_null(99.0).cast(pl.Float64) <= 2.0)
        )
        .fill_null(False)
        .alias("_short_yardage"),
        (pl.col("yardline_100").fill_null(101.0).cast(pl.Float64) <= 20.0)
        .fill_null(False)
        .alias("_red_zone"),
        (pl.col("pass_attempt").fill_null(0) == 1).cast(pl.Float64).alias("_pass"),
    )


def _mean_for(plays: pl.DataFrame, mask: pl.Expr, column: str, label: str) -> float:
    values = plays.filter(mask).get_column(column).drop_nulls()
    if values.len() == 0:
        raise DataContractError(f"situational PBP has no eligible {label} observations")
    value = float(values.cast(pl.Float64).mean())
    if not math.isfinite(value):
        raise DataContractError(f"league {label} mean is non-finite")
    return value


def _league_priors(plays: pl.DataFrame) -> dict[str, float]:
    team_games = plays.select(["game_id", "posteam"]).unique().height
    if team_games <= 0:
        raise DataContractError("situational PBP has no offense team-games")
    return {
        "red_zone_epa": _mean_for(plays, pl.col("_red_zone"), "epa", "red-zone EPA"),
        "third_down_success": _mean_for(
            plays, pl.col("_third_down"), "_success", "third-down success"
        ),
        "short_yardage_success": _mean_for(
            plays, pl.col("_short_yardage"), "_success", "short-yardage success"
        ),
        "sack_rate": _mean_for(plays, pl.col("_dropback"), "_sack", "sack rate"),
        "early_down_pass_rate": _mean_for(
            plays, pl.col("_early_down"), "_pass", "early-down pass rate"
        ),
        "plays_per_game": float(plays.height / team_games),
    }


def _role_aggregate(
    plays: pl.DataFrame,
    *,
    role: str,
    priors: dict[str, float],
) -> pl.DataFrame:
    if role == "offense":
        team_col = "posteam"
        prefix = "off"
    elif role == "defense":
        team_col = "defteam"
        prefix = "def"
    else:
        raise ValueError("role must be offense or defense")

    grouped = plays.group_by(team_col).agg(
        pl.len().alias("_plays"),
        pl.col("game_id").n_unique().alias("_games"),
        pl.when(pl.col("_red_zone"))
        .then(pl.col("epa"))
        .otherwise(None)
        .sum()
        .fill_null(0.0)
        .alias("_rz_sum"),
        pl.col("_red_zone").cast(pl.Int64).sum().alias("_rz_n"),
        pl.when(pl.col("_third_down"))
        .then(pl.col("_success"))
        .otherwise(None)
        .sum()
        .fill_null(0.0)
        .alias("_third_sum"),
        pl.col("_third_down").cast(pl.Int64).sum().alias("_third_n"),
        pl.when(pl.col("_short_yardage"))
        .then(pl.col("_success"))
        .otherwise(None)
        .sum()
        .fill_null(0.0)
        .alias("_short_sum"),
        pl.col("_short_yardage").cast(pl.Int64).sum().alias("_short_n"),
        pl.when(pl.col("_dropback"))
        .then(pl.col("_sack"))
        .otherwise(None)
        .sum()
        .fill_null(0.0)
        .alias("_sack_sum"),
        pl.col("_dropback").cast(pl.Int64).sum().alias("_dropbacks"),
        pl.when(pl.col("_early_down"))
        .then(pl.col("_pass"))
        .otherwise(None)
        .sum()
        .fill_null(0.0)
        .alias("_early_pass_sum"),
        pl.col("_early_down").cast(pl.Int64).sum().alias("_early_n"),
    )

    rz_prior = 20.0
    third_prior = 20.0
    short_prior = 12.0
    sack_prior = 40.0
    early_prior = 80.0
    game_prior = 2.0
    output = grouped.with_columns(
        (
            (pl.col("_rz_sum") + rz_prior * priors["red_zone_epa"])
            / (pl.col("_rz_n") + rz_prior)
        ).alias(
            f"{prefix}_red_zone_epa" if role == "offense" else "def_red_zone_epa_allowed"
        ),
        (
            (pl.col("_third_sum") + third_prior * priors["third_down_success"])
            / (pl.col("_third_n") + third_prior)
        ).alias(
            f"{prefix}_third_down_success"
            if role == "offense"
            else "def_third_down_success_allowed"
        ),
        (
            (pl.col("_short_sum") + short_prior * priors["short_yardage_success"])
            / (pl.col("_short_n") + short_prior)
        ).alias(
            f"{prefix}_short_yardage_success"
            if role == "offense"
            else "def_short_yardage_success_allowed"
        ),
        (
            (pl.col("_sack_sum") + sack_prior * priors["sack_rate"])
            / (pl.col("_dropbacks") + sack_prior)
        ).alias(f"{prefix}_sack_rate" if role == "offense" else "def_sack_rate_generated"),
        (
            (pl.col("_early_pass_sum") + early_prior * priors["early_down_pass_rate"])
            / (pl.col("_early_n") + early_prior)
        ).alias(
            f"{prefix}_early_down_pass_rate"
            if role == "offense"
            else "def_early_down_pass_rate_allowed"
        ),
        (
            (pl.col("_plays") + game_prior * priors["plays_per_game"])
            / (pl.col("_games") + game_prior)
        ).alias(
            f"{prefix}_plays_per_game" if role == "offense" else "def_plays_per_game_allowed"
        ),
    )
    keep = [
        team_col,
        f"{prefix}_red_zone_epa" if role == "offense" else "def_red_zone_epa_allowed",
        f"{prefix}_third_down_success"
        if role == "offense"
        else "def_third_down_success_allowed",
        f"{prefix}_short_yardage_success"
        if role == "offense"
        else "def_short_yardage_success_allowed",
        f"{prefix}_sack_rate" if role == "offense" else "def_sack_rate_generated",
        f"{prefix}_early_down_pass_rate"
        if role == "offense"
        else "def_early_down_pass_rate_allowed",
        f"{prefix}_plays_per_game" if role == "offense" else "def_plays_per_game_allowed",
    ]
    return output.select(keep).rename({team_col: "team"})


def team_situational_features(pbp: pl.DataFrame) -> pl.DataFrame:
    """Return smoothed offense/defense situational state for every observed team."""

    plays = _eligible_plays(pbp)
    if plays.is_empty():
        raise DataContractError("no eligible scrimmage plays for situational features")
    priors = _league_priors(plays)
    offense = _role_aggregate(plays, role="offense", priors=priors)
    defense = _role_aggregate(plays, role="defense", priors=priors)
    result = offense.join(defense, on="team", how="full", coalesce=True).sort("team")
    numeric = [column for column in result.columns if column != "team"]
    if result.select([pl.col(column).is_null().any() for column in numeric]).row(0).count(True):
        raise DataContractError("situational team features contain null values")
    return result


def situational_matchup_signals(
    features: pl.DataFrame,
    home_team: str,
    away_team: str,
) -> dict[str, float]:
    """Build symmetric margin/total matchup signals from smoothed team state."""

    require_columns(features, {"team"}, "situational_team_features")
    rows = {str(row["team"]): row for row in features.iter_rows(named=True)}
    if home_team not in rows or away_team not in rows:
        raise DataContractError(f"missing situational state for {away_team} at {home_team}")
    home = rows[home_team]
    away = rows[away_team]

    pairs = {
        "red_zone_epa": ("off_red_zone_epa", "def_red_zone_epa_allowed"),
        "third_down_success": (
            "off_third_down_success",
            "def_third_down_success_allowed",
        ),
        "short_yardage_success": (
            "off_short_yardage_success",
            "def_short_yardage_success_allowed",
        ),
        "early_down_pass_rate": (
            "off_early_down_pass_rate",
            "def_early_down_pass_rate_allowed",
        ),
        "plays_per_game": ("off_plays_per_game", "def_plays_per_game_allowed"),
    }
    signals: dict[str, float] = {}
    for name, (off_col, def_col) in pairs.items():
        home_expected = (float(home[off_col]) + float(away[def_col])) / 2.0
        away_expected = (float(away[off_col]) + float(home[def_col])) / 2.0
        signals[f"sit_{name}_margin_signal"] = home_expected - away_expected
        signals[f"sit_{name}_total_signal"] = (home_expected + away_expected) / 2.0

    home_sack = (float(home["off_sack_rate"]) + float(away["def_sack_rate_generated"])) / 2.0
    away_sack = (float(away["off_sack_rate"]) + float(home["def_sack_rate_generated"])) / 2.0
    signals["sit_sack_rate_margin_signal"] = away_sack - home_sack
    signals["sit_sack_rate_total_signal"] = -((home_sack + away_sack) / 2.0)
    return signals


def situational_feature_columns(metrics: tuple[str, ...], *, target: str) -> tuple[str, ...]:
    if target == "margin_residual":
        suffix = "margin_signal"
    elif target == "total_residual":
        suffix = "total_signal"
    else:
        raise ValueError(f"unsupported residual target: {target}")
    unknown = set(metrics) - set(SITUATIONAL_METRICS)
    if unknown:
        raise ValueError(f"unknown situational metrics: {sorted(unknown)}")
    return tuple(f"sit_{metric}_{suffix}" for metric in metrics)
