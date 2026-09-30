"""Leakage-safe NFL play-by-play efficiency features.

These features are derived strictly from football play-by-play. Sportsbook prices are
not accepted by this module. The first feature set is deliberately compact and
transparent so each signal can be audited before it is allowed into the fair-score
engine.
"""

import polars as pl

from nfl.contracts import DataContractError, require_columns


ADVANCED_PBP_REQUIRED = {
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
    "yards_gained",
    "down",
}


def pregame_pbp(pbp: pl.DataFrame, season: int, week: int) -> pl.DataFrame:
    """Return plays strictly before the target season/week."""

    require_columns(pbp, {"season", "week"}, "pbp")
    if week < 1:
        raise DataContractError("week must be >= 1")
    return pbp.filter(
        (pl.col("season") < season)
        | ((pl.col("season") == season) & (pl.col("week") < week))
    )


def assert_pbp_strictly_pregame(pbp: pl.DataFrame, season: int, week: int) -> None:
    """Fail closed if a purported feature-history frame contains target/future plays."""

    require_columns(pbp, {"season", "week"}, "pregame_pbp")
    leaked = pbp.filter(
        (pl.col("season") > season)
        | ((pl.col("season") == season) & (pl.col("week") >= week))
    )
    if leaked.height:
        raise DataContractError(f"pregame_pbp contains {leaked.height} target/future play row(s)")


def _eligible_scrimmage_plays(pbp: pl.DataFrame) -> pl.DataFrame:
    require_columns(pbp, ADVANCED_PBP_REQUIRED, "pbp")
    return (
        pbp.filter(
            pl.col("posteam").is_not_null()
            & pl.col("defteam").is_not_null()
            & pl.col("epa").is_not_null()
            & ((pl.col("pass_attempt") == 1) | (pl.col("rush_attempt") == 1))
        )
        .with_columns(
            (pl.col("qb_dropback").fill_null(0) == 1).alias("_dropback"),
            (pl.col("rush_attempt").fill_null(0) == 1).alias("_rush"),
            (pl.col("success").fill_null(0).cast(pl.Float64)).alias("_success"),
            (
                ((pl.col("pass_attempt") == 1) & (pl.col("yards_gained") >= 20))
                | ((pl.col("rush_attempt") == 1) & (pl.col("yards_gained") >= 10))
            )
            .fill_null(False)
            .cast(pl.Float64)
            .alias("_explosive"),
            pl.col("down").is_in([1, 2]).fill_null(False).alias("_early_down"),
        )
    )


def _offense_features(plays: pl.DataFrame) -> pl.DataFrame:
    return (
        plays.group_by("posteam")
        .agg(
            pl.len().alias("off_plays"),
            pl.col("epa").mean().alias("off_epa_per_play"),
            pl.col("_success").mean().alias("off_success_rate"),
            pl.when(pl.col("_dropback"))
            .then(pl.col("epa"))
            .otherwise(None)
            .mean()
            .alias("off_pass_epa_per_dropback"),
            pl.when(pl.col("_rush"))
            .then(pl.col("epa"))
            .otherwise(None)
            .mean()
            .alias("off_rush_epa_per_attempt"),
            pl.col("_explosive").mean().alias("off_explosive_rate"),
            pl.when(pl.col("_early_down"))
            .then(pl.col("epa"))
            .otherwise(None)
            .mean()
            .alias("off_early_down_epa"),
        )
        .rename({"posteam": "team"})
    )


def _defense_features(plays: pl.DataFrame) -> pl.DataFrame:
    return (
        plays.group_by("defteam")
        .agg(
            pl.len().alias("def_plays"),
            pl.col("epa").mean().alias("def_epa_per_play_allowed"),
            pl.col("_success").mean().alias("def_success_rate_allowed"),
            pl.when(pl.col("_dropback"))
            .then(pl.col("epa"))
            .otherwise(None)
            .mean()
            .alias("def_pass_epa_per_dropback_allowed"),
            pl.when(pl.col("_rush"))
            .then(pl.col("epa"))
            .otherwise(None)
            .mean()
            .alias("def_rush_epa_per_attempt_allowed"),
            pl.col("_explosive").mean().alias("def_explosive_rate_allowed"),
            pl.when(pl.col("_early_down"))
            .then(pl.col("epa"))
            .otherwise(None)
            .mean()
            .alias("def_early_down_epa_allowed"),
        )
        .rename({"defteam": "team"})
    )


def team_pbp_features(pbp: pl.DataFrame) -> pl.DataFrame:
    """Aggregate a PBP history into one offense/defense feature row per team."""

    plays = _eligible_scrimmage_plays(pbp)
    if plays.is_empty():
        raise DataContractError("no eligible scrimmage plays are available for PBP features")

    offense = _offense_features(plays)
    defense = _defense_features(plays)
    return offense.join(defense, on="team", how="full", coalesce=True).sort("team")


def pregame_team_pbp_features(
    pbp: pl.DataFrame,
    season: int,
    week: int,
) -> pl.DataFrame:
    """Build team efficiency features from plays known strictly before target week."""

    history = pregame_pbp(pbp, season, week)
    assert_pbp_strictly_pregame(history, season, week)
    return team_pbp_features(history)
