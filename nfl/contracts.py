"""Schema and leakage contracts for NFL source data.

Stage 1 deliberately keeps these contracts small: they define only the fields the
rest of the platform is allowed to assume exist. Additional source columns may be
present and are preserved.
"""

from __future__ import annotations

from collections.abc import Iterable

import polars as pl


class DataContractError(ValueError):
    """Raised when source data violates a required platform contract."""


SCHEDULE_REQUIRED = {
    "season",
    "week",
    "game_id",
    "game_type",
    "gameday",
    "away_team",
    "home_team",
    "away_score",
    "home_score",
}

PBP_REQUIRED = {
    "season",
    "week",
    "game_id",
    "play_id",
    "posteam",
    "defteam",
    "play_type",
    "epa",
}

TEAM_STATS_REQUIRED = {"season", "week", "team"}
ROSTER_REQUIRED = {"season", "team", "position"}
PLAYER_STATS_REQUIRED = {
    "player_id",
    "player_name",
    "position",
    "season",
    "week",
    "season_type",
    "game_id",
    "team",
    "attempts",
    "passing_interceptions",
    "sacks_suffered",
    "passing_epa",
    "passing_cpoe",
}


def normalize_seasons(seasons: int | Iterable[int]) -> list[int]:
    """Return validated, sorted NFL season integers.

    nflverse play-by-play begins in 1999. Future seasons are intentionally not
    rejected here because schedule/roster sources may publish upcoming seasons.
    """

    if isinstance(seasons, bool):
        raise DataContractError("A boolean is not a valid explicit season selection")
    if isinstance(seasons, int):
        values = [seasons]
    else:
        values = list(seasons)

    if not values:
        raise DataContractError("At least one season is required")
    if any(not isinstance(season, int) for season in values):
        raise DataContractError("Seasons must be four-digit integers")
    if any(season < 1999 or season > 2100 for season in values):
        raise DataContractError("Season is outside the supported platform range")

    return sorted(set(values))


def require_columns(frame: pl.DataFrame, required: set[str], dataset: str) -> None:
    """Fail closed when a source schema loses a field the model depends on."""

    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DataContractError(f"{dataset} missing required columns: {', '.join(missing)}")


def require_unique(frame: pl.DataFrame, keys: list[str], dataset: str) -> None:
    """Fail when rows expected to be unique contain duplicate keys."""

    require_columns(frame, set(keys), dataset)
    duplicate_count = frame.group_by(keys).len().filter(pl.col("len") > 1).height
    if duplicate_count:
        raise DataContractError(
            f"{dataset} contains {duplicate_count} duplicated key group(s) for {keys}"
        )
