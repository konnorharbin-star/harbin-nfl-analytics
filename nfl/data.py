"""NFL source ingestion and leakage-safe schedule helpers.

The fair-score model will consume only data exposed through this module. That gives
us one place to enforce schema contracts, cache provenance, and pregame cutoffs.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Callable

import nflreadpy as nfl
import polars as pl

from .contracts import (
    PBP_REQUIRED,
    ROSTER_REQUIRED,
    SCHEDULE_REQUIRED,
    TEAM_STATS_REQUIRED,
    DataContractError,
    normalize_seasons,
    require_columns,
    require_unique,
)


class NFLDataClient:
    """Thin, contract-checked wrapper around nflreadpy/nflverse datasets."""

    def __init__(self, cache_dir: str | Path = "data/cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, dataset: str, seasons: list[int]) -> Path:
        season_key = "-".join(str(season) for season in seasons)
        return self.cache_dir / f"{dataset}_{season_key}.parquet"

    def _load(
        self,
        dataset: str,
        seasons: int | list[int],
        loader: Callable[[list[int]], pl.DataFrame],
        required: set[str],
        *,
        refresh: bool = False,
    ) -> pl.DataFrame:
        years = normalize_seasons(seasons)
        cache_path = self._cache_path(dataset, years)

        if cache_path.exists() and not refresh:
            frame = pl.read_parquet(cache_path)
        else:
            frame = loader(years)
            if not isinstance(frame, pl.DataFrame):
                raise DataContractError(f"{dataset} loader did not return a Polars DataFrame")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            frame.write_parquet(cache_path)

        require_columns(frame, required, dataset)
        return frame

    def load_schedules(
        self, seasons: int | list[int], *, refresh: bool = False
    ) -> pl.DataFrame:
        frame = self._load(
            "schedules",
            seasons,
            lambda years: nfl.load_schedules(years),
            SCHEDULE_REQUIRED,
            refresh=refresh,
        )
        require_unique(frame, ["game_id"], "schedules")
        return frame

    def load_pbp(self, seasons: int | list[int], *, refresh: bool = False) -> pl.DataFrame:
        return self._load(
            "pbp",
            seasons,
            lambda years: nfl.load_pbp(years),
            PBP_REQUIRED,
            refresh=refresh,
        )

    def load_team_stats(
        self, seasons: int | list[int], *, refresh: bool = False
    ) -> pl.DataFrame:
        return self._load(
            "team_stats",
            seasons,
            lambda years: nfl.load_team_stats(years),
            TEAM_STATS_REQUIRED,
            refresh=refresh,
        )

    def load_rosters_weekly(
        self, seasons: int | list[int], *, refresh: bool = False
    ) -> pl.DataFrame:
        return self._load(
            "rosters_weekly",
            seasons,
            lambda years: nfl.load_rosters_weekly(years),
            ROSTER_REQUIRED,
            refresh=refresh,
        )

    def load_injuries(
        self, seasons: int | list[int], *, refresh: bool = False
    ) -> pl.DataFrame:
        # Injury schemas have changed over time; Stage 1 contracts only the temporal keys.
        return self._load(
            "injuries",
            seasons,
            lambda years: nfl.load_injuries(years),
            {"season", "week", "team"},
            refresh=refresh,
        )

    def load_depth_charts(
        self, seasons: int | list[int], *, refresh: bool = False
    ) -> pl.DataFrame:
        # nflverse depth charts are date-based in the modern feed. We intentionally do
        # not assume player/depth column names until Stage 4 personnel work is validated.
        return self._load(
            "depth_charts",
            seasons,
            lambda years: nfl.load_depth_charts(years),
            {"season", "team"},
            refresh=refresh,
        )


def _coerce_date(value: date | datetime | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


def _with_game_date(frame: pl.DataFrame) -> pl.DataFrame:
    require_columns(frame, {"gameday"}, "schedules")
    dtype = frame.schema["gameday"]
    if dtype == pl.Date:
        expr = pl.col("gameday")
    elif isinstance(dtype, pl.Datetime):
        expr = pl.col("gameday").cast(pl.Date)
    else:
        expr = pl.col("gameday").cast(pl.String).str.to_date(strict=False)
    return frame.with_columns(expr.alias("_game_date"))


def completed_games(
    schedules: pl.DataFrame, *, as_of: date | datetime | str | None = None
) -> pl.DataFrame:
    """Return only games with final scores known by the requested calendar date."""

    require_columns(schedules, SCHEDULE_REQUIRED, "schedules")
    out = schedules.filter(
        pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null()
    )
    if as_of is not None:
        cutoff = _coerce_date(as_of)
        out = _with_game_date(out).filter(pl.col("_game_date") <= pl.lit(cutoff)).drop(
            "_game_date"
        )
    return out


def pregame_history(schedules: pl.DataFrame, season: int, week: int) -> pl.DataFrame:
    """Return completed games strictly before ``season/week``.

    This is the primary Stage 1 anti-leakage primitive. Any rolling team state for a
    target game in week W must be created from this history (or an equivalent source
    timestamped before kickoff), never from W results.
    """

    if week < 1:
        raise DataContractError("week must be >= 1")
    history = completed_games(schedules)
    return history.filter(
        (pl.col("season") < season)
        | ((pl.col("season") == season) & (pl.col("week") < week))
    )


def assert_strictly_pregame(history: pl.DataFrame, season: int, week: int) -> None:
    """Fail closed if a supposedly pregame frame contains target/future games."""

    require_columns(history, {"season", "week"}, "pregame_history")
    leaked = history.filter(
        (pl.col("season") > season)
        | ((pl.col("season") == season) & (pl.col("week") >= week))
    )
    if leaked.height:
        raise DataContractError(
            f"pregame_history contains {leaked.height} target/future game row(s)"
        )


def schedule_to_team_games(schedules: pl.DataFrame) -> pl.DataFrame:
    """Convert completed schedule rows into one row per team-game.

    The resulting table is the canonical input for Stage 2 opponent-adjusted ratings.
    """

    games = completed_games(schedules)
    common = ["season", "week", "game_id", "game_type", "gameday"]

    home = games.select(
        common
        + [
            pl.col("home_team").alias("team"),
            pl.col("away_team").alias("opponent"),
            pl.col("home_score").cast(pl.Float64).alias("points_for"),
            pl.col("away_score").cast(pl.Float64).alias("points_against"),
            pl.lit(True).alias("is_home"),
        ]
    )
    away = games.select(
        common
        + [
            pl.col("away_team").alias("team"),
            pl.col("home_team").alias("opponent"),
            pl.col("away_score").cast(pl.Float64).alias("points_for"),
            pl.col("home_score").cast(pl.Float64).alias("points_against"),
            pl.lit(False).alias("is_home"),
        ]
    )

    return pl.concat([home, away], how="vertical").with_columns(
        (pl.col("points_for") - pl.col("points_against")).alias("point_margin")
    )
