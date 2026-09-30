"""Stage 1 data-foundation audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import polars as pl

from .contracts import PBP_REQUIRED, SCHEDULE_REQUIRED, require_columns
from .data import NFLDataClient, completed_games, schedule_to_team_games


@dataclass(frozen=True)
class Stage1Audit:
    season: int
    schedule_rows: int
    completed_games: int
    team_game_rows: int
    pbp_rows: int
    schedule_min_week: int | None
    schedule_max_week: int | None
    pbp_game_count: int

    def to_dict(self) -> dict[str, int | None]:
        return asdict(self)


def _int_or_none(value: object) -> int | None:
    return None if value is None else int(value)


def run_stage1_audit(
    season: int,
    *,
    client: NFLDataClient | None = None,
    refresh: bool = False,
) -> Stage1Audit:
    """Download/cache one season and verify the minimum Stage 1 contracts."""

    source = client or NFLDataClient()
    schedules = source.load_schedules(season, refresh=refresh)
    pbp = source.load_pbp(season, refresh=refresh)

    require_columns(schedules, SCHEDULE_REQUIRED, "schedules")
    require_columns(pbp, PBP_REQUIRED, "pbp")

    completed = completed_games(schedules)
    team_games = schedule_to_team_games(schedules)

    min_week = schedules.select(pl.col("week").min()).item()
    max_week = schedules.select(pl.col("week").max()).item()
    pbp_games = pbp.select(pl.col("game_id").n_unique()).item()

    if team_games.height != completed.height * 2:
        raise RuntimeError("team-game expansion must produce exactly two rows per completed game")

    return Stage1Audit(
        season=season,
        schedule_rows=schedules.height,
        completed_games=completed.height,
        team_game_rows=team_games.height,
        pbp_rows=pbp.height,
        schedule_min_week=_int_or_none(min_week),
        schedule_max_week=_int_or_none(max_week),
        pbp_game_count=int(pbp_games),
    )
