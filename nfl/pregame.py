"""Pregame timing guards for live NFL projections and recommendations."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import polars as pl

from .contracts import DataContractError, require_columns

NFL_SCHEDULE_TIMEZONE = ZoneInfo("America/New_York")


def schedule_kickoff_utc(row: Mapping[str, object]) -> datetime | None:
    """Return the scheduled NFL kickoff in UTC when date/time fields are usable."""

    day = row.get("gameday")
    time = row.get("gametime")
    if day is None or time in {None, ""}:
        return None

    day_text = str(day)[:10]
    try:
        local = datetime.fromisoformat(f"{day_text}T{time}")
    except (TypeError, ValueError):
        return None
    if local.tzinfo is None:
        local = local.replace(tzinfo=NFL_SCHEDULE_TIMEZONE)
    return local.astimezone(UTC)


def filter_future_kickoffs(
    frame: pl.DataFrame,
    *,
    as_of: datetime | None = None,
    require_kickoff: bool = False,
    dataset: str = "target_games",
) -> pl.DataFrame:
    """Keep only games that have not kicked off at the requested cutoff.

    A delayed final-score feed must never leave an already-started game eligible for
    live projection, market comparison, publication, or a new decision. When
    require_kickoff is true, missing/invalid kickoff data fails closed.
    """

    require_columns(frame, {"game_id", "gameday"}, dataset)
    if frame.is_empty():
        return frame

    reference = as_of or datetime.now(UTC)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=UTC)
    reference = reference.astimezone(UTC)

    keep: list[bool] = []
    missing: list[str] = []
    for row in frame.iter_rows(named=True):
        kickoff = schedule_kickoff_utc(row)
        if kickoff is None:
            missing.append(str(row.get("game_id") or "unknown"))
            keep.append(not require_kickoff)
            continue
        keep.append(kickoff > reference)

    if require_kickoff and missing:
        raise DataContractError(
            f"{dataset} missing/invalid kickoff for game(s): " + ", ".join(sorted(missing))
        )
    return frame.filter(pl.Series("_pregame_keep", keep))


def kickoff_iso_map(frame: pl.DataFrame) -> dict[str, str | None]:
    """Map game IDs to normalized UTC kickoff timestamps."""

    require_columns(frame, {"game_id", "gameday"}, "current_targets")
    output: dict[str, str | None] = {}
    for row in frame.iter_rows(named=True):
        kickoff = schedule_kickoff_utc(row)
        output[str(row["game_id"])] = kickoff.isoformat() if kickoff is not None else None
    return output
