"""Persistent verified NFL market snapshots for forward CLV and grading evidence."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from .contracts import require_columns
from .espn_market import ESPNTwoWayMarket

NFL_SCHEDULE_TIMEZONE = ZoneInfo("America/New_York")

SNAPSHOT_FIELDS = (
    "captured_at",
    "kickoff",
    "season",
    "week",
    "game_id",
    "home_team",
    "away_team",
    "market_type",
    "provider",
    "book",
    "source_event_id",
    "first_side",
    "first_line",
    "first_american_odds",
    "second_side",
    "second_line",
    "second_american_odds",
)


def _kickoff(row: dict[str, object]) -> str:
    """Return nflverse schedule kickoff as an explicit UTC timestamp.

    nflverse ``gameday``/``gametime`` schedule values follow the NFL schedule's
    Eastern-time convention. Persisting a UTC value prevents a runner's local timezone
    from changing which market observations qualify as pre-kickoff evidence.
    """

    day = row.get("gameday")
    time = row.get("gametime")
    if day is None or time in {None, ""}:
        return ""
    try:
        local = datetime.fromisoformat(f"{day}T{time}")
    except ValueError:
        return ""
    if local.tzinfo is None:
        local = local.replace(tzinfo=NFL_SCHEDULE_TIMEZONE)
    return local.astimezone(UTC).isoformat()


def _signature(row: dict[str, object]) -> tuple[str, ...]:
    return tuple(str(row.get(field, "")) for field in SNAPSHOT_FIELDS)


def append_market_snapshots(
    markets: Sequence[ESPNTwoWayMarket],
    targets: pl.DataFrame,
    *,
    path: str | Path = "history/market_snapshots.csv",
) -> dict[str, object]:
    """Append only new verified market snapshots, preserving provider/timestamp provenance."""

    require_columns(
        targets,
        {"season", "week", "game_id", "home_team", "away_team", "gameday"},
        "line_capture_targets",
    )
    target_map = {str(row["game_id"]): row for row in targets.iter_rows(named=True)}
    source = Path(path)
    source.parent.mkdir(parents=True, exist_ok=True)

    old_rows: list[dict[str, object]] = []
    if source.exists() and source.stat().st_size:
        try:
            with source.open(newline="") as handle:
                old_rows = list(csv.DictReader(handle))
        except OSError:
            old_rows = []
    known = {_signature(row) for row in old_rows}

    new_rows: list[dict[str, object]] = []
    for market in markets:
        target = target_map.get(market.game_id)
        if target is None:
            continue
        row = {
            "captured_at": market.captured_at.astimezone(UTC).isoformat(),
            "kickoff": _kickoff(target),
            "season": int(target["season"]),
            "week": int(target["week"]),
            "game_id": market.game_id,
            "home_team": target["home_team"],
            "away_team": target["away_team"],
            "market_type": market.market_type,
            "provider": market.provider,
            "book": market.book,
            "source_event_id": market.source_event_id,
            "first_side": market.first_side,
            "first_line": market.first_line,
            "first_american_odds": market.first_american_odds,
            "second_side": market.second_side,
            "second_line": market.second_line,
            "second_american_odds": market.second_american_odds,
        }
        signature = _signature(row)
        if signature not in known:
            known.add(signature)
            new_rows.append(row)

    combined = [*old_rows, *new_rows]
    if new_rows or not source.exists():
        with source.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(SNAPSHOT_FIELDS))
            writer.writeheader()
            writer.writerows(combined)
    return {
        "path": str(source),
        "captured_markets": len(markets),
        "appended_rows": len(new_rows),
        "total_rows": len(combined),
    }


def load_market_snapshots(path: str | Path = "history/market_snapshots.csv") -> pl.DataFrame:
    source = Path(path)
    if not source.exists() or not source.stat().st_size:
        return pl.DataFrame()
    return pl.read_csv(source, try_parse_dates=False)


def latest_pre_kickoff_snapshot(
    snapshots: pl.DataFrame,
    *,
    game_id: str,
    market_type: str,
    book: str,
    decision_at: str | datetime,
) -> dict[str, object] | None:
    """Return the latest same-book snapshot after decision time but before kickoff."""

    if snapshots.is_empty():
        return None
    require_columns(
        snapshots,
        {"game_id", "market_type", "book", "captured_at", "kickoff"},
        "market_snapshots",
    )
    decision = (
        decision_at
        if isinstance(decision_at, datetime)
        else datetime.fromisoformat(str(decision_at).replace("Z", "+00:00"))
    )
    if decision.tzinfo is None:
        decision = decision.replace(tzinfo=UTC)
    candidates: list[tuple[datetime, dict[str, object]]] = []
    for row in snapshots.filter(
        (pl.col("game_id").cast(pl.String) == str(game_id))
        & (pl.col("market_type") == market_type)
        & (pl.col("book") == book)
    ).iter_rows(named=True):
        try:
            captured = datetime.fromisoformat(str(row["captured_at"]).replace("Z", "+00:00"))
            kickoff = datetime.fromisoformat(str(row["kickoff"]).replace("Z", "+00:00"))
        except ValueError:
            continue
        if captured.tzinfo is None:
            captured = captured.replace(tzinfo=UTC)
        if kickoff.tzinfo is None:
            continue
        if decision < captured < kickoff:
            candidates.append((captured, row))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]
