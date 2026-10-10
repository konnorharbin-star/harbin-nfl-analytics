"""Persistent verified NFL market snapshots for forward CLV and grading evidence."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from .contracts import require_columns
from .espn_market import ESPNTwoWayMarket
from .schedule_market import is_research_only_market

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
    return tuple(
        str(row.get(field) if row.get(field) is not None else "") for field in SNAPSHOT_FIELDS
    )


def _immutable_event(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as temp:
        json.dump(record, temp, sort_keys=True, indent=2, allow_nan=False)
        temp.write("\n")
        temp.flush()
        os.fsync(temp.fileno())
        name = temp.name
    try:
        os.link(name, path)
    except FileExistsError:
        pass
    finally:
        os.unlink(name)


def append_market_snapshots(
    markets: Sequence[ESPNTwoWayMarket],
    targets: pl.DataFrame,
    *,
    path: str | Path = "history/market_snapshots.csv",
) -> dict[str, object]:
    """Append observed book quotes; archive origin provenance separately and fail closed."""

    require_columns(
        targets,
        {"season", "week", "game_id", "home_team", "away_team", "gameday"},
        "line_capture_targets",
    )
    target_map = {str(row["game_id"]): row for row in targets.iter_rows(named=True)}
    source = Path(path)
    source.parent.mkdir(parents=True, exist_ok=True)

    events = source.with_name(source.name + ".events")
    new_rows = []
    skipped_research_only = 0
    eligible_markets = 0
    verified_origins = 0
    # Lock the actual file across readers/writers; append never rewrites its prefix.
    with source.open("a+", newline="") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        reader = csv.DictReader(handle)
        if reader.fieldnames and tuple(reader.fieldnames) != SNAPSHOT_FIELDS:
            raise ValueError("Unexpected legacy market history schema")
        old_rows = list(reader)
        known = {_signature(row) for row in old_rows}
        for market in markets:
            if is_research_only_market(market):
                skipped_research_only += 1
                continue
            eligible_markets += 1
            target = target_map.get(market.game_id)
            if target is None:
                continue
            if market.captured_at.tzinfo is None:
                raise ValueError("Market capture requires an aware timestamp")
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
            # Even a replay of an old CSV observation may now carry provenance.
            # The event is written first: a failed append can be safely retried.
            archived_at = datetime.now(UTC)
            origin = market.source_quote_at
            origin_valid = False
            if origin is not None and origin.tzinfo is not None and row["kickoff"]:
                kickoff = datetime.fromisoformat(row["kickoff"])
                origin_valid = (
                    market.source_quote_time_verified is True
                    and origin <= market.captured_at <= archived_at < kickoff
                )
            verified_origins += int(origin_valid)
            event = {
                "schema": 1,
                "archived_at": archived_at.isoformat(),
                "original_market_row": row,
                "source_quote_at": origin.isoformat() if origin is not None else None,
                "source_quote_time_verified": origin_valid,
                "reported_source_quote_time_verified": market.source_quote_time_verified,
                "original_request_url": None,
                "origin_time_usable": origin_valid,
                "source_quote_age_seconds": (market.captured_at - origin).total_seconds()
                if origin is not None and origin.tzinfo is not None
                else None,
                "capture_is_book_update_time": False,
            }
            identity = json.dumps(signature, separators=(",", ":"))
            digest = hashlib.sha256(identity.encode()).hexdigest()
            _immutable_event(events / (digest + ".json"), event)
            if signature not in known:
                known.add(signature)
                new_rows.append(row)
        handle.seek(0, os.SEEK_END)
        writer = csv.DictWriter(handle, fieldnames=list(SNAPSHOT_FIELDS))
        if not handle.tell():
            writer.writeheader()
        writer.writerows(new_rows)
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return {
        "path": str(source),
        "provenance_events": str(events),
        "captured_markets": len(markets),
        "eligible_verified_markets": eligible_markets,
        "origin_time_usable_markets": verified_origins,
        "skipped_research_only": skipped_research_only,
        "appended_rows": len(new_rows),
        "total_rows": len(old_rows) + len(new_rows),
    }


def load_market_snapshots(path: str | Path = "history/market_snapshots.csv") -> pl.DataFrame:
    source = Path(path)
    if not source.exists() or not source.stat().st_size:
        return pl.DataFrame()
    frame = pl.read_csv(source, try_parse_dates=False)
    events = source.with_name(source.name + ".events")
    provenance = []
    with source.open(newline="") as handle:
        for row in csv.DictReader(handle):
            identity = json.dumps(_signature(row), separators=(",", ":"))
            path = events / (hashlib.sha256(identity.encode()).hexdigest() + ".json")
            entry = json.loads(path.read_text()) if path.exists() else {}
            provenance.append(
                {
                    "source_quote_at": entry.get("source_quote_at"),
                    "source_quote_time_verified": entry.get("source_quote_time_verified", False),
                    "origin_time_usable": entry.get("origin_time_usable", False),
                }
            )
    if provenance:
        extras = pl.DataFrame(
            provenance,
            schema={
                "source_quote_at": pl.String,
                "source_quote_time_verified": pl.Boolean,
                "origin_time_usable": pl.Boolean,
            },
        )
        frame = frame.hstack(extras)
    return frame


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
