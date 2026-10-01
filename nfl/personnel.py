"""NFL roster/depth-chart availability for current context.

Modern nflverse depth charts include a load timestamp (``dt``); older depth-chart
schemas are week based. This module respects whichever point-in-time key is available
and keeps personnel state downstream of the independent fair-score model.
"""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from .contracts import DataContractError, require_columns


def _column(frame: pl.DataFrame, *names: str) -> str | None:
    lookup = {name.lower(): name for name in frame.columns}
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None


def _text(value: object) -> str:
    return "" if value is None else str(value).strip()


def _as_utc(value: datetime | str | None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _player_key(row: dict[str, object], id_column: str | None, name_column: str | None) -> str:
    if id_column:
        value = _text(row.get(id_column))
        if value:
            return f"id:{value}"
    if name_column:
        value = _text(row.get(name_column)).lower()
        if value:
            return f"name:{value}"
    return ""


def _active_from_status(value: object) -> tuple[bool, bool]:
    status = _text(value).lower()
    if not status:
        return True, False
    if any(token in status for token in ("inactive", "injured reserve", "reserve/injured", "suspended")):
        return False, True
    if status in {"out", "ir", "pup"}:
        return False, True
    if any(token in status for token in ("active", "available", "healthy")):
        return True, True
    return True, False


def normalize_depth_charts(
    frame: pl.DataFrame,
    *,
    season: int,
    week: int,
    as_of: datetime | str | None = None,
) -> pl.DataFrame:
    """Normalize modern timestamped or legacy weekly depth charts."""

    if frame.is_empty():
        return pl.DataFrame()
    require_columns(frame, {"season", "team"}, "depth_charts")
    cutoff = _as_utc(as_of)

    id_column = _column(frame, "gsis_id", "espn_id", "player_id")
    name_column = _column(frame, "player_name", "full_name", "football_name")
    position_column = _column(frame, "pos_abb", "position", "depth_position")
    rank_column = _column(frame, "pos_rank", "depth_team")
    timestamp_column = _column(frame, "dt", "date_modified", "updated_at")
    week_column = _column(frame, "week")

    filtered = frame.filter(pl.col("season") == season)
    rows: list[dict[str, object]] = []
    for source in filtered.iter_rows(named=True):
        captured: datetime | None = None
        if timestamp_column and source.get(timestamp_column) not in {None, ""}:
            try:
                captured = _as_utc(str(source[timestamp_column]))
            except ValueError:
                captured = None
        if captured is not None and captured > cutoff:
            continue
        row_week: int | None = None
        if week_column and source.get(week_column) not in {None, ""}:
            try:
                row_week = int(float(source[week_column]))
            except (TypeError, ValueError):
                row_week = None
            if row_week is not None and row_week > week:
                continue
        key = _player_key(source, id_column, name_column)
        if not key:
            continue
        rank: int | None = None
        if rank_column and source.get(rank_column) not in {None, ""}:
            try:
                rank = int(float(source[rank_column]))
            except (TypeError, ValueError):
                rank = None
        rows.append(
            {
                "team": str(source["team"]),
                "player_key": key,
                "gsis_id": _text(source.get(id_column)) if id_column else "",
                "player_name": _text(source.get(name_column)) if name_column else "",
                "position": _text(source.get(position_column)).upper() if position_column else "",
                "depth_rank": rank,
                "depth_week": row_week,
                "depth_captured_at": captured.isoformat() if captured is not None else None,
            }
        )
    if not rows:
        return pl.DataFrame()

    normalized = pl.DataFrame(rows)
    # Modern feeds publish complete timestamped snapshots. Retain only the most recent
    # admissible snapshot for each team when timestamps are present.
    if normalized.get_column("depth_captured_at").drop_nulls().len():
        latest: dict[str, str] = {}
        for row in normalized.filter(pl.col("depth_captured_at").is_not_null()).iter_rows(named=True):
            team = str(row["team"])
            stamp = str(row["depth_captured_at"])
            if team not in latest or stamp > latest[team]:
                latest[team] = stamp
        normalized = normalized.filter(
            pl.col("depth_captured_at").is_null()
            | pl.struct(["team", "depth_captured_at"]).map_elements(
                lambda value: latest.get(str(value["team"])) == value["depth_captured_at"],
                return_dtype=pl.Boolean,
            )
        )
    return normalized.unique(
        subset=["team", "player_key"],
        keep="last",
        maintain_order=True,
    ).sort(["team", "position", "depth_rank", "player_name"])


def normalize_rosters(
    frame: pl.DataFrame,
    *,
    season: int,
    week: int,
) -> pl.DataFrame:
    """Return latest admissible weekly roster state per team/player."""

    if frame.is_empty():
        return pl.DataFrame()
    require_columns(frame, {"season", "team", "position"}, "rosters_weekly")
    id_column = _column(frame, "gsis_id", "espn_id", "player_id")
    name_column = _column(frame, "full_name", "player_name")
    status_column = _column(frame, "status_description_abbr", "status", "roster_status")
    week_column = _column(frame, "week")

    filtered = frame.filter(pl.col("season") == season)
    if week_column:
        filtered = filtered.filter(
            pl.col(week_column).cast(pl.Int64, strict=False).is_null()
            | (pl.col(week_column).cast(pl.Int64, strict=False) <= week)
        )
    rows: list[dict[str, object]] = []
    for source in filtered.iter_rows(named=True):
        key = _player_key(source, id_column, name_column)
        if not key:
            continue
        active, status_known = _active_from_status(
            source.get(status_column) if status_column else None
        )
        row_week: int | None = None
        if week_column and source.get(week_column) not in {None, ""}:
            try:
                row_week = int(float(source[week_column]))
            except (TypeError, ValueError):
                row_week = None
        rows.append(
            {
                "team": str(source["team"]),
                "player_key": key,
                "gsis_id": _text(source.get(id_column)) if id_column else "",
                "player_name": _text(source.get(name_column)) if name_column else "",
                "position": _text(source.get("position")).upper(),
                "roster_week": row_week,
                "roster_status": _text(source.get(status_column)) if status_column else "",
                "active": active,
                "status_known": status_known,
            }
        )
    if not rows:
        return pl.DataFrame()
    normalized = pl.DataFrame(rows).sort(
        ["team", "player_key", "roster_week"],
        nulls_last=False,
    )
    return normalized.unique(
        subset=["team", "player_key"],
        keep="last",
        maintain_order=True,
    ).sort(["team", "position", "player_name"])


def _injury_lookup(injuries: pl.DataFrame) -> dict[tuple[str, str], float]:
    if injuries.is_empty():
        return {}
    require_columns(injuries, {"team", "player_key", "severity"}, "normalized_injuries")
    return {
        (str(row["team"]), str(row["player_key"])): float(row["severity"])
        for row in injuries.iter_rows(named=True)
    }


def summarize_team_personnel(
    depth: pl.DataFrame,
    rosters: pl.DataFrame,
    injuries: pl.DataFrame,
) -> pl.DataFrame:
    """Combine depth, roster, and injury state into one confidence-only team summary."""

    injury = _injury_lookup(injuries)
    teams = set()
    for frame in (depth, rosters, injuries):
        if not frame.is_empty() and "team" in frame.columns:
            teams.update(str(value) for value in frame.get_column("team").unique().to_list())
    if not teams:
        return pl.DataFrame()

    rows: list[dict[str, object]] = []
    for team in sorted(teams):
        team_depth = depth.filter(pl.col("team") == team) if not depth.is_empty() else pl.DataFrame()
        team_roster = (
            rosters.filter(pl.col("team") == team) if not rosters.is_empty() else pl.DataFrame()
        )
        starters = (
            team_depth.filter(pl.col("depth_rank").is_null() | (pl.col("depth_rank") <= 1))
            if not team_depth.is_empty()
            else pl.DataFrame()
        )
        starter_risk = 0.0
        group_risk = {"ol": 0.0, "skill": 0.0, "defense": 0.0, "qb": 0.0}
        if not starters.is_empty():
            for player in starters.iter_rows(named=True):
                severity = injury.get((team, str(player["player_key"])), 0.0)
                starter_risk += severity
                position = str(player.get("position") or "").upper()
                if position == "QB":
                    group_risk["qb"] = max(group_risk["qb"], severity)
                elif position in {"LT", "RT", "OT", "G", "LG", "RG", "OG", "C", "OL"}:
                    group_risk["ol"] += severity
                elif position in {"WR", "TE", "RB", "FB"}:
                    group_risk["skill"] += severity
                else:
                    group_risk["defense"] += severity

        active_qbs = 0
        roster_status_known = False
        inactive_share: float | None = None
        if not team_roster.is_empty():
            roster_status_known = bool(team_roster.get_column("status_known").any())
            qb_rows = team_roster.filter(pl.col("position") == "QB")
            active_qbs = qb_rows.filter(pl.col("active")).height
            if roster_status_known:
                inactive = team_roster.filter(~pl.col("active")).height
                inactive_share = inactive / team_roster.height

        rows.append(
            {
                "team": team,
                "depth_players": team_depth.height,
                "starter_slots": starters.height,
                "starter_injury_risk": min(1.0, starter_risk / max(1, starters.height / 4)),
                "ol_injury_risk": min(1.0, group_risk["ol"] / 3.0),
                "skill_injury_risk": min(1.0, group_risk["skill"] / 3.0),
                "defense_injury_risk": min(1.0, group_risk["defense"] / 6.0),
                "depth_qb_injury_risk": min(1.0, group_risk["qb"]),
                "roster_count": team_roster.height,
                "roster_status_known": roster_status_known,
                "roster_inactive_share": inactive_share,
                "active_qb_count": active_qbs,
                "depth_source_available": team_depth.height > 0,
                "roster_source_available": team_roster.height > 0,
            }
        )
    return pl.DataFrame(rows).sort("team")
