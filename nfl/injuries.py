"""Point-in-time NFL injury normalization and team availability risk.

The injury feed is current/pregame context only. It never enters the fair-score fit.
Current-week reports supersede prior-week designations; older reports remain diagnostic
only when the current-week feed has not arrived and can never create a fresh betting
signal.
"""

from __future__ import annotations

from datetime import UTC, datetime
from math import isfinite

import polars as pl

from .contracts import DataContractError, require_columns

MAX_INJURY_REPORT_AGE_DAYS = 7.0
MIN_INJURY_REPORT_TIMESTAMP_COVERAGE = 0.95
REPORT_TIMESTAMP_ALIASES = (
    "date_modified", "reported_at", "report_datetime", "updated_at",
    "last_updated", "timestamp", "report_date", "date",
)


def _source_report_timestamp(value: object) -> datetime | None:
    """Only a source-origin, timezone-aware report time proves freshness.

    Date-only strings and naive timestamps cannot establish an as-of time.
    A collector's own retrieval timestamp is not an injury-report timestamp.
    """
    if isinstance(value, datetime):
        stamp = value
    elif isinstance(value, str) and ("T" in value or " " in value):
        try:
            stamp = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if stamp.tzinfo is None:
        return None
    return stamp.astimezone(UTC)



def _column(frame: pl.DataFrame, *names: str) -> str | None:
    lookup = {name.lower(): name for name in frame.columns}
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None


def _text(value: object) -> str:
    return "" if value is None else str(value).strip()


def _player_key(
    row: dict[str, object],
    id_column: str | None,
    name_column: str | None,
) -> str:
    if id_column:
        value = _text(row.get(id_column))
        if value:
            return f"id:{value}"
    if name_column:
        value = _text(row.get(name_column)).lower()
        if value:
            return f"name:{value}"
    return ""


def injury_severity(status: object, practice_status: object = None) -> float:
    """Respect definitive game status over practice participation.

    Injury report designation and practice participation describe different
    things. A player ruled OUT may still have practiced fully earlier; a
    QUESTIONABLE player can have full participation. Also, "inactive" must
    never match "active" through a substring search.
    """

    designation = _text(status).strip().lower()
    practice = _text(practice_status).strip().lower()
    if any(
        token in designation
        for token in (
            "inactive",
            "injured reserve",
            "reserve/injured",
            "out for season",
            "season-ending",
            "suspended",
        )
    ):
        return 1.0
    if designation == "out" or designation.startswith("out ") or designation.endswith(" out"):
        return 1.0
    if "doubt" in designation:
        return 0.80
    if "question" in designation:
        return 0.45
    if "prob" in designation:
        return 0.10
    if designation in {"active", "available", "cleared", "healthy"}:
        return 0.0
    # Only use practice participation when game availability is unresolved.
    # Practice activity must not override a definitive game-day designation.
    if "did not participate" in practice or "dnp" in practice:
        return 0.40
    if "limited" in practice:
        return 0.25
    if "day-to-day" in designation or "day to day" in designation:
        return 0.25
    if "full participation" in practice or practice == "full":
        return 0.0
    return 0.0


def position_importance(position: object) -> float:
    """Relative availability importance used only for confidence/risk aggregation."""

    pos = _text(position).upper()
    if pos == "QB":
        return 1.50
    if pos in {"LT", "RT", "OT", "G", "LG", "RG", "OG", "C", "OL"}:
        return 1.15
    if pos in {"WR", "TE", "RB", "FB"}:
        return 1.00
    if pos in {
        "CB",
        "S",
        "FS",
        "SS",
        "DB",
        "EDGE",
        "DE",
        "DT",
        "NT",
        "LB",
        "ILB",
        "OLB",
    }:
        return 1.00
    if pos in {"K", "P", "LS"}:
        return 0.60
    return 0.85


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


def _admissible_rows(
    frame: pl.DataFrame,
    *,
    season: int,
    week: int,
    cutoff: datetime,
) -> tuple[list[dict[str, object]], str | None]:
    filtered = frame.filter(
        (pl.col("season") == season)
        & (pl.col("week").cast(pl.Int64, strict=False) <= week)
    )
    available = [_column(frame, field) for field in REPORT_TIMESTAMP_ALIASES]
    candidates = [name for name in available if name is not None]
    source_rows = filtered.to_dicts()

    # Some feed versions include an all-null legacy date_modified column
    # alongside a populated reported_at field. Select source-provenance time
    # with greatest coverage, never a synthetic fetch timestamp.
    modified_column = (
        max(candidates, key=lambda name: sum(
            _source_report_timestamp(item.get(name)) is not None
            for item in source_rows if int(item.get("week") or 0) == week
        ))
        if candidates else None
    )
    rows: list[dict[str, object]] = []
    for source in source_rows:
        modified = (
            _source_report_timestamp(source.get(modified_column))
            if modified_column else None
        )
        if modified is not None and modified > cutoff:
            continue
        item = dict(source)
        item["_reported_at"] = modified
        rows.append(item)
    return rows, modified_column


def injury_feed_freshness(
    frame: pl.DataFrame,
    *,
    season: int,
    week: int,
    as_of: datetime | str | None = None,
) -> dict[str, object]:
    """Classify whether the league injury feed is current enough for betting."""

    if frame.is_empty():
        return {
            "status": "UNKNOWN",
            "current_week_rows": 0,
            "latest_reported_at": None,
            "latest_age_days": None,
            "reason": "injury feed is empty",
        }
    require_columns(frame, {"season", "week", "team"}, "injuries")
    cutoff = _as_utc(as_of)
    rows, modified_column = _admissible_rows(
        frame,
        season=season,
        week=week,
        cutoff=cutoff,
    )
    current = [row for row in rows if int(row["week"]) == week]
    if not current:
        return {
            "status": "STALE" if rows else "UNKNOWN",
            "current_week_rows": 0,
            "latest_reported_at": None,
            "latest_age_days": None,
            "reason": (
                f"no admissible injury reports for target week {week}"
                if rows
                else "no admissible injury reports are available"
            ),
        }
    if modified_column is None:
        return {
            "status": "UNKNOWN",
            "current_week_rows": len(current),
            "latest_reported_at": None,
            "latest_age_days": None,
            "reason": "current-week injury rows lack a timestamp column",
        }

    stamps = [
        row["_reported_at"]
        for row in current
        if isinstance(row.get("_reported_at"), datetime)
    ]
    if not stamps:
        return {
            "status": "UNKNOWN",
            "current_week_rows": len(current),
            "latest_reported_at": None,
            "latest_age_days": None,
            "reason": "current-week injury timestamps are missing or invalid",
        }
    coverage = len(stamps) / len(current)
    latest = max(stamps)
    age_days = max(0.0, (cutoff - latest).total_seconds() / 86400.0)
    # One timestamped player cannot certify hundreds of undated rows.
    if coverage < MIN_INJURY_REPORT_TIMESTAMP_COVERAGE:
        return {
            "status": "UNKNOWN",
            "current_week_rows": len(current),
            "timestamped_current_week_rows": len(stamps),
            "timestamp_coverage": round(coverage, 4),
            "latest_reported_at": latest.isoformat(),
            "latest_age_days": round(age_days, 4),
            "reason": (
                f"only {len(stamps)}/{len(current)} current-week injury "
                "rows have source-origin report timestamps"
            ),
        }
    status = "FRESH" if age_days <= MAX_INJURY_REPORT_AGE_DAYS else "STALE"
    return {
        "status": status,
        "current_week_rows": len(current),
        "latest_reported_at": latest.isoformat(),
        "latest_age_days": round(age_days, 4),
        "reason": (
            "current-week injury feed is timestamp-fresh"
            if status == "FRESH"
            else (
                "latest current-week injury report is "
                f"{age_days:.1f} days old"
            )
        ),
    }


def normalize_injuries(
    frame: pl.DataFrame,
    *,
    season: int,
    week: int,
    as_of: datetime | str | None = None,
) -> pl.DataFrame:
    """Return point-in-time injury state without carrying stale weekly labels forward."""

    if frame.is_empty():
        return pl.DataFrame()
    require_columns(frame, {"season", "week", "team"}, "injuries")
    if week < 1:
        raise DataContractError("week must be >= 1")

    cutoff = _as_utc(as_of)
    id_column = _column(frame, "gsis_id", "player_id", "athlete_id")
    name_column = _column(frame, "full_name", "player_name", "athlete_name")
    position_column = _column(
        frame,
        "position",
        "position_abbr",
        "position_abbreviation",
    )
    report_status = _column(frame, "report_status", "status")
    practice_status = _column(frame, "practice_status")
    primary = _column(frame, "report_primary_injury", "primary_injury")
    secondary = _column(frame, "report_secondary_injury", "secondary_injury")

    admissible, _ = _admissible_rows(
        frame,
        season=season,
        week=week,
        cutoff=cutoff,
    )
    current_week_available = any(int(row["week"]) == week for row in admissible)
    selected = (
        [row for row in admissible if int(row["week"]) == week]
        if current_week_available
        else admissible
    )

    rows: list[dict[str, object]] = []
    for source in selected:
        modified = source.get("_reported_at")
        modified_at = modified if isinstance(modified, datetime) else None
        key = _player_key(source, id_column, name_column)
        if not key:
            continue
        status = source.get(report_status) if report_status else None
        practice = source.get(practice_status) if practice_status else None
        severity = injury_severity(status, practice)
        position = source.get(position_column) if position_column else None
        report_week = int(source["week"])
        age_days = (
            (cutoff - modified_at).total_seconds() / 86400.0
            if modified_at is not None
            else None
        )
        if report_week < week:
            freshness = "STALE"
        elif age_days is None:
            freshness = "UNKNOWN"
        elif age_days <= MAX_INJURY_REPORT_AGE_DAYS:
            freshness = "FRESH"
        else:
            freshness = "STALE"
        injury_text = " / ".join(
            value
            for value in (
                _text(source.get(primary)) if primary else "",
                _text(source.get(secondary)) if secondary else "",
            )
            if value
        )
        rows.append(
            {
                "season": season,
                "week": report_week,
                "team": str(source["team"]),
                "player_key": key,
                "gsis_id": _text(source.get(id_column)) if id_column else "",
                "player_name": _text(source.get(name_column)) if name_column else "",
                "position": _text(position).upper(),
                "report_status": _text(status),
                "practice_status": _text(practice),
                "injury": injury_text,
                "severity": severity,
                "weighted_severity": severity * position_importance(position),
                "reported_at": (
                    modified_at.isoformat() if modified_at is not None else None
                ),
                "report_age_days": age_days,
                "report_week_gap": week - report_week,
                "freshness_status": freshness,
            }
        )
    if not rows:
        return pl.DataFrame()

    normalized = pl.DataFrame(rows).sort(
        ["team", "player_key", "week", "reported_at"],
        nulls_last=False,
    )
    return normalized.unique(
        subset=["team", "player_key"],
        keep="last",
        maintain_order=True,
    ).sort(["team", "position", "player_name"])


def summarize_team_injuries(normalized: pl.DataFrame) -> pl.DataFrame:
    """Aggregate fresh injury state while retaining stale/unknown diagnostics."""

    if normalized.is_empty():
        return pl.DataFrame()
    require_columns(
        normalized,
        {
            "team",
            "position",
            "severity",
            "weighted_severity",
            "report_age_days",
            "freshness_status",
        },
        "normalized_injuries",
    )
    rows: list[dict[str, object]] = []
    for team_frame in normalized.partition_by("team", maintain_order=True):
        team = str(team_frame.get_column("team")[0])
        fresh = team_frame.filter(pl.col("freshness_status") == "FRESH")
        stale = team_frame.filter(pl.col("freshness_status") == "STALE")
        unknown = team_frame.filter(pl.col("freshness_status") == "UNKNOWN")
        active = fresh.filter(pl.col("severity") >= 0.05)
        qb = fresh.filter(pl.col("position") == "QB")
        ages = [
            float(value)
            for value in fresh.get_column("report_age_days").drop_nulls().to_list()
            if isfinite(float(value))
        ]
        raw_risk = (
            float(active.get_column("weighted_severity").sum())
            if active.height
            else 0.0
        )
        if stale.height:
            freshness = "STALE"
        elif unknown.height:
            freshness = "UNKNOWN"
        else:
            freshness = "FRESH"
        rows.append(
            {
                "team": team,
                "injury_reports": team_frame.height,
                "fresh_injury_reports": fresh.height,
                "stale_injury_reports": stale.height,
                "unknown_injury_reports": unknown.height,
                "injury_count": active.height,
                "injury_risk": min(1.0, raw_risk / 8.0),
                "qb_injury_risk": (
                    float(qb.get_column("severity").max()) if qb.height else 0.0
                ),
                "injury_report_age_days": min(ages) if ages else None,
                "injury_freshness_status": freshness,
                "injury_source_available": True,
            }
        )
    return pl.DataFrame(rows).sort("team")
