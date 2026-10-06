"""Cheap scheduler guard for targeted NFL pregame refresh checkpoints."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CHECKPOINT_MINUTES = (120, 75, 20)
POLL_INTERVAL_MINUTES = 15


def _parse_utc(value: object) -> datetime | None:
    if value in {None, ""}:
        return None
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=UTC)
    return stamp.astimezone(UTC)


def refresh_checkpoint(
    report: dict[str, Any],
    *,
    now: datetime | None = None,
    checkpoints: tuple[int, ...] = CHECKPOINT_MINUTES,
    poll_interval_minutes: int = POLL_INTERVAL_MINUTES,
) -> dict[str, object]:
    """Return whether the latest published slate is due for a pregame refresh.

    Each checkpoint owns the preceding polling interval. With a 15-minute cron,
    120 means (105, 120], 75 means (60, 75], and 20 means (5, 20].
    """

    reference = now or datetime.now(UTC)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=UTC)
    reference = reference.astimezone(UTC)
    interval = max(1, int(poll_interval_minutes))

    meta = report.get("meta") if isinstance(report.get("meta"), dict) else {}
    games = report.get("games") if isinstance(report.get("games"), list) else []
    due: list[dict[str, object]] = []

    for game in games:
        if not isinstance(game, dict):
            continue
        kickoff = _parse_utc(game.get("kickoff"))
        if kickoff is None:
            continue
        minutes = (kickoff - reference).total_seconds() / 60.0
        if minutes <= 0:
            continue
        for checkpoint in checkpoints:
            target = float(checkpoint)
            if target - interval < minutes <= target:
                due.append(
                    {
                        "game_id": str(game.get("game_id") or ""),
                        "kickoff": kickoff.isoformat(),
                        "minutes_to_kickoff": round(minutes, 3),
                        "checkpoint_minutes": int(checkpoint),
                    }
                )
                break

    if not due:
        return {
            "run": False,
            "season": meta.get("season"),
            "week": meta.get("week"),
            "checkpoint_minutes": None,
            "due_games": [],
        }

    due.sort(
        key=lambda item: (
            int(item["checkpoint_minutes"]),
            float(item["minutes_to_kickoff"]),
            str(item["game_id"]),
        )
    )
    return {
        "run": True,
        "season": meta.get("season"),
        "week": meta.get("week"),
        "checkpoint_minutes": due[0]["checkpoint_minutes"],
        "due_games": due,
    }


def load_refresh_checkpoint(
    path: str | Path,
    *,
    now: datetime | None = None,
) -> dict[str, object]:
    source = Path(path)
    if not source.exists():
        return {
            "run": False,
            "season": None,
            "week": None,
            "checkpoint_minutes": None,
            "due_games": [],
            "reason": "current model report is unavailable",
        }
    try:
        report = json.loads(source.read_text())
    except (OSError, json.JSONDecodeError):
        return {
            "run": False,
            "season": None,
            "week": None,
            "checkpoint_minutes": None,
            "due_games": [],
            "reason": "current model report is unreadable",
        }
    if not isinstance(report, dict):
        return {
            "run": False,
            "season": None,
            "week": None,
            "checkpoint_minutes": None,
            "due_games": [],
            "reason": "current model report is invalid",
        }
    return refresh_checkpoint(report, now=now)
