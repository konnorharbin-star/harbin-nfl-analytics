"""Read-only NFL publication watchdog for the actual pregame refresh windows.

A run that commits nothing after a network failure must not be mistaken for a
fresh betting model merely because the previous weekly PNG remains on GitHub.
This is an operational alert, not a market/execution/profitability assertion.
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

MAX_FUTURE_SKEW_MINUTES = 5
CRITICAL_WINDOW_MINUTES = 120


def _clock(value: object) -> datetime | None:
    if isinstance(value, datetime):
        stamp = value
    elif isinstance(value, str):
        try:
            stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    return stamp.astimezone(UTC) if stamp.tzinfo is not None else None


def evaluate_publication_watchdog(
    report: dict[str, Any],
    *,
    now: datetime,
    manifest: dict[str, object],
) -> dict[str, object]:
    """Check verified file state and recent model generation near NFL kickoff."""
    if now.tzinfo is None:
        raise ValueError("watchdog now must have a timezone")
    reference = now.astimezone(UTC)
    reasons = []
    meta = report.get("meta")
    publication = report.get("publication")
    if not isinstance(meta, dict) or not isinstance(publication, dict):
        reasons.append("NO_CANONICAL_MODEL_OR_PUBLICATION")
        meta = {}
    if manifest.get("status") != "PASS":
        reasons.append(
            "PUBLICATION_MANIFEST_INVALID: "
            + str(manifest.get("reason") or "missing manifest")
        )
    timestamp = _clock(meta.get("generated_at"))
    if timestamp is None:
        reasons.append("MODEL_GENERATED_AT_UNKNOWN_OR_UNTZONED")
    elif (timestamp - reference).total_seconds() > MAX_FUTURE_SKEW_MINUTES * 60:
        reasons.append("MODEL_GENERATED_IN_FUTURE")

    kickoff_by_game: dict[str, datetime] = {}
    games = report.get("games")
    for row in games if isinstance(games, list) else []:
        if not isinstance(row, dict):
            continue
        identity = str(row.get("game_id") or "")
        kickoff = _clock(row.get("kickoff"))
        if identity and kickoff is not None:
            old = kickoff_by_game.get(identity)
            if old is not None and kickoff != old:
                reasons.append(f"GAME_KICKOFF_INCONSISTENT: {identity}")
            kickoff_by_game[identity] = kickoff
    imminent = sorted(
        (
            (identity, (kickoff - reference).total_seconds() / 60)
            for identity, kickoff in kickoff_by_game.items()
            if 0 < (kickoff - reference).total_seconds() / 60 <= CRITICAL_WINDOW_MINUTES
        ),
        key=lambda pair: pair[1],
    )
    age = (
        (reference - timestamp).total_seconds() / 60
        if timestamp is not None else None
    )
    remaining = imminent[0][1] if imminent else None
    allowed_age = None
    if remaining is not None:
        # 120/75/20 minute model refresh checkpoints already exist.
        # Check AFTER the checkpoint should have reached GitHub, with a
        # conservative tolerance for GitHub-hosted runner delays.
        if remaining <= 15:
            allowed_age = 35
        elif remaining <= 60:
            allowed_age = 85
        else:
            allowed_age = 130
        if age is None or age > allowed_age:
            reasons.append(
                f"STALE_MODEL_NEAR_KICKOFF: age={age!r}m "
                f"max={allowed_age}m remaining={remaining:.1f}m"
            )

    release_state = str(report.get("release_state") or "UNKNOWN")
    portfolio = report.get("portfolio")
    try:
        approved = float(
            portfolio.get("approved_units", 0)
            if isinstance(portfolio, dict) else 0
        )
    except (TypeError, ValueError):
        approved = float("nan")
    if release_state != "PRODUCTION" and (not 0 <= approved <= 0):
        reasons.append("NONPRODUCTION_HAS_APPROVED_STAKE_OR_INVALID_PORTFOLIO")
    if reasons:
        status = "FAIL"
    elif imminent:
        status = "PASS_PREGAME_FRESH"
    else:
        status = "PASS_OUTSIDE_PREGAME_WINDOW"
    return {
        "status": status,
        "checked_at": reference.isoformat(),
        "model_generated_at": timestamp.isoformat() if timestamp else None,
        "model_age_minutes": round(age, 3) if age is not None else None,
        "critical_window_minutes": CRITICAL_WINDOW_MINUTES,
        "nearest_kickoff_minutes": round(remaining, 3) if remaining is not None else None,
        "max_model_age_minutes": allowed_age,
        "imminent_games": [identity for identity, _ in imminent],
        "verified_file_count": manifest.get("file_count", 0),
        "release_state": release_state,
        "approved_units": approved,
        "blocking_reasons": reasons,
        "watchdog_changes_bets_or_scores": False,
        "warning": (
            "GitHub schedule delivery can be delayed; this detects stale model "
            "publication, not sportsbook line availability, fills or positive EV."
        ),
    }
