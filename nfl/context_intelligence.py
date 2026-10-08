"""NFL Step 6: immutable, provenance-aware game context and matchup risk audit.

Existing independent fair scores, market probabilities, release gates and
portfolio decisions remain authoritative. All numeric cutoffs below label
descriptive research stress, not validated predictive betting adjustments.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from .contracts import DataContractError

SPEC = "nfl_context_forensics_first_snapshot_v1"
SEASON = 2026
# Diagnostic-only definitions. DO NOT tune on settled results.
OL_STRESS = 0.20
SKILL_STRESS = 0.25
DEFENSE_STRESS = 0.25
SHORT_REST_DAYS = 6.0
LONG_TRAVEL_MILES = 1500.0
TIMEZONE_SHIFT_HOURS = 2.0
GAME_FIELDS = (
    "season", "week", "game_id", "kickoff", "home_team", "away_team",
    "evaluated_at", "source_context_as_of", "status", "source_status",
    "source_reasons", "matched_market_candidates",
    "qb_identity_both_known", "qb_decision_ready",
    "home_qb_change", "away_qb_change",
    "home_qb_id", "away_qb_id",
    "home_qb_confidence", "away_qb_confidence",
    "home_qb_source", "away_qb_source",
    "injury_feed_status", "injury_feed_reported_at",
    "home_injury_status", "away_injury_status",
    "home_depth_status", "away_depth_status",
    "home_roster_status", "away_roster_status",
    "personnel_fresh", "rest_travel_available",
    "weather_available", "weather_error",
    "home_injury_risk", "away_injury_risk",
    "home_starter_injury_risk", "away_starter_injury_risk",
    "home_ol_injury_risk", "away_ol_injury_risk",
    "home_skill_injury_risk", "away_skill_injury_risk",
    "home_defense_injury_risk", "away_defense_injury_risk",
    "home_rest_days", "away_rest_days",
    "home_travel_miles", "away_travel_miles",
    "away_timezone_shift_hours",
    "home_ol_vs_away_defense_availability",
    "away_ol_vs_home_defense_availability",
    "stacked_qb_ol", "stacked_rest_travel",
    "flags", "readiness_blockers", "research_staking_authorized",
)
FROZEN_FIELDS = ("spec", "captured_at", *GAME_FIELDS)


def _parse(value: object) -> datetime | None:
    if isinstance(value, datetime):
        result = value
    elif value is None or value == "":
        return None
    else:
        try:
            result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    return result.astimezone(UTC) if result.tzinfo is not None else None


def _number(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _yes(value: object) -> bool:
    return value is True or (
        isinstance(value, str) and value.strip().lower() == "true"
    )


def _text(value: object) -> str:
    return "" if value is None else str(value)


def _fresh(value: object) -> bool:
    return _text(value).upper() == "FRESH"


def _assert_clock(value: datetime | None, now: datetime) -> bool:
    return value is not None and value <= now


def _signature(row: Mapping[str, object]) -> tuple[object, ...]:
    """Same game's context must not depend on selected market or sportsbook."""
    important = (
        "season", "week", "home_team", "away_team", "kickoff",
        "context_as_of",
        "home_expected_qb_id", "away_expected_qb_id",
        "home_expected_qb_decision_ready", "away_expected_qb_decision_ready",
        "context_injuries_personnel_fresh",
        "home_injury_freshness_status", "away_injury_freshness_status",
        "home_ol_injury_risk", "away_ol_injury_risk",
        "home_rest_days", "away_rest_days",
        "away_travel_miles", "away_timezone_shift_hours",
    )
    return tuple(row.get(key) for key in important)


def _game_audit(
    game: Mapping[str, object], *,
    count: int, consistent: bool, now: datetime,
) -> dict[str, object]:
    blocked: list[str] = []
    flags: list[str] = []
    source_at = _parse(game.get("context_as_of"))
    kickoff = _parse(game.get("kickoff"))
    if source_at is None or source_at > now:
        blocked.append("UNKNOWN_OR_FUTURE_CONTEXT_TIMESTAMP")
    if kickoff is None or now >= kickoff:
        blocked.append("NO_VALID_FUTURE_KICKOFF")
    if not consistent:
        blocked.append("CONTEXT_DISAGREEMENT_BETWEEN_MARKETS")
    home_id = _text(game.get("home_expected_qb_id")).strip()
    away_id = _text(game.get("away_expected_qb_id")).strip()
    qb_identity = bool(home_id and away_id)
    qb_ready = (
        qb_identity
        and _yes(game.get("home_expected_qb_decision_ready"))
        and _yes(game.get("away_expected_qb_decision_ready"))
    )
    if not qb_ready:
        blocked.append("QB_IDENTITY_OR_STARTER_CERTAINTY_UNRESOLVED")

    injury_feed = _text(game.get("injury_feed_freshness_status")).upper() or "UNKNOWN"
    home_injury = _text(game.get("home_injury_freshness_status")).upper() or "UNKNOWN"
    away_injury = _text(game.get("away_injury_freshness_status")).upper() or "UNKNOWN"
    home_depth = _text(game.get("home_depth_freshness_status")).upper() or "UNKNOWN"
    away_depth = _text(game.get("away_depth_freshness_status")).upper() or "UNKNOWN"
    home_roster = _text(game.get("home_roster_freshness_status")).upper() or "UNKNOWN"
    away_roster = _text(game.get("away_roster_freshness_status")).upper() or "UNKNOWN"
    personnel_fresh = (
        _yes(game.get("context_injuries_personnel_fresh"))
        and all(_fresh(status) for status in (
            injury_feed, home_injury, away_injury, home_depth,
            away_depth, home_roster, away_roster,
        ))
    )
    if not _fresh(injury_feed):
        blocked.append("INJURY_FEED_UNTIMED_OR_STALE")
    if not _fresh(home_injury):
        blocked.append("HOME_INJURY_REPORT_UNVERIFIED")
    if not _fresh(away_injury):
        blocked.append("AWAY_INJURY_REPORT_UNVERIFIED")
    if not _fresh(home_depth) or not _fresh(away_depth):
        blocked.append("DEPTH_CHART_STALE_OR_UNKNOWN")
    if not _fresh(home_roster) or not _fresh(away_roster):
        blocked.append("ROSTER_STALE_OR_UNKNOWN")
    if not personnel_fresh and not any(
        reason in blocked for reason in (
            "INJURY_FEED_UNTIMED_OR_STALE", "HOME_INJURY_REPORT_UNVERIFIED",
            "AWAY_INJURY_REPORT_UNVERIFIED", "DEPTH_CHART_STALE_OR_UNKNOWN",
            "ROSTER_STALE_OR_UNKNOWN",
        )
    ):
        blocked.append("PERSONNEL_FRESHNESS_NOT_ESTABLISHED")

    rest_travel = _yes(game.get("context_rest_travel_available"))
    if not rest_travel:
        blocked.append("REST_OR_TRAVEL_PROVENANCE_UNKNOWN")
    weather_available = _yes(game.get("context_weather_stadium_available"))
    if not weather_available:
        flags.append("WEATHER_OR_VENUE_CONTEXT_UNKNOWN")

    home_change = _yes(game.get("home_expected_qb_changed_from_last_observed"))
    away_change = _yes(game.get("away_expected_qb_changed_from_last_observed"))
    if home_change:
        flags.append("HOME_QB_CHANGE_UNREPRICED")
    if away_change:
        flags.append("AWAY_QB_CHANGE_UNREPRICED")
    values = {
        key: _number(game.get(key))
        for key in (
            "home_injury_risk", "away_injury_risk",
            "home_starter_injury_risk", "away_starter_injury_risk",
            "home_ol_injury_risk", "away_ol_injury_risk",
            "home_skill_injury_risk", "away_skill_injury_risk",
            "home_defense_injury_risk", "away_defense_injury_risk",
            "home_rest_days", "away_rest_days",
            "home_travel_miles", "away_travel_miles",
            "away_timezone_shift_hours",
        )
    }
    if personnel_fresh:
        for side in ("home", "away"):
            for group, threshold in (
                ("ol", OL_STRESS), ("skill", SKILL_STRESS),
                ("defense", DEFENSE_STRESS),
            ):
                risk = values[f"{side}_{group}_injury_risk"]
                if risk is not None and risk >= threshold:
                    flags.append(f"{side.upper()}_{group.upper()}_HEALTH_STRESS")
    else:
        # Quantified availability risk from stale personnel is NOT proof of
        # any current advantage or of the absence of an injured starter.
        flags.append("PERSONNEL_RISK_SCORES_NOT_CURRENT")

    if rest_travel:
        for side in ("home", "away"):
            days = values[f"{side}_rest_days"]
            if days is not None and days <= SHORT_REST_DAYS:
                flags.append(f"{side.upper()}_SHORT_REST")
        away_miles = values["away_travel_miles"]
        if away_miles is not None and away_miles >= LONG_TRAVEL_MILES:
            flags.append("AWAY_LONG_TRAVEL")
        tz = values["away_timezone_shift_hours"]
        if tz is not None and abs(tz) >= TIMEZONE_SHIFT_HOURS:
            flags.append("AWAY_TIMEZONE_SHIFT")
    if _yes(game.get("neutral_site")):
        flags.append("NEUTRAL_SITE")

    # These are descriptive paired-availability contexts; the underlying
    # opponent pressure, continuity and replacement talent are NOT measured.
    home_matchup = None
    away_matchup = None
    if personnel_fresh:
        home_ol = values["home_ol_injury_risk"]
        away_ol = values["away_ol_injury_risk"]
        home_def = values["home_defense_injury_risk"]
        away_def = values["away_defense_injury_risk"]
        if home_ol is not None and away_def is not None:
            home_matchup = round(home_ol - away_def, 4)
        if away_ol is not None and home_def is not None:
            away_matchup = round(away_ol - home_def, 4)
    qb_ol_stack = personnel_fresh and (
        (home_change and values["home_ol_injury_risk"] is not None
         and values["home_ol_injury_risk"] >= OL_STRESS)
        or (away_change and values["away_ol_injury_risk"] is not None
            and values["away_ol_injury_risk"] >= OL_STRESS)
    )
    if qb_ol_stack:
        flags.append("QB_CHANGE_PLUS_OL_STRESS")
    rest_travel_stack = (
        rest_travel
        and values["away_rest_days"] is not None
        and values["away_rest_days"] <= SHORT_REST_DAYS
        and values["away_travel_miles"] is not None
        and values["away_travel_miles"] >= LONG_TRAVEL_MILES
    )
    if rest_travel_stack:
        flags.append("AWAY_SHORT_REST_PLUS_LONG_TRAVEL")
    status = (
        "BLOCKED_CONTEXT_PROVENANCE" if blocked
        else "OBSERVED_RESEARCH_ONLY"
    )
    output = {
        "season": game.get("season"), "week": game.get("week"),
        "game_id": game.get("game_id"), "kickoff": game.get("kickoff"),
        "home_team": game.get("home_team"), "away_team": game.get("away_team"),
        "evaluated_at": now.isoformat(),
        "source_context_as_of": game.get("context_as_of"),
        "status": status,
        "source_status": "PREKICKOFF_ONLY" if _assert_clock(source_at, now) else "UNKNOWN",
        "source_reasons": _text(game.get("context_freshness_reason")),
        "matched_market_candidates": count,
        "qb_identity_both_known": qb_identity, "qb_decision_ready": qb_ready,
        "home_qb_change": home_change, "away_qb_change": away_change,
        "home_qb_id": home_id or None, "away_qb_id": away_id or None,
        "home_qb_confidence": _number(game.get("home_expected_qb_confidence")),
        "away_qb_confidence": _number(game.get("away_expected_qb_confidence")),
        "home_qb_source": game.get("home_expected_qb_source"),
        "away_qb_source": game.get("away_expected_qb_source"),
        "injury_feed_status": injury_feed,
        "injury_feed_reported_at": game.get("injury_feed_latest_reported_at"),
        "home_injury_status": home_injury, "away_injury_status": away_injury,
        "home_depth_status": home_depth, "away_depth_status": away_depth,
        "home_roster_status": home_roster, "away_roster_status": away_roster,
        "personnel_fresh": personnel_fresh,
        "rest_travel_available": rest_travel,
        "weather_available": weather_available,
        "weather_error": game.get("weather_error"),
        **values,
        "home_ol_vs_away_defense_availability": home_matchup,
        "away_ol_vs_home_defense_availability": away_matchup,
        "stacked_qb_ol": qb_ol_stack,
        "stacked_rest_travel": rest_travel_stack,
        "flags": "|".join(sorted(set(flags))),
        "readiness_blockers": "|".join(sorted(set(blocked))),
        "research_staking_authorized": False,
    }
    return output


def build_context_intelligence(
    candidates: pl.DataFrame,
    *,
    as_of: datetime,
    source_meta: Mapping[str, object] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Read only current pregame annotations; never modify betting decisions."""
    if as_of.tzinfo is None:
        raise ValueError("context audit as_of needs an explicit timezone")
    now = as_of.astimezone(UTC)
    grouped: dict[str, list[dict[str, object]]] = {}
    if not candidates.is_empty():
        for row in candidates.iter_rows(named=True):
            key = _text(row.get("game_id")).strip()
            if key:
                grouped.setdefault(key, []).append(row)
    results = []
    for key, market_rows in sorted(grouped.items()):
        selected = market_rows[0]
        consistent = all(
            _signature(item) == _signature(selected) for item in market_rows
        )
        results.append(
            _game_audit(selected, count=len(market_rows),
                        consistent=consistent, now=now)
        )
    flags = Counter()
    blockers = Counter()
    for row in results:
        flags.update(filter(None, row["flags"].split("|")))
        blockers.update(filter(None, row["readiness_blockers"].split("|")))
    report: dict[str, object] = {
        "spec": SPEC,
        "status": (
            "NO_GAME_CONTEXT" if not results
            else "CONTEXT_PROVENANCE_BLOCKED" if blockers
            else "RESEARCH_CONTEXT_OBSERVED"
        ),
        "as_of": now.isoformat(),
        "season": SEASON,
        "games": len(results),
        "market_candidate_rows": sum(len(rows) for rows in grouped.values()),
        "ready_research_games": sum(
            row["status"] == "OBSERVED_RESEARCH_ONLY" for row in results
        ),
        "blocked_games": sum(
            row["status"] == "BLOCKED_CONTEXT_PROVENANCE" for row in results
        ),
        "qb_ready_games": sum(row["qb_decision_ready"] for row in results),
        "fresh_personnel_games": sum(row["personnel_fresh"] for row in results),
        "rest_travel_ready_games": sum(row["rest_travel_available"] for row in results),
        "weather_venue_ready_games": sum(row["weather_available"] for row in results),
        "flag_counts": dict(sorted(flags.items())),
        "blocker_counts": dict(sorted(blockers.items())),
        "current_context_status": (source_meta or {}).get("status"),
        "source_injury_feed_status": (
            ((source_meta or {}).get("injury_feed_freshness") or {}).get("status")
            if isinstance((source_meta or {}).get("injury_feed_freshness"), dict)
            else None
        ),
        "cutoffs_are_descriptive_not_optimized": True,
        "new_moneyline_spread_total_adjustment_enabled": False,
        "betting_authorized": False,
        "portfolio_and_fair_scores_unchanged": True,
        "historical_context_backfill_allowed": False,
        "limitations": [
            "No QB replacement effect, OL continuity coefficient, pass-rush or "
            "opponent-matchup strength was learned or applied from current feeds.",
            "Absent team-specific injuries cannot be inferred healthy from a "
            "league-wide fresh timestamp or another team's injury report.",
            "Missing injury-feed timestamps remain unknown and cannot be "
            "treated as up-to-date sportsbook execution evidence.",
            "Rest and travel proxies do not establish causal spread/total value.",
            "Each first pregame context snapshot must be frozen before kickoff; "
            "game outcomes are never read during feature extraction.",
            "Weather availability is descriptive; no unsupported weather "
            "adjustment changes the official score.",
        ],
    }
    return results, report


def append_first_context_snapshots(
    rows: list[dict[str, object]],
    *,
    path: str | Path = "history/context_forward_snapshots_v1.csv",
    as_of: datetime,
) -> dict[str, object]:
    """First observed context per game, including blocked contexts, is immutable."""
    if as_of.tzinfo is None:
        raise ValueError("context freeze as_of needs timezone")
    now = as_of.astimezone(UTC)
    dest = Path(path)
    prior = []
    if dest.exists() and dest.stat().st_size:
        with dest.open(newline="", encoding="utf-8") as handle:
            prior = list(csv.DictReader(handle))
    seen = set()
    for row in prior:
        key = _text(row.get("game_id"))
        if not key or key in seen:
            raise DataContractError("duplicate or empty forward context game key")
        seen.add(key)
    additions = []
    skipped = Counter()
    for row in sorted(rows, key=lambda r: str(r["game_id"])):
        game_id = _text(row.get("game_id"))
        if not game_id or game_id in seen:
            skipped["previously_frozen"] += 1
            continue
        kickoff = _parse(row.get("kickoff"))
        context_at = _parse(row.get("source_context_as_of"))
        if (
            _number(row.get("season")) != SEASON
            or kickoff is None or now >= kickoff
        ):
            skipped["historical_or_invalid_kickoff"] += 1
            continue
        # Unknown or future context provenance cannot be a valid first snapshot.
        if context_at is None or context_at > now:
            skipped["context_time_unknown_or_future"] += 1
            continue
        item = {field: row.get(field) for field in GAME_FIELDS}
        item["spec"] = SPEC
        item["captured_at"] = now.isoformat()
        additions.append(item)
        seen.add(game_id)
    if additions:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(FROZEN_FIELDS))
            writer.writeheader()
            writer.writerows(prior + additions)
    return {
        "status": "APPEND_ONLY_FIRST_PREGAME_CONTEXT",
        "appended": len(additions),
        "total": len(prior) + len(additions),
        "skipped": dict(sorted(skipped.items())),
        "research_only": True,
        "official_score_or_stake_changed": False,
        "historical_backfill": False,
    }


def write_context_intelligence(
    rows: list[dict[str, object]],
    report: Mapping[str, object],
    *,
    outputs_dir: str | Path = "outputs",
    docs_dir: str | Path = "docs",
) -> None:
    payload = json.dumps(dict(report), indent=2, sort_keys=True, default=str)
    for root in (Path(outputs_dir), Path(docs_dir)):
        root.mkdir(parents=True, exist_ok=True)
        (root / "context_intelligence_report.json").write_text(
            payload, encoding="utf-8"
        )
        with (root / "context_risk_register.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(GAME_FIELDS))
            writer.writeheader()
            writer.writerows(
                {field: row.get(field) for field in GAME_FIELDS}
                for row in rows
            )
