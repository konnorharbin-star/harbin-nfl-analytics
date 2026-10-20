"""Exact-event, current public observations. Never infer player availability.

The collection clock is provenance for a live observation, not a publisher
timestamp and not permission to backfill an earlier forecast.
"""

import re
from datetime import UTC, datetime, timedelta
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
MAX_AGE = timedelta(hours=6)
ALIASES = {"LAR": "LA", "WSH": "WAS", "JAC": "JAX"}


def clock(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("ZONED_TIME_REQUIRED")
    return value.astimezone(UTC)


def scoreboard_url(game):
    day = clock(game["kickoff_utc"]).astimezone(ZoneInfo("America/New_York"))
    return SCOREBOARD + "?" + urlencode({"dates": day.strftime("%Y%m%d"), "limit": 100})


def verify_game_snapshot(
    payload, game, *, source_url, snapshot_sha256, captured, asof, summary_event_id=None
):
    """Bind one structured provider event to exact teams/time, failing closed.

    Acceptance verifies the provenance and event binding of provider assertions;
    it does not establish that every situational input is present or correct.
    """
    result = {
        "status": "REJECTED",
        "reasons": [],
        "source_url": source_url,
        "snapshot_sha256": snapshot_sha256,
        "captured_utc": None,
        "source_publication_utc": None,
        "source_kind": "ESPN_CURRENT_EVENT_SNAPSHOT",
        "facts": {},
        "eligible_for_historical_backfill": False,
        "model_adjustment": None,
        "used_as_calibrated_feature": False,
    }

    def reject(reason):
        result["reasons"].append(reason)
        return result

    try:
        kickoff, observed, decision = map(clock, (game["kickoff_utc"], captured, asof))
        result["captured_utc"] = observed.isoformat()
        expected_url = scoreboard_url(game)
        if summary_event_id is not None:
            if not isinstance(summary_event_id, str) or not re.fullmatch(
                r"[0-9]+", summary_event_id
            ):
                return reject("INVALID_EXPECTED_SUMMARY_EVENT_ID")
            expected_url = (
                "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event="
                + summary_event_id
            )
            result["source_kind"] = "ESPN_CURRENT_GAME_SUMMARY"
    except (KeyError, TypeError, ValueError):
        return reject("INVALID_GAME_OR_CAPTURE_CLOCK")
    if source_url != expected_url:
        return reject("UNAPPROVED_SOURCE_OR_GAME_DATE")
    if (
        not isinstance(snapshot_sha256, str)
        or len(snapshot_sha256) != 64
        or any(c not in "0123456789abcdef" for c in snapshot_sha256)
    ):
        return reject("MISSING_RAW_SNAPSHOT_HASH")
    if observed > decision:
        return reject("FUTURE_CAPTURE")
    if decision >= kickoff:
        return reject("NOT_PREGAME")
    if decision - observed > MAX_AGE:
        return reject("STALE_CAPTURE")
    events = payload.get("events") if isinstance(payload, dict) else None
    if not isinstance(events, list):
        return reject("INVALID_EVENT_FEED")
    matches = []
    for event in events:
        if not isinstance(event, dict):
            continue
        competitions = event.get("competitions")
        if not isinstance(competitions, list) or len(competitions) != 1:
            continue
        competition = competitions[0]
        if not isinstance(competition, dict):
            continue
        competitors = competition.get("competitors")
        if not isinstance(competitors, list) or len(competitors) != 2:
            continue
        teams = {}
        for entry in competitors:
            if not isinstance(entry, dict) or not isinstance(entry.get("team"), dict):
                continue
            side = entry.get("homeAway")
            team = entry["team"].get("abbreviation")
            if side in teams or side not in ("home", "away") or not isinstance(team, str):
                continue
            teams[side] = ALIASES.get(team, team)
        if teams != {"home": game.get("home_team"), "away": game.get("away_team")}:
            continue
        try:
            if clock(event.get("date")) != kickoff or clock(competition.get("date")) != kickoff:
                continue
        except (TypeError, ValueError):
            continue
        matches.append((event, competition))
    if len(matches) != 1:
        return reject("NO_UNIQUE_EXACT_MATCHUP_AND_KICKOFF")
    event, competition = matches[0]
    event_id = event.get("id")
    if not isinstance(event_id, str) or not event_id.isascii() or not event_id.isdigit():
        return reject("MISSING_PROVIDER_EVENT_ID")
    if summary_event_id is not None and event_id != summary_event_id:
        return reject("SUMMARY_EVENT_IDENTITY_MISMATCH")
    status = event.get("status")
    state = status.get("type") if isinstance(status, dict) else None
    if (
        not isinstance(state, dict)
        or state.get("state") != "pre"
        or state.get("completed") is not False
        or state.get("name") != "STATUS_SCHEDULED"
    ):
        return reject("SOURCE_EVENT_NOT_SCHEDULED_PREGAME")
    venue = competition.get("venue")
    venue = venue if isinstance(venue, dict) else {}
    facts = {
        "away_team": game["away_team"],
        "home_team": game["home_team"],
        "kickoff_utc": kickoff.isoformat(),
    }
    for source_key, fact_key in (("fullName", "venue_name"), ("id", "venue_id")):
        value = venue.get(source_key)
        if isinstance(value, str) and value.strip():
            facts[fact_key] = value.strip()
    if type(venue.get("indoor")) is bool:
        facts["venue_reported_indoor"] = venue["indoor"]
    if type(competition.get("neutralSite")) is bool:
        facts["neutral_site"] = competition["neutralSite"]
    result.update(
        status="SOURCE_BOUND_CURRENT_OBSERVATION",
        provider_event_id=event_id,
        facts=facts,
    )
    return result


def input_contract(observation, *, asof, game):
    """Recheck freshness at consumption, even after an earlier accepted capture."""
    reasons = list(observation.get("reasons", []))
    accepted = observation.get("status") == "SOURCE_BOUND_CURRENT_OBSERVATION"
    try:
        at, kickoff = map(clock, (asof, game["kickoff_utc"]))
        captured = clock(observation.get("captured_utc"))
        facts = observation.get("facts", {})
        if (
            facts.get("home_team") != game.get("home_team")
            or facts.get("away_team") != game.get("away_team")
            or clock(facts.get("kickoff_utc")) != kickoff
        ):
            reasons.append("OBSERVATION_GAME_MISMATCH")
        if captured > at:
            reasons.append("FUTURE_CAPTURE")
        if at >= kickoff:
            reasons.append("NOT_PREGAME")
        if at - captured > MAX_AGE:
            reasons.append("STALE_CAPTURE")
    except (KeyError, TypeError, ValueError):
        reasons.append("INVALID_GAME_OR_CAPTURE_CLOCK")
    accepted = accepted and not reasons
    return {
        "status": "PARTIAL_VERIFIED_INPUTS" if accepted else "ESSENTIAL_GAME_BINDING_BLOCKED",
        "event_binding_passed": accepted,
        "blocking_reasons": list(dict.fromkeys(reasons)),
        "verified_fields": sorted(observation.get("facts", {})) if accepted else [],
        "missing_required_inputs": [
            "confirmed_starting_qbs",
            "game_specific_player_availability",
            "game_time_weather_and_roof_status",
        ],
        "ready_for_context_adjusted_prediction": False,
        "forecast_effect": "NO_WEB_CONTEXT_ADJUSTMENT",
    }
