"""Dated, exact-event public injury observations and venue-bound NWS forecasts.

Reported injury status is not a final active list or a starter declaration.
Weather describes an outdoor forecast, not confirmed field/roof conditions.
"""

import math
import re
from collections import Counter
from datetime import timedelta

from nfl.official_personnel import personnel_from_articles
from nfl.public_game_inputs import clock, input_contract, verify_game_snapshot
from nfl.public_roof_inputs import roof_evidence
from nfl.weather import KNOWN_NEUTRAL_VENUES, TEAM_HOME, _venue_key

SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event="
# Match the actual event venue name, never just the nominal home team. These
# reuse the existing approximate coordinate registry; unknown venues stay unknown.
_HOME_VENUES = {
    "State Farm Stadium": "ARI",
    "Mercedes-Benz Stadium": "ATL",
    "M&T Bank Stadium": "BAL",
    "Bank of America Stadium": "CAR",
    "Soldier Field": "CHI",
    "Paycor Stadium": "CIN",
    "Huntington Bank Field": "CLE",
    "AT&T Stadium": "DAL",
    "Empower Field at Mile High": "DEN",
    "Ford Field": "DET",
    "Lambeau Field": "GB",
    "NRG Stadium": "HOU",
    "Lucas Oil Stadium": "IND",
    "EverBank Stadium": "JAX",
    "GEHA Field at Arrowhead Stadium": "KC",
    "Arrowhead Stadium": "KC",
    "SoFi Stadium": "LA",
    "Allegiant Stadium": "LV",
    "Hard Rock Stadium": "MIA",
    "U.S. Bank Stadium": "MIN",
    "Gillette Stadium": "NE",
    "Caesars Superdome": "NO",
    "MetLife Stadium": "NYG",
    "Lincoln Financial Field": "PHI",
    "Acrisure Stadium": "PIT",
    "Lumen Field": "SEA",
    "Levi's Stadium": "SF",
    "Raymond James Stadium": "TB",
    "Nissan Stadium": "TEN",
    "Northwest Stadium": "WAS",
}
VENUES = {_venue_key(name): TEAM_HOME[team][:2] for name, team in _HOME_VENUES.items()}
VENUES.update({_venue_key(name): point for name, point in KNOWN_NEUTRAL_VENUES.items()})
INJURY_STATUSES = {
    "Out",
    "Doubtful",
    "Questionable",
    "Injured Reserve",
    "Suspension",
    "Physically Unable to Perform",
    "Day-To-Day",
}


def numeric_id(value):
    if isinstance(value, bool) or value is None:
        return None
    text = str(value)
    return str(int(text)) if text.isascii() and text.isdigit() and int(text) > 0 else None


def unique_player_ids(rows):
    """Only a one-to-one GSIS/ESPN directory mapping can link a model player."""
    records = [
        (r.get("gsis_id"), numeric_id(r.get("espn_id")))
        for r in rows
        if isinstance(r, dict) and isinstance(r.get("gsis_id"), str)
    ]
    gsis_counts = Counter(g for g, _ in records if g)
    espn_counts = Counter(e for _, e in records if e)
    return {g: e for g, e in records if g and e and gsis_counts[g] == espn_counts[e] == 1}


def check_source(source, *, url, game, asof, max_age=timedelta(hours=6)):
    if source.get("source_url") != url:
        raise ValueError("SOURCE_URL_MISMATCH")
    digest = source.get("snapshot_sha256")
    if not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest):
        raise ValueError("MISSING_RAW_SNAPSHOT_HASH")
    captured, at, kickoff = map(clock, (source.get("captured_utc"), asof, game["kickoff_utc"]))
    if not captured <= at < kickoff or at - captured > max_age:
        raise ValueError("SOURCE_CAPTURE_NOT_FRESH_PREGAME")
    return captured


def reported_player_statuses(summary, players, game, *, captured):
    """A dated source assertion is accepted, never an inferred healthy player."""
    groups = summary.get("injuries")
    groups = groups if isinstance(groups, list) else []
    teams = {game["home_team"], game["away_team"]}
    aliases = {"LAR": "LA", "JAC": "JAX", "WSH": "WAS"}
    records = {}
    rejected = []
    for group in groups:
        if not isinstance(group, dict) or not isinstance(group.get("team"), dict):
            continue
        team = group["team"].get("abbreviation")
        team = aliases.get(team, team)
        if team not in teams:
            rejected.append("INJURY_TEAM_NOT_IN_EVENT")
            continue
        injuries = group.get("injuries")
        if not isinstance(injuries, list):
            rejected.append("MALFORMED_TEAM_INJURIES")
            continue
        for row in injuries:
            if not isinstance(row, dict):
                continue
            athlete = row.get("athlete")
            eid = numeric_id(athlete.get("id")) if isinstance(athlete, dict) else None
            if eid:
                records.setdefault((team, eid), []).append(row)
    players = [
        p
        for p in players
        if p.get("game_id") == game["game_id"] and not p.get("is_unallocated_bucket")
    ]
    ids = Counter(numeric_id(p.get("espn_athlete_id")) for p in players)
    out = []
    for player in players:
        if player.get("is_unallocated_bucket") or player.get("game_id") != game["game_id"]:
            continue
        team, eid = player.get("team"), numeric_id(player.get("espn_athlete_id"))
        result = {
            "player_id": player["player_id"],
            "team": team,
            "espn_athlete_id": eid,
            "status": "PLAYER_ID_NOT_VERIFIED",
            "reported_status": None,
            "source_update_utc": None,
            "confirmed_active": False,
            "confirmed_starting_qb": False,
            "model_adjustment": None,
        }
        matches = records.get((team, eid), [])
        if team not in teams:
            result["status"] = "PLAYER_TEAM_NOT_IN_EVENT"
        elif not eid or ids[eid] != 1:
            result["status"] = "PLAYER_ID_NOT_UNIQUE"
        elif not matches:
            result["status"] = "NOT_LISTED_DOES_NOT_PROVE_ACTIVE"
        else:
            eligible = []
            for row in matches:
                try:
                    updated = clock(row.get("date"))
                    if (
                        updated > captured
                        or updated >= clock(game["kickoff_utc"])
                        or captured - updated > timedelta(days=7)
                    ):
                        continue
                    if row.get("status") in INJURY_STATUSES:
                        eligible.append((updated, row["status"]))
                except (TypeError, ValueError):
                    continue
            if eligible:
                latest = max(t for t, _ in eligible)
                statuses = {s for t, s in eligible if t == latest}
                if len(statuses) == 1:
                    result.update(
                        status="DATED_EVENT_REPORTED_STATUS",
                        reported_status=statuses.pop(),
                        source_update_utc=latest.isoformat(),
                    )
                else:
                    result["status"] = "CONFLICTING_LATEST_INJURY_STATUSES"
            else:
                result["status"] = "NO_FRESH_DATED_INJURY_STATUS"
        out.append(result)
    return {
        "status": "SOURCE_BOUND_REPORTED_STATUSES_ONLY",
        "players": out,
        "rejections": rejected,
        "dated_reported_statuses": sum(p["reported_status"] is not None for p in out),
        "final_game_availability_verified": False,
        "starting_qbs_verified": False,
    }


def venue_weather(observation, game, *, read, now):
    """Use literal event venue name and a reviewed coordinate registry, not a team guess."""
    if not input_contract(observation, asof=now(), game=game)["event_binding_passed"]:
        raise ValueError("GAME_BINDING_REQUIRED_FOR_WEATHER")
    name = observation["facts"].get("venue_name")
    key = _venue_key(name)
    neutral = observation["facts"].get("neutral_site")
    home_venue = {_venue_key(n): team for n, team in _HOME_VENUES.items()}
    expected_home = home_venue.get(key)
    shared_home = {"LA": {"LA", "LAC"}, "NYG": {"NYG", "NYJ"}}
    if expected_home:
        if neutral is not False or game["home_team"] not in shared_home.get(
            expected_home, {expected_home}
        ):
            raise ValueError("EVENT_VENUE_LOCATION_NOT_CONFIRMED")
    elif neutral is not True:
        raise ValueError("NEUTRAL_VENUE_LOCATION_NOT_CONFIRMED")
    point = VENUES.get(key)
    if point is None:
        raise ValueError("EVENT_VENUE_COORDINATES_NOT_REGISTERED")
    latitude, longitude = point
    if not (24 <= latitude <= 50 and -125 <= longitude <= -66):
        raise ValueError("EVENT_VENUE_OUTSIDE_NWS_COVERAGE")
    point_url = f"https://api.weather.gov/points/{latitude:.4f},{longitude:.4f}"
    point_source = read(point_url)
    point_time = check_source(point_source, url=point_url, game=game, asof=now())
    hourly = point_source["payload"].get("properties", {}).get("forecastHourly")
    if not isinstance(hourly, str) or not re.fullmatch(
        r"https://api\.weather\.gov/gridpoints/[A-Z0-9]+/\d+,\d+/forecast/hourly", hourly
    ):
        raise ValueError("INVALID_NWS_HOURLY_SOURCE_URL")
    forecast = read(hourly)
    captured = check_source(forecast, url=hourly, game=game, asof=now())
    if captured < point_time:
        raise ValueError("WEATHER_SOURCE_CAPTURE_ORDER_INVALID")
    props = forecast["payload"].get("properties", {})
    issued = clock(props.get("updateTime") or props.get("updated") or props.get("generatedAt"))
    if issued > captured or captured - issued > timedelta(hours=6):
        raise ValueError("WEATHER_ISSUE_TIME_FUTURE_OR_STALE")
    kickoff = clock(game["kickoff_utc"])
    matches = []
    for period in props.get("periods", []):
        if not isinstance(period, dict):
            continue
        try:
            start, end = map(clock, (period.get("startTime"), period.get("endTime")))
            if start <= kickoff < end and end - start == timedelta(hours=1):
                matches.append((start, end, period))
        except (ValueError, TypeError):
            continue
    if len(matches) != 1:
        raise ValueError("NO_UNIQUE_KICKOFF_WEATHER_PERIOD")
    start, end, row = matches[0]
    temperature = row.get("temperature")
    unit = row.get("temperatureUnit")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or unit not in ("F", "C")
    ):
        raise ValueError("INVALID_WEATHER_TEMPERATURE_OR_UNIT")
    fahrenheit = float(temperature) if unit == "F" else float(temperature) * 9 / 5 + 32
    wind = row.get("windSpeed")
    wind_match = re.fullmatch(r"(\d+)(?: to (\d+))? mph", wind or "")
    if not wind_match:
        raise ValueError("INVALID_WEATHER_WIND_OR_UNIT")
    low = int(wind_match[1])
    high = int(wind_match[2] or wind_match[1])
    if low > high:
        raise ValueError("INVALID_WEATHER_WIND_RANGE")
    precipitation = row.get("probabilityOfPrecipitation")
    precipitation = precipitation if isinstance(precipitation, dict) else {}
    probability = precipitation.get("value")
    if probability is not None and (
        isinstance(probability, bool)
        or not isinstance(probability, (int, float))
        or not math.isfinite(probability)
        or not 0 <= probability <= 100
        or precipitation.get("unitCode") != "wmoUnit:percent"
    ):
        raise ValueError("INVALID_PRECIPITATION_VALUE_OR_UNIT")
    return {
        "status": "VENUE_BOUND_NWS_KICKOFF_FORECAST",
        "venue_name": name,
        "coordinate_source": "EXACT_EVENT_VENUE_NAME_EXISTING_APPROXIMATE_REGISTRY",
        "latitude": latitude,
        "longitude": longitude,
        "point_source": {k: v for k, v in point_source.items() if k != "payload"},
        "forecast_source": {k: v for k, v in forecast.items() if k != "payload"},
        "issued_utc": issued.isoformat(),
        "period_start_utc": start.isoformat(),
        "period_end_utc": end.isoformat(),
        "temperature_f": fahrenheit,
        "wind_mph_min": low,
        "wind_mph_max": high,
        "precipitation_probability_percent": probability,
        "roof_position_verified": False,
        "on_field_conditions_verified": False,
        "model_adjustment": None,
    }


def collect_context_inputs(game, observation, players, *, read, now):
    result = {
        "status": "PARTIAL_CONTEXT_ONLY",
        "source_errors": [],
        "availability": {
            "status": "SOURCE_UNAVAILABLE",
            "players": [],
            "dated_reported_statuses": 0,
            "final_game_availability_verified": False,
            "starting_qbs_verified": False,
        },
        "weather": {"status": "SOURCE_UNAVAILABLE"},
    }
    if not input_contract(observation, asof=now(), game=game)["event_binding_passed"]:
        result["status"] = "GAME_BINDING_BLOCKED"
        return result
    try:
        event_id = observation["provider_event_id"]
        url = SUMMARY + event_id
        source = read(url)
        captured = check_source(source, url=url, game=game, asof=now())
        summary = source["payload"]
        header = summary.get("header", {})
        competitions = header.get("competitions", [])
        if header.get("id") != event_id or len(competitions) != 1:
            raise ValueError("SUMMARY_EVENT_IDENTITY_MISMATCH")
        competition = competitions[0]
        event = {
            "id": header.get("id"),
            "date": competition.get("date"),
            "competitions": competitions,
            "status": competition.get("status"),
        }
        bound = verify_game_snapshot(
            {"events": [event]},
            game,
            source_url=url,
            summary_event_id=event_id,
            snapshot_sha256=source["snapshot_sha256"],
            captured=captured,
            asof=now(),
        )
        if bound["status"] != "SOURCE_BOUND_CURRENT_OBSERVATION":
            raise ValueError("SUMMARY_MATCHUP_OR_TIME_NOT_VERIFIED")
        result["availability"] = reported_player_statuses(summary, players, game, captured=captured)
        result["availability"]["source"] = {k: v for k, v in source.items() if k != "payload"}
    except (ValueError, TypeError, KeyError, AttributeError, OSError) as error:
        result["source_errors"].append({"source": "EVENT_INJURIES", "reason": str(error)[:150]})
    try:
        result["weather"] = venue_weather(observation, game, read=read, now=now)
    except (ValueError, TypeError, KeyError, AttributeError, OSError) as error:
        result["weather"] = {"status": "WEATHER_NOT_VERIFIED", "reason": str(error)[:150]}
        result["source_errors"].append({"source": "VENUE_WEATHER", "reason": str(error)[:150]})
    return result


def context_contract(observation, context, game, *, asof, players=None):
    """Publish only evidence still eligible at the actual consumption clock."""
    result = input_contract(observation, asof=asof, game=game)
    result.update(reported_injury_statuses_verified=0, outdoor_weather_verified=False)
    result.update(
        confirmed_starting_qb_teams=[],
        official_inactive_teams=[],
        explicit_player_designations_verified=0,
        roof_operating_position_required=None,
        roof_status="ROOF_CLASSIFICATION_UNVERIFIED",
        outdoor_weather_required=True,
    )
    if not result["event_binding_passed"]:
        return result
    availability = context.get("availability", {})
    allowed = (
        None
        if players is None
        else {
            (p.get("player_id"), p.get("team"), numeric_id(p.get("espn_athlete_id")))
            for p in players
            if p.get("game_id") == game["game_id"] and not p.get("is_unallocated_bucket")
        }
    )
    try:
        captured = check_source(
            availability.get("source", {}),
            url=SUMMARY + observation["provider_event_id"],
            game=game,
            asof=asof,
        )
        result["reported_injury_statuses_verified"] = sum(
            p.get("status") == "DATED_EVENT_REPORTED_STATUS"
            and (
                allowed is None
                or (p.get("player_id"), p.get("team"), numeric_id(p.get("espn_athlete_id")))
                in allowed
            )
            and p.get("reported_status") in INJURY_STATUSES
            and clock(asof) - timedelta(days=7) <= clock(p.get("source_update_utc")) <= captured
            for p in availability.get("players", [])
        )
    except (KeyError, TypeError, ValueError):
        pass
    weather = context.get("weather", {})
    try:
        if weather.get("status") == "VENUE_BOUND_NWS_KICKOFF_FORECAST":
            source = weather["forecast_source"]
            url = source["source_url"]
            if not isinstance(url, str) or not re.fullmatch(
                r"https://api\.weather\.gov/gridpoints/[A-Z0-9]+/\d+,\d+/forecast/hourly", url
            ):
                raise ValueError("INVALID_WEATHER_SOURCE")
            captured = check_source(source, url=url, game=game, asof=asof)
            issued = clock(weather["issued_utc"])
            kickoff = clock(game["kickoff_utc"])
            start, end = map(clock, (weather["period_start_utc"], weather["period_end_utc"]))
            if (
                weather.get("venue_name") == observation["facts"].get("venue_name")
                and issued <= captured
                and clock(asof) - issued <= timedelta(hours=6)
                and start <= kickoff < end
                and end - start == timedelta(hours=1)
            ):
                result["outdoor_weather_verified"] = True
    except (KeyError, TypeError, ValueError):
        pass
    if result["reported_injury_statuses_verified"]:
        result["verified_fields"].append("dated_reported_player_statuses")
    if result["outdoor_weather_verified"]:
        result["verified_fields"].append("outdoor_kickoff_weather_forecast")
    official = personnel_from_articles(
        context.get("official_articles", []), game, players or [], asof=asof
    )
    result["confirmed_starting_qb_teams"] = sorted(official["starters"])
    result["official_inactive_teams"] = sorted(
        t
        for t, r in official["inactive_reports"].items()
        if r["status"] == "OFFICIAL_GAME_DAY_INACTIVE_LIST"
    )
    result["explicit_player_designations_verified"] = official["confirmed_player_designations"]
    result["model_players_needing_availability"] = max(
        0, official["model_players_reviewed"] - official["confirmed_player_designations"]
    )
    roof = roof_evidence(context.get("roof_schedule_snapshot"), observation, game, asof=asof)
    result["roof_operating_position_required"] = roof["operating_position_required"]
    result["roof_status"] = roof["status"]
    result["outdoor_weather_required"] = roof["outdoor_weather_required"]
    result["missing_required_inputs"] = []
    if len(result["confirmed_starting_qb_teams"]) != 2:
        result["missing_required_inputs"].append("confirmed_starting_qbs")
    if not official["model_players_reviewed"] or result["model_players_needing_availability"]:
        result["missing_required_inputs"].append("final_game_specific_player_availability")
    if roof["operating_position_required"] is not False:
        result["missing_required_inputs"].append("game_day_roof_position")
    if not result["outdoor_weather_verified"] and roof["outdoor_weather_required"]:
        result["missing_required_inputs"].append("venue_bound_kickoff_weather_forecast")
    if result["confirmed_starting_qb_teams"]:
        result["verified_fields"].append("explicit_official_starting_quarterback_announcements")
    if result["official_inactive_teams"]:
        result["verified_fields"].append("official_game_day_inactive_lists")
    if roof["operating_position_required"] is False:
        result["verified_fields"].append("roof_operating_position_not_applicable")
    result["essential_context_inputs_complete"] = not result["missing_required_inputs"]
    return result
