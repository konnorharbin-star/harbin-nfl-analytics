"""Event-bound nflverse roof classifications, never guessed from indoor=False."""

import hashlib
import json
import re
from datetime import timedelta

from nfl.coupled_nflverse_asof import kickoff_utc
from nfl.public_game_inputs import clock, input_contract
from nfl.weather import _venue_key


def schedule_snapshot(rows, *, captured):
    fields = (
        "game_id",
        "season",
        "week",
        "home_team",
        "away_team",
        "gameday",
        "gametime",
        "stadium",
        "roof",
        "espn",
    )
    records = [{k: row.get(k) for k in fields} for row in rows]
    raw = json.dumps(records, sort_keys=True, default=str).encode()
    records = json.loads(raw)
    # This hashes the normalized exported rows, not upstream raw provider bytes.
    return {
        "source_method": "nflreadpy.load_schedules",
        "captured_utc": clock(captured).isoformat(),
        "snapshot_sha256": hashlib.sha256(raw).hexdigest(),
        "records": records,
        "hash_scope": "NORMALIZED_EXPORTED_ROWS_NOT_ORIGINAL_PROVIDER_BYTES",
    }


def roof_evidence(snapshot, observation, game, *, asof):
    result = {
        "status": "ROOF_CLASSIFICATION_UNVERIFIED",
        "operating_position_required": None,
        "outdoor_weather_required": True,
        "model_adjustment": None,
    }
    try:
        if not input_contract(observation, asof=asof, game=game)["event_binding_passed"]:
            raise ValueError("GAME_BINDING_REQUIRED_FOR_ROOF")
        if (
            not isinstance(snapshot, dict)
            or snapshot.get("source_method") != "nflreadpy.load_schedules"
        ):
            raise ValueError("SCHEDULE_SOURCE_UNAVAILABLE")
        captured, at, kickoff = map(clock, (snapshot["captured_utc"], asof, game["kickoff_utc"]))
        if not captured <= at < kickoff or at - captured > timedelta(hours=6):
            raise ValueError("SCHEDULE_NOT_FRESH_PREGAME")
        if not re.fullmatch(r"[0-9a-f]{64}", snapshot.get("snapshot_sha256", "")):
            raise ValueError("SCHEDULE_SNAPSHOT_MISSING")
        raw = json.dumps(snapshot["records"], sort_keys=True, default=str).encode()
        if hashlib.sha256(raw).hexdigest() != snapshot["snapshot_sha256"]:
            raise ValueError("SCHEDULE_SNAPSHOT_CHANGED")
        matches = [r for r in snapshot["records"] if r.get("game_id") == game["game_id"]]
        if len(matches) != 1:
            raise ValueError("NO_UNIQUE_SCHEDULE_GAME")
        row = matches[0]
        venue = observation["facts"].get("venue_name")
        if not isinstance(row.get("stadium"), str) or not row["stadium"].strip() or not venue:
            raise ValueError("SCHEDULE_VENUE_NOT_VERIFIED")
        source_event = str(row.get("espn"))
        if re.fullmatch(r"[0-9]+\.0", source_event):
            source_event = source_event[:-2]
        if (
            row.get("home_team") != game["home_team"]
            or row.get("away_team") != game["away_team"]
            or str(row.get("season")) != game["game_id"].split("_")[0]
            or int(row.get("week")) != int(game["game_id"].split("_")[1])
            or kickoff_utc(row) != kickoff
            or _venue_key(row.get("stadium")) != _venue_key(observation["facts"].get("venue_name"))
            or source_event != observation.get("provider_event_id")
        ):
            raise ValueError("SCHEDULE_EVENT_OR_VENUE_MISMATCH")
        result["source"] = {k: v for k, v in snapshot.items() if k != "records"}
        result["source_row"] = row
        result["reported_roof"] = row.get("roof")
        if row.get("roof") == "outdoors":
            result.update(
                status="OUTDOOR_ROOF_POSITION_NOT_APPLICABLE", operating_position_required=False
            )
        elif row.get("roof") == "dome":
            result.update(
                status="FIXED_ROOF_POSITION_NOT_APPLICABLE",
                operating_position_required=False,
                outdoor_weather_required=observation["facts"].get("venue_reported_indoor")
                is not True,
            )
        elif row.get("roof") in ("open", "closed", "retractable"):
            result.update(
                status="RETRACTABLE_ROOF_POSITION_UNCONFIRMED", operating_position_required=True
            )
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        result["reason"] = str(error)[:150]
    return result
