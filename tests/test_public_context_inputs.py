import copy
from datetime import UTC, datetime, timedelta

import pytest

from nfl.public_context_inputs import (
    SUMMARY,
    collect_context_inputs,
    context_contract,
    reported_player_statuses,
    unique_player_ids,
    venue_weather,
)
from nfl.public_game_inputs import scoreboard_url, verify_game_snapshot

AT = datetime(2026, 10, 19, 23, tzinfo=UTC)
GAME = {
    "game_id": "2026_06_WAS_SF",
    "home_team": "SF",
    "away_team": "WAS",
    "kickoff_utc": "2026-10-20T00:15:00Z",
}
PLAYER = {
    "game_id": GAME["game_id"],
    "team": "SF",
    "player_id": "00-001",
    "espn_athlete_id": "12345",
    "player_name": "Example Player",
}
STATE = {"type": {"state": "pre", "name": "STATUS_SCHEDULED", "completed": False}}
COMPETITION = {
    "date": GAME["kickoff_utc"],
    "status": STATE,
    "neutralSite": False,
    "competitors": [
        {"homeAway": "home", "team": {"abbreviation": "SF"}},
        {"homeAway": "away", "team": {"abbreviation": "WSH"}},
    ],
    "venue": {"fullName": "Levi's Stadium", "id": "4738", "indoor": False},
}
POINT = "https://api.weather.gov/points/37.4030,-121.9700"
HOURLY = "https://api.weather.gov/gridpoints/MTR/85,105/forecast/hourly"


def fixture():
    event = {
        "id": "401873008",
        "date": GAME["kickoff_utc"],
        "status": STATE,
        "competitions": [copy.deepcopy(COMPETITION)],
    }
    observation = verify_game_snapshot(
        {"events": [event]},
        GAME,
        source_url=scoreboard_url(GAME),
        snapshot_sha256="a" * 64,
        captured=AT,
        asof=AT,
    )
    summary = {
        "header": {"id": "401873008", "competitions": [copy.deepcopy(COMPETITION)]},
        "injuries": [
            {
                "team": {"abbreviation": "SF"},
                "injuries": [
                    {
                        "date": (AT - timedelta(minutes=15)).isoformat(),
                        "status": "Out",
                        "athlete": {"id": "12345"},
                    }
                ],
            }
        ],
    }
    forecast = {
        "properties": {
            "updateTime": (AT - timedelta(minutes=30)).isoformat(),
            "periods": [
                {
                    "startTime": "2026-10-20T00:00:00Z",
                    "endTime": "2026-10-20T01:00:00Z",
                    "temperature": 64,
                    "temperatureUnit": "F",
                    "windSpeed": "5 to 10 mph",
                    "probabilityOfPrecipitation": {"value": 0, "unitCode": "wmoUnit:percent"},
                }
            ],
        }
    }
    sources = {
        SUMMARY + "401873008": summary,
        POINT: {"properties": {"forecastHourly": HOURLY}},
        HOURLY: forecast,
    }

    def read(url):
        return {
            "source_url": url,
            "snapshot_sha256": "b" * 64,
            "captured_utc": AT.isoformat(),
            "payload": sources[url],
        }

    return observation, sources, read


def test_collector_accepts_dated_status_and_outdoor_forecast_without_inventing_actives():
    observation, _, read = fixture()
    result = collect_context_inputs(GAME, observation, [PLAYER], read=read, now=lambda: AT)
    assert result["source_errors"] == []
    status = result["availability"]["players"][0]
    assert status["reported_status"] == "Out"
    assert not status["confirmed_active"] and not status["confirmed_starting_qb"]
    weather = result["weather"]
    assert weather["status"] == "VENUE_BOUND_NWS_KICKOFF_FORECAST"
    assert weather["temperature_f"] == 64
    assert weather["wind_mph_min"] == 5 and weather["wind_mph_max"] == 10
    assert weather["precipitation_probability_percent"] == 0
    assert weather["roof_position_verified"] is False
    gate = context_contract(observation, result, GAME, asof=AT)
    assert gate["reported_injury_statuses_verified"] == 1
    assert gate["outdoor_weather_verified"] is True
    assert not gate["ready_for_context_adjusted_prediction"]
    assert "confirmed_starting_qbs" in gate["missing_required_inputs"]


def test_directory_mapping_requires_unique_ids_on_both_sides():
    rows = [
        {"gsis_id": "good", "espn_id": "10"},
        {"gsis_id": "bad", "espn_id": "20"},
        {"gsis_id": "other", "espn_id": "20"},
        {"gsis_id": "repeat", "espn_id": "30"},
        {"gsis_id": "repeat", "espn_id": "30"},
        {"gsis_id": "bool", "espn_id": True},
    ]
    assert unique_player_ids(rows) == {"good": "10"}


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s["injuries"][0]["injuries"][0].update(date="2026-10-20T01:00:00Z"),
        lambda s: s["injuries"][0]["injuries"][0].update(date="2026-10-01T00:00:00Z"),
        lambda s: s["injuries"][0]["injuries"][0].update(date="2026-10-19T20:00:00"),
        lambda s: s["injuries"][0]["injuries"][0].update(status="Healthy"),
        lambda s: s["injuries"][0]["injuries"][0]["athlete"].update(id="99999"),
        lambda s: s["injuries"][0]["team"].update(abbreviation="DEN"),
    ],
)
def test_wrong_id_team_undated_future_or_stale_injuries_cannot_verify_player(change):
    _, sources, _ = fixture()
    summary = sources[SUMMARY + "401873008"]
    change(summary)
    result = reported_player_statuses(summary, [PLAYER], GAME, captured=AT)
    assert result["dated_reported_statuses"] == 0
    assert result["players"][0]["reported_status"] is None
    assert result["players"][0]["confirmed_active"] is False


def test_conflicting_latest_rows_and_ambiguous_player_mapping_fail_closed():
    _, sources, _ = fixture()
    summary = sources[SUMMARY + "401873008"]
    rows = summary["injuries"][0]["injuries"]
    rows.append({**rows[0], "status": "Questionable"})
    result = reported_player_statuses(summary, [PLAYER], GAME, captured=AT)
    assert result["players"][0]["status"] == "CONFLICTING_LATEST_INJURY_STATUSES"
    result = reported_player_statuses(
        summary, [PLAYER, {**PLAYER, "player_id": "other"}], GAME, captured=AT
    )
    assert result["dated_reported_statuses"] == 0


def test_absent_row_remains_unknown_and_other_game_player_does_not_create_ambiguity():
    _, sources, _ = fixture()
    summary = sources[SUMMARY + "401873008"]
    result = reported_player_statuses(
        summary, [PLAYER, {**PLAYER, "game_id": "other"}], GAME, captured=AT
    )
    assert result["dated_reported_statuses"] == 1
    summary["injuries"] = []
    result = reported_player_statuses(summary, [PLAYER], GAME, captured=AT)
    assert result["players"][0]["status"] == "NOT_LISTED_DOES_NOT_PROVE_ACTIVE"


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s["header"].update(id="99999"),
        lambda s: s["header"]["competitions"][0].update(date="2026-10-21T00:15Z"),
        lambda s: s["header"]["competitions"][0]["competitors"][0]["team"].update(
            abbreviation="KC"
        ),
        lambda s: s["header"]["competitions"][0]["status"]["type"].update(state="post"),
    ],
)
def test_summary_requires_exact_event_match_before_accepting_injuries(change):
    observation, sources, read = fixture()
    change(sources[SUMMARY + "401873008"])
    result = collect_context_inputs(GAME, observation, [PLAYER], read=read, now=lambda: AT)
    assert result["availability"]["dated_reported_statuses"] == 0
    assert any(e["source"] == "EVENT_INJURIES" for e in result["source_errors"])


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p.update(updateTime="2026-10-20T00:00:00Z"),
        lambda p: p.update(updateTime="2026-10-19T10:00:00Z"),
        lambda p: p["periods"][0].update(temperatureUnit="unknown"),
        lambda p: p["periods"][0].update(temperature=True),
        lambda p: p["periods"][0].update(windSpeed="10 knots"),
        lambda p: p["periods"][0].update(startTime="2026-10-20T01:00:00Z"),
        lambda p: p["periods"][0]["probabilityOfPrecipitation"].update(value=105),
        lambda p: p["periods"].append(copy.deepcopy(p["periods"][0])),
    ],
)
def test_weather_rejects_wrong_clock_units_periods_and_invalid_measurements(change):
    observation, sources, read = fixture()
    change(sources[HOURLY]["properties"])
    with pytest.raises(ValueError):
        venue_weather(observation, GAME, read=read, now=lambda: AT)


def test_neutral_changed_or_unknown_venue_never_falls_back_to_home_coordinates():
    observation, _, read = fixture()
    for venue, neutral in [("Unregistered Stadium", False), ("Levi's Stadium", True)]:
        changed = copy.deepcopy(observation)
        changed["facts"].update(venue_name=venue, neutral_site=neutral)
        with pytest.raises(ValueError):
            venue_weather(changed, GAME, read=read, now=lambda: AT)


def test_weather_does_not_follow_arbitrary_url_or_assume_missing_precipitation_is_zero():
    observation, sources, read = fixture()
    sources[POINT]["properties"]["forecastHourly"] = "https://api.weather.gov.evil.test/grid"
    with pytest.raises(ValueError, match="SOURCE_URL"):
        venue_weather(observation, GAME, read=read, now=lambda: AT)
    sources[POINT]["properties"]["forecastHourly"] = HOURLY
    sources[HOURLY]["properties"]["periods"][0]["probabilityOfPrecipitation"] = {}
    result = venue_weather(observation, GAME, read=read, now=lambda: AT)
    assert result["precipitation_probability_percent"] is None


def test_network_elapsed_time_is_captured_without_future_timestamp_false_rejection():
    observation, _, fixed_read = fixture()
    current = [AT]

    def read(url):
        current[0] += timedelta(seconds=1)
        record = fixed_read(url)
        record["captured_utc"] = current[0].isoformat()
        return record

    result = venue_weather(observation, GAME, read=read, now=lambda: current[0])
    assert result["status"] == "VENUE_BOUND_NWS_KICKOFF_FORECAST"


def test_consumption_rechecks_age_postkickoff_and_game_binding():
    observation, _, read = fixture()
    context = collect_context_inputs(GAME, observation, [PLAYER], read=read, now=lambda: AT)
    after = context_contract(observation, context, GAME, asof="2026-10-20T00:15:00Z")
    assert not after["outdoor_weather_verified"]
    assert after["reported_injury_statuses_verified"] == 0
    assert not after["event_binding_passed"]


def test_consumption_rejects_foreign_weather_source_and_wrong_model_player_id():
    observation, _, read = fixture()
    context = collect_context_inputs(GAME, observation, [PLAYER], read=read, now=lambda: AT)
    context["weather"]["forecast_source"]["source_url"] = "https://untrusted.test/forecast"
    contract = context_contract(
        observation, context, GAME, asof=AT, players=[{**PLAYER, "espn_athlete_id": "wrong"}]
    )
    assert contract["outdoor_weather_verified"] is False
    assert contract["reported_injury_statuses_verified"] == 0
