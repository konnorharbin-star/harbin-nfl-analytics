from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from nfl.context import build_current_context
from nfl.contracts import DataContractError
from nfl.injuries import normalize_injuries, summarize_team_injuries
from nfl.personnel import normalize_depth_charts, normalize_rosters, summarize_team_personnel
from nfl.weather import OpenMeteoNFLWeather

AS_OF = datetime(2026, 10, 1, 16, tzinfo=UTC)


def _injuries() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "season": 2026,
                "week": 4,
                "team": "BUF",
                "gsis_id": "qb-buf",
                "position": "QB",
                "full_name": "Buffalo QB",
                "report_status": "Out",
                "practice_status": "Did Not Participate",
                "date_modified": "2026-09-29T12:00:00Z",
            },
            {
                "season": 2026,
                "week": 4,
                "team": "BUF",
                "gsis_id": "qb-buf",
                "position": "QB",
                "full_name": "Buffalo QB",
                "report_status": "Active",
                "practice_status": "Full Participation",
                "date_modified": "2026-09-30T12:00:00Z",
            },
            {
                "season": 2026,
                "week": 4,
                "team": "BUF",
                "gsis_id": "qb-buf",
                "position": "QB",
                "full_name": "Buffalo QB",
                "report_status": "Out",
                "practice_status": "Did Not Participate",
                "date_modified": "2026-10-02T12:00:00Z",
            },
            {
                "season": 2026,
                "week": 4,
                "team": "NE",
                "gsis_id": "wr-ne",
                "position": "WR",
                "full_name": "New England WR",
                "report_status": "Questionable",
                "practice_status": "Limited Participation",
                "date_modified": "2026-09-30T13:00:00Z",
            },
        ]
    )


def _depth() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "season": 2026,
                "team": "BUF",
                "gsis_id": "qb-buf",
                "player_name": "Buffalo QB",
                "pos_abb": "QB",
                "pos_rank": 1,
                "dt": "2026-09-30T10:00:00Z",
            },
            {
                "season": 2026,
                "team": "BUF",
                "gsis_id": "qb-buf",
                "player_name": "Buffalo QB",
                "pos_abb": "QB",
                "pos_rank": 2,
                "dt": "2026-10-02T10:00:00Z",
            },
            {
                "season": 2026,
                "team": "NE",
                "gsis_id": "wr-ne",
                "player_name": "New England WR",
                "pos_abb": "WR",
                "pos_rank": 1,
                "dt": "2026-09-30T10:00:00Z",
            },
        ]
    )


def _rosters() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "season": 2026,
                "week": 4,
                "team": "BUF",
                "position": "QB",
                "gsis_id": "qb-buf",
                "full_name": "Buffalo QB",
                "status": "Active",
            },
            {
                "season": 2026,
                "week": 4,
                "team": "NE",
                "position": "WR",
                "gsis_id": "wr-ne",
                "full_name": "New England WR",
                "status": "Active",
            },
            {
                "season": 2026,
                "week": 4,
                "team": "NE",
                "position": "QB",
                "gsis_id": "qb-ne",
                "full_name": "New England QB",
                "status": "Active",
            },
        ]
    )


def _target(*, roof: str = "dome") -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "season": 2026,
                "week": 4,
                "game_id": "2026_04_NE_BUF",
                "game_type": "REG",
                "gameday": "2026-10-04",
                "gametime": "13:00",
                "away_team": "NE",
                "home_team": "BUF",
                "away_score": None,
                "home_score": None,
                "away_rest": 7,
                "home_rest": 7,
                "location": "Home",
                "roof": roof,
                "stadium": "Test Stadium",
            }
        ]
    )


def test_injury_normalization_uses_latest_admissible_status() -> None:
    normalized = normalize_injuries(_injuries(), season=2026, week=4, as_of=AS_OF)
    buffalo = normalized.filter(pl.col("team") == "BUF")
    assert buffalo.height == 1
    assert buffalo.get_column("report_status")[0] == "Active"
    assert buffalo.get_column("severity")[0] == 0.0
    summary = summarize_team_injuries(normalized)
    ne = summary.filter(pl.col("team") == "NE")
    assert ne.get_column("injury_count")[0] == 1
    assert ne.get_column("injury_risk")[0] > 0


def test_depth_chart_future_snapshot_is_excluded() -> None:
    depth = normalize_depth_charts(_depth(), season=2026, week=4, as_of=AS_OF)
    buffalo = depth.filter(pl.col("team") == "BUF")
    assert buffalo.height == 1
    assert buffalo.get_column("depth_rank")[0] == 1


def test_personnel_matches_injuries_to_top_depth_players() -> None:
    injuries = normalize_injuries(_injuries(), season=2026, week=4, as_of=AS_OF)
    depth = normalize_depth_charts(_depth(), season=2026, week=4, as_of=AS_OF)
    rosters = normalize_rosters(_rosters(), season=2026, week=4)
    summary = summarize_team_personnel(depth, rosters, injuries)
    ne = summary.filter(pl.col("team") == "NE")
    assert ne.get_column("starter_injury_risk")[0] > 0
    assert ne.get_column("skill_injury_risk")[0] > 0


def test_indoor_context_does_not_call_weather_source() -> None:
    def forbidden_fetch(_: str) -> dict[str, object]:
        raise AssertionError("indoor game should not fetch weather")

    weather = OpenMeteoNFLWeather(fetch_json=forbidden_fetch)
    context, meta = build_current_context(
        _target(roof="dome"),
        season=2026,
        week=4,
        injuries=_injuries(),
        depth_charts=_depth(),
        rosters=_rosters(),
        as_of=AS_OF,
        weather_client=weather,
    )
    assert context.get_column("indoor")[0]
    assert context.get_column("weather_risk")[0] == 0.0
    assert context.get_column("weather_source")[0] == "indoor"
    assert meta["score_adjustment_enabled"] is False


def test_outdoor_context_uses_nearest_open_meteo_hour() -> None:
    def fake_fetch(url: str) -> dict[str, object]:
        assert "api.open-meteo.com" in url
        return {
            "hourly": {
                "time": ["2026-10-04T16:00", "2026-10-04T17:00", "2026-10-04T18:00"],
                "temperature_2m": [55.0, 52.0, 50.0],
                "precipitation_probability": [10.0, 70.0, 80.0],
                "wind_speed_10m": [8.0, 22.0, 25.0],
                "wind_gusts_10m": [12.0, 35.0, 38.0],
            }
        }

    weather = OpenMeteoNFLWeather(fetch_json=fake_fetch)
    context, meta = build_current_context(
        _target(roof="outdoors"),
        season=2026,
        week=4,
        injuries=_injuries(),
        depth_charts=_depth(),
        rosters=_rosters(),
        as_of=AS_OF,
        weather_client=weather,
    )
    assert context.get_column("weather_source")[0] == "Open-Meteo"
    assert context.get_column("weather_risk")[0] > 0
    assert context.get_column("away_travel_miles")[0] > 0
    assert meta["components"]["weather_stadium"] == 1.0
    assert meta["components"]["rest_travel"] == 1.0


def test_current_context_fails_closed_for_historical_season() -> None:
    with pytest.raises(DataContractError, match="disabled for historical season"):
        build_current_context(
            _target(),
            season=2025,
            week=4,
            as_of=AS_OF,
        )
