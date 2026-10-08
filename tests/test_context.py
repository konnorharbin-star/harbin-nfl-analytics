from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from nfl.context import (
    apply_context_confidence_veto,
    apply_context_freshness_veto,
    build_current_context,
)
from nfl.contracts import DataContractError
from nfl.injuries import (
    injury_feed_freshness,
    normalize_injuries,
    summarize_team_injuries,
)
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


def test_prior_week_injury_status_does_not_carry_into_fresh_week() -> None:
    frame = pl.DataFrame(
        [
            {
                "season": 2026,
                "week": 3,
                "team": "BUF",
                "gsis_id": "qb-buf",
                "position": "QB",
                "full_name": "Buffalo QB",
                "report_status": "Out",
                "practice_status": "Did Not Participate",
                "date_modified": "2026-09-24T12:00:00Z",
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

    normalized = normalize_injuries(
        frame,
        season=2026,
        week=4,
        as_of=AS_OF,
    )

    assert normalized.filter(pl.col("team") == "BUF").is_empty()
    assert normalized.filter(pl.col("team") == "NE").height == 1


def test_injury_feed_without_target_week_is_stale_and_risk_is_not_carried() -> None:
    prior = _injuries().with_columns(pl.lit(3).alias("week"))

    freshness = injury_feed_freshness(
        prior,
        season=2026,
        week=4,
        as_of=AS_OF,
    )
    normalized = normalize_injuries(
        prior,
        season=2026,
        week=4,
        as_of=AS_OF,
    )
    summary = summarize_team_injuries(normalized)

    assert freshness["status"] == "STALE"
    assert freshness["current_week_rows"] == 0
    assert summary.get_column("injury_count").sum() == 0
    assert set(summary.get_column("injury_freshness_status").to_list()) == {
        "STALE"
    }


def test_depth_chart_future_snapshot_is_excluded() -> None:
    depth = normalize_depth_charts(_depth(), season=2026, week=4, as_of=AS_OF)
    buffalo = depth.filter(pl.col("team") == "BUF")
    assert buffalo.height == 1
    assert buffalo.get_column("depth_rank")[0] == 1


def test_depth_and_roster_sources_expose_freshness_state() -> None:
    depth = normalize_depth_charts(
        _depth(),
        season=2026,
        week=4,
        as_of=AS_OF,
    )
    rosters = normalize_rosters(_rosters(), season=2026, week=4)
    injuries = normalize_injuries(
        _injuries(),
        season=2026,
        week=4,
        as_of=AS_OF,
    )
    summary = summarize_team_personnel(depth, rosters, injuries)

    assert set(depth.get_column("depth_freshness_status").to_list()) == {
        "FRESH"
    }
    assert set(rosters.get_column("roster_freshness_status").to_list()) == {
        "FRESH"
    }
    assert set(summary.get_column("personnel_freshness_status").to_list()) == {
        "FRESH"
    }


def test_roster_normalization_uses_latest_team_snapshot() -> None:
    rosters = pl.DataFrame(
        [
            {
                "season": 2026,
                "week": 3,
                "team": "BUF",
                "position": "QB",
                "gsis_id": "old-qb",
                "full_name": "Old QB",
                "status": "Active",
            },
            {
                "season": 2026,
                "week": 4,
                "team": "BUF",
                "position": "QB",
                "gsis_id": "current-qb",
                "full_name": "Current QB",
                "status": "Active",
            },
            {
                "season": 2026,
                "week": 4,
                "team": "BUF",
                "position": "WR",
                "gsis_id": "current-wr",
                "full_name": "Current WR",
                "status": "Active",
            },
        ]
    )

    normalized = normalize_rosters(rosters, season=2026, week=4)

    assert normalized.height == 2
    assert "old-qb" not in normalized.get_column("gsis_id").to_list()
    assert normalized.filter(pl.col("position") == "QB").height == 1
    assert set(normalized.get_column("roster_week").to_list()) == {4}


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


def test_current_context_marks_fresh_injury_personnel_state() -> None:
    context, meta = build_current_context(
        _target(roof="dome"),
        season=2026,
        week=4,
        injuries=_injuries(),
        depth_charts=_depth(),
        rosters=_rosters(),
        as_of=AS_OF,
    )

    row = context.row(0, named=True)
    assert row["context_injuries_personnel_available"] is True
    assert row["context_injuries_personnel_fresh"] is True
    assert row["injury_feed_freshness_status"] == "FRESH"
    assert row["home_personnel_freshness_status"] == "FRESH"
    assert row["away_personnel_freshness_status"] == "FRESH"
    assert meta["injury_feed_fresh"] is True
    assert meta["components"]["injuries_personnel"] == 1.0


def test_context_freshness_veto_blocks_production_but_preserves_research_signal() -> None:
    candidates = pl.DataFrame(
        [
            {
                "game_id": "g1",
                "quant_market": "total",
                "quant_signal": "BET",
                "production_signal": "BET",
                "research_signal": "STRONG",
                "stake_units": 0.4,
                "research_stake_units": 0.4,
                "context_injuries_personnel_fresh": False,
                "context_freshness_reason": "injury feed stale",
            }
        ]
    )

    row = apply_context_freshness_veto(candidates).row(0, named=True)

    assert row["context_freshness_veto"] is True
    assert row["quant_signal"] == "PASS"
    assert row["production_signal"] == "PASS"
    assert row["research_signal"] == "STRONG"
    assert row["stake_units"] == 0.0
    assert row["research_stake_units"] == 0.0
    assert row["research_context_pending"] is True
    assert "injury feed stale" in row["context_freshness_veto_reason"]


def test_current_context_fails_closed_for_historical_season() -> None:
    with pytest.raises(DataContractError, match="disabled for historical season"):
        build_current_context(
            _target(),
            season=2025,
            week=4,
            as_of=AS_OF,
        )


def test_stacked_adverse_context_veto_blocks_side_bet() -> None:
    candidates = pl.DataFrame(
        [
            {
                "game_id": "2026_04_ATL_NO",
                "quant_market": "moneyline",
                "quant_side": "home",
                "quant_signal": "PASS",
                "production_signal": "PASS",
                "research_signal": "STRONG",
                "stake_units": 0.0,
                "research_stake_units": 0.20,
                "context_injuries_personnel_available": True,
                "context_injuries_personnel_fresh": True,
                "context_rest_travel_available": True,
                "home_injury_risk": 0.875,
                "away_injury_risk": 0.3125,
                "home_starter_injury_risk": 0.5739,
                "away_starter_injury_risk": 0.2083,
                "home_rest_advantage_days": -3.0,
            }
        ]
    )
    row = apply_context_confidence_veto(candidates).row(0, named=True)
    assert row["context_veto"] is True
    assert "stacked adverse context" in row["context_veto_reason"]
    assert row["research_signal"] == "PASS"
    assert row["research_stake_units"] == 0.0


def test_context_veto_does_not_block_when_rest_is_not_adverse() -> None:
    candidates = pl.DataFrame(
        [
            {
                "game_id": "g1",
                "quant_market": "spread",
                "quant_side": "home",
                "quant_signal": "PASS",
                "production_signal": "PASS",
                "research_signal": "BET",
                "stake_units": 0.0,
                "research_stake_units": 0.15,
                "context_injuries_personnel_available": True,
                "context_injuries_personnel_fresh": True,
                "context_rest_travel_available": True,
                "home_injury_risk": 0.90,
                "away_injury_risk": 0.30,
                "home_starter_injury_risk": 0.60,
                "away_starter_injury_risk": 0.20,
                "home_rest_advantage_days": 0.0,
            }
        ]
    )
    row = apply_context_confidence_veto(candidates).row(0, named=True)
    assert row["context_veto"] is False
    assert row["research_signal"] == "BET"
    assert row["research_stake_units"] == 0.15


def test_fresh_league_injury_feed_does_not_certify_missing_away_team() -> None:
    """An observed injury report for BUF cannot mark missing NE injuries healthy."""
    only_buf = _injuries().filter(pl.col("team") == "BUF")
    context, meta = build_current_context(
        _target(roof="dome"),
        season=2026, week=4,
        injuries=only_buf,
        depth_charts=_depth(),
        rosters=_rosters(),
        as_of=AS_OF,
    )
    row = context.row(0, named=True)
    assert meta["injury_feed_fresh"] is True
    assert row["home_injury_freshness_status"] == "FRESH"
    assert row["away_injury_freshness_status"] == "UNKNOWN"
    assert row["home_injury_source_available"] is True
    assert row["away_injury_source_available"] is False
    assert row["context_injuries_personnel_fresh"] is False
    assert "NE injury state unknown" in row["context_freshness_reason"]


def test_explicit_official_inactive_overrides_full_practice_report() -> None:
    altered = _injuries().with_columns(
        pl.when(pl.col("team") == "BUF")
        .then(pl.lit("Inactive"))
        .otherwise(pl.col("report_status"))
        .alias("report_status"),
        pl.when(pl.col("team") == "BUF")
        .then(pl.lit("Full Participation"))
        .otherwise(pl.col("practice_status"))
        .alias("practice_status"),
    )
    # Use the 9/30 row, ignoring the 10/02 post-as-of update.
    normalized = normalize_injuries(
        altered, season=2026, week=4, as_of=AS_OF
    )
    buf = normalized.filter(pl.col("team") == "BUF").row(0, named=True)
    assert buf["severity"] == 1.0
