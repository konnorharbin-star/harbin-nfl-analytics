import copy
from datetime import date, timedelta

import pytest

from nfl.public_context_inputs import context_contract
from nfl.public_roof_inputs import roof_evidence, schedule_snapshot
from tests.test_public_context_inputs import AT, GAME, fixture

ROW = {
    "game_id": GAME["game_id"],
    "season": 2026,
    "week": 6,
    "home_team": "SF",
    "away_team": "WAS",
    "gameday": date(2026, 10, 19),
    "gametime": "20:15",
    "stadium": "Levi's Stadium",
    "roof": "outdoors",
    "espn": "401873008",
}


def test_event_bound_outdoor_roof_is_not_applicable_and_datetime_rows_serialize():
    observation, _, _ = fixture()
    snapshot = schedule_snapshot([ROW], captured=AT)
    roof = roof_evidence(snapshot, observation, GAME, asof=AT)
    assert roof["status"] == "OUTDOOR_ROOF_POSITION_NOT_APPLICABLE"
    assert roof["operating_position_required"] is False
    contract = context_contract(observation, {"roof_schedule_snapshot": snapshot}, GAME, asof=AT)
    assert "game_day_roof_position" not in contract["missing_required_inputs"]
    assert contract["outdoor_weather_required"] is True
    assert snapshot["records"][0]["gameday"] == "2026-10-19"


@pytest.mark.parametrize("roof", ["open", "closed", "retractable", None])
def test_retractable_or_unknown_roof_never_becomes_confirmed_position(roof):
    observation, _, _ = fixture()
    snapshot = schedule_snapshot([{**ROW, "roof": roof}], captured=AT)
    result = roof_evidence(snapshot, observation, GAME, asof=AT)
    assert result["operating_position_required"] is not False


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r.update(home_team="DEN"),
        lambda r: r.update(stadium="Different Venue"),
        lambda r: r.update(espn="99999"),
        lambda r: r.update(gametime="20:16"),
    ],
)
def test_exact_game_venue_event_id_and_kickoff_are_required(change):
    observation, _, _ = fixture()
    row = copy.deepcopy(ROW)
    change(row)
    result = roof_evidence(schedule_snapshot([row], captured=AT), observation, GAME, asof=AT)
    assert result["status"] == "ROOF_CLASSIFICATION_UNVERIFIED"


def test_missing_stale_duplicate_changed_or_postkickoff_schedule_stays_unknown():
    observation, _, _ = fixture()
    good = schedule_snapshot([ROW], captured=AT)
    changed = copy.deepcopy(good)
    changed["records"][0]["roof"] = "dome"
    for snapshot, asof in [
        (None, AT),
        (schedule_snapshot([ROW, ROW], captured=AT), AT),
        (changed, AT),
        (schedule_snapshot([ROW], captured=AT - timedelta(hours=7)), AT),
        (good, "2026-10-20T00:16:00Z"),
    ]:
        assert (
            roof_evidence(snapshot, observation, GAME, asof=asof)["operating_position_required"]
            is None
        )


def test_fixed_roof_weather_is_not_suppressed_for_open_sided_indoor_false_venue():
    observation, _, _ = fixture()
    snapshot = schedule_snapshot([{**ROW, "roof": "dome"}], captured=AT)
    result = roof_evidence(snapshot, observation, GAME, asof=AT)
    assert result["operating_position_required"] is False
    assert result["outdoor_weather_required"] is True
    observation["facts"]["venue_reported_indoor"] = True
    assert roof_evidence(snapshot, observation, GAME, asof=AT)["outdoor_weather_required"] is False
