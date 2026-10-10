from datetime import UTC, datetime, timedelta

import pytest

from scripts.personnel_evidence import evaluate, evidence_state, injury_records

NOW = datetime(2026, 10, 10, 18, tzinfo=UTC)


def record(**changes):
    r = {
        "player_id": "123",
        "source_item_id": "injury1",
        "source_status": "Out",
        "source_item_date": "2026-10-10T15:35Z",
        "position": "QB",
        "team": {"id": "2", "abbreviation": "BUF", "displayName": "Buffalo Bills"},
        "group_team_id": "2",
    }
    return {**r, **changes}


def state(r):
    return evidence_state(
        r, observed=NOW, kickoff=NOW + timedelta(days=1), source_season=2026, game_season=2026
    )


def test_response_freshness_cannot_refresh_old_injury_item():
    assert state(record(source_item_date="2020-11-21T18:31Z")) == "STALE_ITEM_DATE"
    assert state(record()) == "FRESH_DATED_SECONDARY_STATUS"


@pytest.mark.parametrize("dated", [None, "2026-10-10", "2026-10-10T15:35"])
def test_report_requires_aware_item_time(dated):
    assert state(record(source_item_date=dated)) == "MISSING_AWARE_ITEM_DATE"


def test_future_report_and_identity_mismatch_block():
    assert state(record(source_item_date="2026-10-11T15:35Z")) == "FUTURE_OR_POSTKICKOFF_ITEM_DATE"
    assert state(record(group_team_id="wrong")) == "TEAM_IDENTITY_MISMATCH"
    assert state(record(player_id=None)) == "MISSING_PLAYER_ID_OR_STATUS"


def test_dated_status_does_not_verify_starter_or_roster_health():
    games = [
        {
            "game_id": "g",
            "sport": "nfl",
            "season": 2026,
            "kickoff": (NOW + timedelta(days=1)).isoformat(),
            "home_team": "BUF",
            "away_team": "NE",
        }
    ]
    result, _ = evaluate([record()], games + games, observed=NOW, source_season=2026)
    assert len(result) == 2
    assert result[0]["fresh_source_record_count"] == 1
    assert result[0]["starter_verified"] is False
    assert result[0]["score_adjustment_enabled"] is False
    assert result[1]["status"] == "NO_FRESH_PERSONNEL_EVIDENCE"
    assert result[1]["healthy_roster_inferred"] is False


def test_started_games_never_receive_backfilled_personnel():
    games = [
        {
            "game_id": "g",
            "sport": "nfl",
            "season": 2026,
            "kickoff": NOW.isoformat(),
            "home_team": "BUF",
            "away_team": "NE",
        }
    ]
    result, counts = evaluate([record()], games, observed=NOW, source_season=2026)
    assert result == [] and counts["games_already_started"] == 1


def test_player_id_is_extracted_from_source_link_without_name_guessing():
    payload = {
        "injuries": [
            {
                "id": "2",
                "injuries": [
                    {
                        "id": "inj1",
                        "date": "2026-10-10T15:35Z",
                        "status": "Out",
                        "athlete": {
                            "displayName": "Fixture",
                            "links": [{"href": "https://www.espn.com/nfl/player/_/id/123/fixture"}],
                            "team": {"id": "2", "abbreviation": "BUF"},
                            "position": {"abbreviation": "QB"},
                        },
                    }
                ],
            }
        ]
    }
    r = injury_records(payload)[0]
    assert r["player_id"] == "123" and r["source_item_date"] == "2026-10-10T15:35Z"
    payload["injuries"][0]["injuries"][0]["athlete"]["id"] = "456"
    assert injury_records(payload)[0]["player_id"] is None
