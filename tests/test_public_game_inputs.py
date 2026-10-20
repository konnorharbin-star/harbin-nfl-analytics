import copy
import json
from datetime import UTC, datetime, timedelta

import pytest

from nfl.public_game_inputs import input_contract, scoreboard_url, verify_game_snapshot
from scripts.nfl_web_model_intake import collect, coverage_markdown

AT = datetime(2026, 10, 14, 12, tzinfo=UTC)
GAME = {
    "game_id": "2026_06_SEA_DEN",
    "home_team": "DEN",
    "away_team": "SEA",
    "kickoff_utc": "2026-10-16T00:15:00+00:00",
    "status": "INDEPENDENT_CONDITIONAL_FORECAST",
}
EVENT = {
    "id": "401234567",
    "date": GAME["kickoff_utc"],
    "status": {"type": {"state": "pre", "completed": False, "name": "STATUS_SCHEDULED"}},
    "competitions": [
        {
            "date": GAME["kickoff_utc"],
            "neutralSite": False,
            "venue": {"id": "12", "fullName": "Example Stadium", "indoor": False},
            "competitors": [
                {"homeAway": "away", "team": {"abbreviation": "SEA"}},
                {"homeAway": "home", "team": {"abbreviation": "DEN"}},
            ],
        }
    ],
}


def verify(payload=None, **kwargs):
    args = dict(source_url=scoreboard_url(GAME), snapshot_sha256="a" * 64, captured=AT, asof=AT)
    args.update(kwargs)
    return verify_game_snapshot(payload or {"events": [copy.deepcopy(EVENT)]}, GAME, **args)


def test_current_event_extracts_only_literal_typed_fields_and_keeps_unknowns():
    result = verify()
    assert result["status"] == "SOURCE_BOUND_CURRENT_OBSERVATION"
    assert result["facts"]["venue_reported_indoor"] is False
    assert result["facts"]["neutral_site"] is False
    assert "starting_qb" not in result["facts"]
    assert result["source_publication_utc"] is None
    assert result["eligible_for_historical_backfill"] is False
    gate = input_contract(result, asof=AT, game=GAME)
    assert gate["event_binding_passed"]
    assert not gate["ready_for_context_adjusted_prediction"]
    assert "confirmed_starting_qbs" in gate["missing_required_inputs"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda e: e.update(date="2026-10-15T00:15:00Z"),
        lambda e: e["competitions"][0].update(date="2026-10-15T00:15:00Z"),
        lambda e: e["competitions"][0]["competitors"][0]["team"].update(abbreviation="KC"),
        lambda e: e["competitions"][0]["competitors"][0].update(homeAway="home"),
        lambda e: e.update(competitions=[]),
        lambda e: e.update(competitions=[None]),
        lambda e: e.update(date="2026-10-16T00:15:00"),
    ],
)
def test_trusted_feed_wrong_game_or_ambiguous_identity_is_rejected(mutation):
    event = copy.deepcopy(EVENT)
    mutation(event)
    result = verify({"events": [event]})
    assert result["status"] == "REJECTED" and result["facts"] == {}


@pytest.mark.parametrize(
    "state",
    [
        {"state": "in", "completed": False, "name": "STATUS_IN_PROGRESS"},
        {"state": "post", "completed": True, "name": "STATUS_FINAL"},
        {"state": "pre", "completed": False, "name": "STATUS_POSTPONED"},
        {"state": "pre"},
        None,
    ],
)
def test_postponed_live_final_and_unknown_statuses_block_inputs(state):
    event = copy.deepcopy(EVENT)
    event["status"] = {"type": state}
    assert verify({"events": [event]})["reasons"] == ["SOURCE_EVENT_NOT_SCHEDULED_PREGAME"]


def test_duplicate_matching_events_are_not_arbitrarily_selected():
    assert verify({"events": [EVENT, EVENT]})["reasons"] == ["NO_UNIQUE_EXACT_MATCHUP_AND_KICKOFF"]


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        (
            {"source_url": "https://site.api.espn.com.evil.test/scoreboard"},
            "UNAPPROVED_SOURCE_OR_GAME_DATE",
        ),
        ({"snapshot_sha256": "x" * 64}, "MISSING_RAW_SNAPSHOT_HASH"),
        ({"captured": AT + timedelta(seconds=1)}, "FUTURE_CAPTURE"),
        ({"captured": AT - timedelta(hours=6, seconds=1)}, "STALE_CAPTURE"),
        ({"asof": GAME["kickoff_utc"]}, "NOT_PREGAME"),
        ({"captured": AT.replace(tzinfo=None)}, "INVALID_GAME_OR_CAPTURE_CLOCK"),
    ],
)
def test_source_hash_and_chronology_checks(overrides, reason):
    assert reason in verify(**overrides)["reasons"]


def test_consumption_rechecks_age_and_exact_game_instead_of_trusting_old_pass():
    result = verify()
    assert not input_contract(result, asof=AT + timedelta(hours=7), game=GAME)[
        "event_binding_passed"
    ]
    assert not input_contract(result, asof=AT, game={**GAME, "away_team": "KC"})[
        "event_binding_passed"
    ]
    assert not input_contract(result, asof=GAME["kickoff_utc"], game=GAME)["event_binding_passed"]


def test_missing_or_malformed_optional_fields_never_become_zero_or_false():
    event = copy.deepcopy(EVENT)
    event["competitions"][0].update(venue={"indoor": "false"}, neutralSite=None)
    facts = verify({"events": [event]})["facts"]
    assert set(facts) == {"home_team", "away_team", "kickoff_utc"}


def test_source_date_is_eastern_game_date_not_utc_date():
    assert "dates=20261015" in scoreboard_url(GAME)


def test_intake_archives_raw_source_once_and_surfaces_partial_verification():
    requests, snapshots = [], []

    def fetch(url, **kwargs):
        requests.append(url)
        if kwargs.get("search"):
            return b"<rss><channel/></rss>"
        if "/scoreboard?" in url:
            return json.dumps({"events": [EVENT]}).encode()
        return b"<p>No structured facts.</p>"

    second = {**GAME, "game_id": "2026_06_KC_NYJ", "away_team": "KC", "home_team": "NYJ"}
    result = collect(
        {"games": [GAME, second], "player_props": []},
        fetch=fetch,
        now=lambda: AT,
        max_articles=0,
        save_raw=lambda digest, raw: snapshots.append((digest, raw)),
    )
    assert sum("/scoreboard?" in url for url in requests) == 1
    assert result["game_bindings_verified"] == 1
    first = result["games"][0]
    assert first["input_data_contract"]["status"] == "PARTIAL_VERIFIED_INPUTS"
    assert first["structured_game_observation"]["snapshot_sha256"] in dict(snapshots)
    assert result["games"][1]["input_data_contract"]["status"] == "ESSENTIAL_GAME_BINDING_BLOCKED"
    assert result["context_adjusted_predictions_ready"] == 0
    assert "PARTIAL_VERIFIED_INPUTS" in coverage_markdown(result)


def test_unavailable_feed_is_visible_and_does_not_erase_game_coverage():
    def fetch(url, **kwargs):
        raise OSError("offline")

    result = collect({"games": [GAME], "player_props": []}, fetch=fetch, now=lambda: AT)
    assert result["game_bindings_verified"] == 0
    assert any(e.get("topic") == "structured_game_inputs" for e in result["source_errors"])
    assert (
        "STRUCTURED_GAME_SOURCE_UNAVAILABLE"
        in result["games"][0]["input_data_contract"]["blocking_reasons"]
    )


def test_structured_source_redirect_to_another_approved_publisher_is_rejected(monkeypatch):
    from nfl.web_model_intake import fetch_page

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def geturl(self):
            return "https://www.espn.com/other-resource"

        def read(self, limit):
            raise AssertionError("Redirected content must not be read as the requested feed")

    monkeypatch.setattr("nfl.web_model_intake.urlopen", lambda *a, **k: Response())
    with pytest.raises(ValueError, match="STRUCTURED_SOURCE_REDIRECT"):
        fetch_page(scoreboard_url(GAME), exact_source=True)
