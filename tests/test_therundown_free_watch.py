"""Free main-line watchdog may observe bookmaker-labeled odds but never certify bets."""
import copy
import hashlib
import json
from datetime import UTC, datetime, timedelta

import pytest

from scripts.therundown_free_watch import (
    _event_matching,
    _fetch,
    _team_match,
    _url,
    extract_pair,
    scan,
)

NOW = datetime(2026, 10, 10, 21, 30, tzinfo=UTC)
KICKOFF = NOW + timedelta(hours=20)


def fixed_forecast(sport="nfl"):
    home = "ATL" if sport == "nfl" else "Alabama"
    away = "BAL" if sport == "nfl" else "Georgia"
    return {
        "game_id": "2026_05_BAL_ATL" if sport == "nfl" else "401856712",
        "sport": sport,
        "home_team": home, "away_team": away,
        "kickoff": KICKOFF.isoformat(),
        "captured_at": (NOW - timedelta(hours=1)).isoformat(),
        "spec": "walters_key_number_forward_v1",
        "release_state": "SHADOW",
        "fixed_alpha": 1.0,
        "sportsbook_odds_used": False,
        "season": 2026,
        "independent_home_margin": -1.2,
        "model_mean": .3, "model_sigma": 13.0,
        "key_multipliers": {"3": 2.2, "7": 1.4},
    }


def market_event(sport="nfl"):
    f = fixed_forecast(sport)
    names = (
        ("Baltimore Ravens", "Atlanta Falcons")
        if sport == "nfl" else
        ("Georgia Bulldogs", "Alabama Crimson Tide")
    )
    book_times = {
        "19": (NOW - timedelta(minutes=6)).isoformat(),
        "22": (NOW - timedelta(minutes=6)).isoformat(),
        "23": (NOW - timedelta(minutes=6)).isoformat(),
    }
    def prices(odds):
        return {
            k: {
                "price": odds, "is_main_line": True,
                "updated_at": date, "id": "price-" + k,
            } for k, date in book_times.items()
        }
    away = {
        "id": 10, "type": "TYPE_TEAM", "name": names[0],
        "lines": [{"id": "line-away", "value": "+3", "prices": prices(-110)}],
    }
    home = {
        "id": 20, "type": "TYPE_TEAM", "name": names[1],
        "lines": [{"id": "line-home", "value": "-3", "prices": prices(-110)}],
    }
    return {
        "event_id": "provider-match-id",
        "sport_id": 2 if sport == "nfl" else 1,
        "event_date": f["kickoff"],
        "teams": [
            {"name": names[0], "abbreviation": "BAL",
             "team_id": 10},
            {"name": names[1], "abbreviation": "ATL",
             "team_id": 20},
        ],
        "markets": [{
            "market_id": 2, "period_id": 0,
            "participants": [away, home],
        }],
    }


def _paths(tmp_path, sport="nfl"):
    f = fixed_forecast(sport)
    directory = tmp_path / "forecasts"
    directory.mkdir()
    (directory / (f["game_id"] + ".json")).write_text(json.dumps(f))
    evidence = tmp_path / "indicative"
    return directory, evidence


def commit(_):
    return NOW - timedelta(minutes=10)


def dummy_fetch(sport="nfl"):
    events = [market_event(sport)]
    result = {"events": events}
    return result, hashlib.sha256(json.dumps(result).encode()).hexdigest()


def test_frozen_three_book_spreads_match_and_no_bet(tmp_path):
    root, target = _paths(tmp_path)
    calls = []
    def fetch(url, key):
        assert key == "token_in_header_only"
        calls.append(url)
        return dummy_fetch()
    report = scan("nfl", root, target, now=NOW,
                  key="token_in_header_only", request_fn=fetch,
                  commit_lookup=commit)
    assert report["forecasts_with_immutable_pregame_publication"] == 1
    assert report["indicative_paired_quotes"] == 3
    assert report["observations_written"] == 3
    assert len(list(target.glob("*.json"))) == 3
    assert report["measured_clv"] is None
    assert report["betting_authorized"] is False
    assert all(r["status"] == "INDICATIVE_AGGREGATOR_WATCH_NOT_BET"
               for r in report["research_price_comparisons"])
    assert all(r["wager_authorized"] is False
               for r in report["research_price_comparisons"])
    assert all("token_in_header_only" not in url for url in calls)
    assert "token_in_header_only" not in json.dumps(report)
    assert not any("token_in_header_only" in p.read_text()
                   for p in target.glob("*.json"))
    second = scan("nfl", root, target, now=NOW,
                  key="token_in_header_only", request_fn=fetch,
                  commit_lookup=commit)
    assert second["observations_written"] == 0


def test_absent_free_key_does_not_call_provider(tmp_path):
    root, target = _paths(tmp_path)
    def never(*_):
        raise AssertionError("Network request should not occur")
    report = scan("nfl", root, target, now=NOW,
                  key=None, request_fn=never, commit_lookup=commit)
    assert report["mode"] == "NO_KEY_NO_PRICE"
    assert report["research_price_comparisons"] == []
    assert not target.exists()


def test_invalid_pubtime_never_fetches_or_backfills(tmp_path):
    root, target = _paths(tmp_path)
    report = scan("nfl", root, target, now=NOW, key="fake",
                  request_fn=lambda *_: (_ for _ in ()).throw(
                      AssertionError("Never query before model is frozen")
                  ),
                  commit_lookup=lambda _: None)
    assert report["forecasts_with_immutable_pregame_publication"] == 0
    assert not target.exists()
    assert report["betting_authorized"] is False


def test_same_book_opposite_prices_and_integer_push():
    event = market_event()
    paired = extract_pair(event, fixed_forecast(), "19", NOW)
    assert paired["home"]["line"] == -3
    assert paired["away"]["line"] == +3
    assert paired["home"]["odds"] == -110
    assert paired["away"]["odds"] == -110


def test_rejects_incoherent_opposing_books_and_lines():
    event = market_event()
    event["markets"][0]["participants"][0]["lines"][0]["value"] = "+3.5"
    with pytest.raises(ValueError, match="DIFFERING"):
        extract_pair(event, fixed_forecast(), "19", NOW)
    event = market_event()
    event["markets"][0]["participants"][0]["lines"][0]["prices"]["19"][
        "price"
    ] = 0.0001
    with pytest.raises(ValueError, match="MISSING"):
        extract_pair(event, fixed_forecast(), "19", NOW)


def test_rejects_stale_and_future_price_updates():
    event = market_event()
    for timestamp in (
        NOW - timedelta(minutes=16), NOW + timedelta(seconds=1)
    ):
        e = copy.deepcopy(event)
        e["markets"][0]["participants"][0]["lines"][0]["prices"]["19"][
            "updated_at"
        ] = timestamp.isoformat()
        with pytest.raises(ValueError, match="MISSING"):
            extract_pair(e, fixed_forecast(), "19", NOW)
    event["markets"][0]["participants"][0]["lines"][0]["prices"]["19"][
        "updated_at"
    ] = (NOW - timedelta(minutes=14)).isoformat()
    with pytest.raises(ValueError, match="ASYNCHRONOUS"):
        extract_pair(event, fixed_forecast(), "19", NOW)


def test_source_metadata_requires_canonical_team_identity_and_period():
    event = market_event()
    assert _event_matching(fixed_forecast(), event)
    assert not _event_matching(
        fixed_forecast(), {**event, "event_date": (
            KICKOFF + timedelta(hours=2)
        ).isoformat()}
    )
    event["markets"][0]["period_id"] = 1
    with pytest.raises(ValueError, match="MISSING"):
        extract_pair(event, fixed_forecast(), "19", NOW)
    assert not _team_match("BAL", {"name": "Baltimore Ravens"}, "nfl")


def test_ncaaf_school_names_and_exact_participant_ids(tmp_path):
    f = fixed_forecast("cfb")
    e = market_event("cfb")
    assert _event_matching(f, e)
    assert _team_match("Alabama", {"name": "Alabama Crimson Tide"}, "cfb")
    assert not _team_match("Alabama", {"name": "Auburn Tigers"}, "cfb")
    directory, target = _paths(tmp_path, "cfb")
    result = scan("cfb", directory, target, now=NOW, key="free_key",
                  request_fn=lambda *_: dummy_fetch("cfb"),
                  commit_lookup=commit)
    assert result["indicative_paired_quotes"] == 3
    assert all(v["home_team"] == "Alabama"
               for v in result["research_price_comparisons"])
    e["markets"][0]["participants"][0]["id"] = 30
    with pytest.raises(ValueError, match="MISSING_OPPOSITE_TEAM"):
        extract_pair(e, f, "19", NOW)


def test_provider_failure_is_declared_without_fabricated_bets(tmp_path):
    directory, target = _paths(tmp_path)
    def offline(*_):
        raise TimeoutError("No internet on runner")
    result = scan("nfl", directory, target, now=NOW, key="free",
                  request_fn=offline, commit_lookup=commit)
    assert not result["research_price_comparisons"]
    assert len(result["source_failures"]) >= 1
    assert not target.exists()


def test_request_uses_documented_free_markets_and_no_key():
    url = _url("nfl", "2026-10-11")
    assert "/sports/2/events/2026-10-11" in url
    assert "market_ids=2" in url and "affiliate_ids=19%2C22%2C23" in url
    assert "main_line=true" in url and "offset=300" in url
    assert "apiKey" not in url
    assert _url("cfb", "2026-10-10").split("/events/")[0].endswith(
        "/sports/1"
    )


def test_no_demo_or_secretless_live_request():
    with pytest.raises(ValueError, match="MISSING_FREE_API_KEY"):
        _fetch("https://therundown.io/api/v2/sports/2/events/2026-10-11", "")


def test_full_validation_rejects_ambiguous_provider_events(tmp_path):
    directory, target = _paths(tmp_path)
    response = {"events": [market_event(), market_event()]}
    response["events"][1]["event_id"] = "collision"
    out = scan("nfl", directory, target, now=NOW, key="fake",
               request_fn=lambda *_: (response, "testhash"),
               commit_lookup=commit)
    assert out["research_price_comparisons"] == []
    assert "UNMATCHED_OR_AMBIGUOUS_PROVIDER_EVENT" in out[
        "forecast_exclusions"
    ]
