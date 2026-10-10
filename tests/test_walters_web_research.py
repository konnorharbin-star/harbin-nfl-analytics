"""Public web search observations are evidence, not executable betting orders."""
import json
from datetime import UTC, datetime, timedelta

import pytest

from scripts.walters_web_research import _one, audit

NOW = datetime(2026, 10, 10, 22, 40, tzinfo=UTC)
OBSERVED = NOW - timedelta(minutes=25)
KICKOFF = NOW + timedelta(hours=16)
GID = "2026_05_BAL_ATL"


def forecast():
    return {
        "spec": "walters_key_number_forward_v1",
        "release_state": "SHADOW",
        "sport": "nfl",
        "season": 2026,
        "game_id": GID,
        "home_team": "ATL",
        "away_team": "BAL",
        "kickoff": KICKOFF.isoformat(),
        "captured_at": (OBSERVED - timedelta(hours=2)).isoformat(),
        "sportsbook_odds_used": False,
        "fixed_alpha": 1.0,
        "independent_home_margin": -0.84,
        "model_mean": 0.15,
        "model_sigma": 13.,
        "key_multipliers": {"3": 2.2, "7": 1.4},
    }


def observation():
    return {
        "spec": "walters_web_research_observation_v1",
        "observation_id": "nfl_20261010_test_fanduel",
        "sport": "nfl",
        "observed_at": OBSERVED.isoformat(),
        "source_kind": "public_webpage_read",
        "source_url": "https://fanduel.com/research/nfl-week-5-schedule-odds-for-every-game",
        "publisher": "FanDuel Research",
        "book": "FanDuel",
        "publisher_is_official_book": True,
        "verified_executable_sportsbook_quote": False,
        "page_price_updated_at": None,
        "quotes": [{
            "game_id": GID, "home_team": "ATL", "away_team": "BAL",
            "kickoff": KICKOFF.isoformat(),
            "home_spread": -3.5, "away_spread": 3.5,
            "home_american_odds": +100, "away_american_odds": -122,
            "quote_context": "Two separate opposing prices displayed under the exact game.",
        }],
    }


def paths(tmp_path, doc=None):
    fd, wd = tmp_path / "forecasts", tmp_path / "web_observations"
    fd.mkdir()
    wd.mkdir()
    (fd / (GID + ".json")).write_text(json.dumps(forecast()))
    document = doc or observation()
    (wd / (document["observation_id"] + ".json")).write_text(
        json.dumps(document)
    )
    return fd, wd


def published(path):
    if path.parent.name == "forecasts":
        return OBSERVED - timedelta(hours=1)
    return OBSERVED + timedelta(minutes=1)


def test_web_read_official_public_page_against_prior_immutable_forecast(tmp_path):
    fd, wd = paths(tmp_path)
    view = audit(fd, wd, now=NOW, commit_lookup=published)
    assert view["frozen_original_forecasts"] == 1
    assert view["accepted_paired_web_spreads"] == 1
    row = view["research_comparisons"][0]
    assert row["book"] == "FanDuel"
    assert row["home_spread"] == -3.5
    assert row["models"]["key_number"]["away"]["exact_american_odds"] == -122
    assert row["models"]["key_number"]["away"]["push"] == 0
    assert row["source_price_timestamp_verified"] is False
    assert row["book_execution_independently_verified"] is False
    assert view["historical_profitable_edge_proven"] is False
    assert view["actual_wagers"] == 0
    assert view["betting_authorized"] is False


def test_unpublished_model_cannot_qualify_observation(tmp_path):
    fd, wd = paths(tmp_path)
    result = audit(
        fd, wd, now=NOW,
        commit_lookup=lambda path: OBSERVED + timedelta(minutes=2),
    )
    assert result["accepted_paired_web_spreads"] == 0
    assert result["blocked_observations"]
    assert result["betting_authorized"] is False


def test_uncommitted_web_page_is_not_backfilled(tmp_path):
    fd, wd = paths(tmp_path)
    result = audit(
        fd, wd, now=NOW,
        commit_lookup=lambda path: published(path)
        if path.parent.name == "forecasts" else None,
    )
    assert result["accepted_paired_web_spreads"] == 0
    assert "WEB_CAPTURE_NOT_ORIGINALLY_COMMITTED_AFTER_READ" in str(
        result["blocked_observations"]
    )


def test_invalid_sportsbook_pair_is_quarantined(tmp_path):
    obj = observation()
    obj["quotes"][0]["away_spread"] = 4.5
    fd, wd = paths(tmp_path, obj)
    result = audit(fd, wd, now=NOW, commit_lookup=published)
    assert result["accepted_paired_web_spreads"] == 0
    assert "SIDES_DIFFERENT_HANDICAP" in str(result["blocked_observations"])


def test_injected_web_page_update_is_not_a_quote_timestamp():
    obj = observation()
    obj["page_price_updated_at"] = OBSERVED.isoformat()
    with pytest.raises(ValueError, match="NO_INDEPENDENT_PRICE_UPDATE_SOURCE"):
        _one(forecast(), obj["quotes"][0], obj, now=NOW)


def test_publishers_must_match_named_book():
    obj = observation()
    obj["source_url"] = "https://other-domain.example/fake-odds"
    with pytest.raises(ValueError, match="SITE_NOT_ASSOCIATED"):
        _one(forecast(), obj["quotes"][0], obj, now=NOW)
    obj["source_url"] = "http://fanduel.com/research/fake"
    with pytest.raises(ValueError, match="INVALID_DIRECT_HTTPS_URL"):
        _one(forecast(), obj["quotes"][0], obj, now=NOW)


def test_wrong_team_wrong_game_and_future_quote_are_blocked():
    obj = observation()
    q = obj["quotes"][0]
    q["away_team"] = "NYG"
    with pytest.raises(ValueError, match="MISMATCH"):
        _one(forecast(), q, obj, now=NOW)
    q["away_team"] = "BAL"
    obj["observed_at"] = (KICKOFF + timedelta(minutes=1)).isoformat()
    with pytest.raises(ValueError, match="OUTSIDE_PREGAME_FORECAST_WINDOW"):
        _one(forecast(), q, obj, now=KICKOFF + timedelta(minutes=4))


def test_always_no_bet_even_if_model_estimates_positive_ev(tmp_path):
    fd, wd = paths(tmp_path)
    r = audit(fd, wd, now=NOW, commit_lookup=published)
    row = r["research_comparisons"][0]
    assert row["status"] == "PUBLIC_WEB_RESEARCH_ONLY_NO_EXECUTABLE_EDGE"
    assert not row["wager_authorized"]
    assert not row["betting_authorized"]
    assert all("unshrunk_model_ev_per_unit" in row["models"][model]["away"]
               for model in ("discrete_gaussian", "key_number"))


def test_public_article_not_an_accessible_sportsbook_quote():
    obj = observation()
    obj["verified_executable_sportsbook_quote"] = True
    with pytest.raises(ValueError, match="MUST_BE_RESEARCH_ONLY"):
        _one(forecast(), obj["quotes"][0], obj, now=NOW)


def test_duplicate_game_is_not_counted_twice(tmp_path):
    obj = observation()
    obj["quotes"].append(dict(obj["quotes"][0]))
    fd, wd = paths(tmp_path, obj)
    report = audit(fd, wd, now=NOW, commit_lookup=published)
    assert report["accepted_paired_web_spreads"] == 1
    assert "DUPLICATE_SOURCE_BOOK_OBSERVATION" in str(
        report["blocked_observations"]
    )


def test_impossible_american_odds_are_rejected():
    obj = observation()
    obj["quotes"][0]["away_american_odds"] = -10
    with pytest.raises(ValueError):
        _one(forecast(), obj["quotes"][0], obj, now=NOW)


def test_no_source_evidence_no_advisory():
    obj = observation()
    obj["quotes"][0]["quote_context"] = ""
    with pytest.raises(ValueError, match="MISSING_HUMAN_READ_WEB_EVIDENCE"):
        _one(forecast(), obj["quotes"][0], obj, now=NOW)
