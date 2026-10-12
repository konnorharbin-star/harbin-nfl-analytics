"""No published editorial prop can become an accepted BetMGM ticket."""
from copy import deepcopy
import pytest

from scripts.first_party_prop_quote_bridge import research_match

OBSERVED = "2026-10-12T04:00:00Z"


def frozen(line=64.5, market="rushing_yards", player="GSIS_TEST_0001"):
    return {
        "version": "PROP_PARTICIPATION_CONDITIONAL_V2_RESEARCH",
        "recommended_bet": "False", "first_observed_at_utc": "2026-10-12T03:41:07Z",
        "game_id": "2026_05_BUF_LA", "player_id": player,
        "player_name": "Test Player", "market": market, "synthetic_line": str(line),
        "p_over": "0.59", "p_under": "0.41",
        "kickoff_utc": "2026-10-13T00:15:00Z", "bookmaker": "",
        "american_odds": "", "model_ev": "",
    }


def quote(line=69.5):
    return {
        "event_id": "2026_05_BUF_LA", "player_id": "GSIS_TEST_0001",
        "market": "rushing_yards", "line": line, "player_name": "Test Player",
        "player_id_crosswalk_verified": True,
        "player_id_crosswalk_source_url":
            "https://nflreadr.nflverse.com/reference/load_players.html",
        "period": "FULL_GAME", "bookmaker": "BetMGM",
        "source_kind": "FIRST_PARTY_EDITORIAL",
        "source_url":
            "https://sports.betmgm.com/en/blog/nfl/2-best-nfl-player-prop-bets-mnf-rams-vs-bills-week-5-bm16/",
        "source_publication_utc": None, "book_origin_quote_utc": None,
        "account_offer_verified": False, "kickoff_utc": "2026-10-13T00:15:00Z",
        "over_american": -115, "under_american": -115,
    }


def test_actual_69_5_threshold_cannot_use_64_5_or_79_5():
    r = research_match([frozen(64.5), frozen(79.5)],
                       {"quotes": [quote()]}, observed_utc=OBSERVED)
    assert r["decision"] == "PASS"
    assert r["exact_frozen_contract_matches"] == 0
    assert r["profitable_edges_verified"] == 0
    assert "NO_ORIGINAL_FROZEN_PROBABILITY_AT_EXACT_THRESHOLD_AND_PLAYER" in r["contracts"][0]["blockers"]


def test_only_dated_exact_model_match_has_research_gap_not_ev():
    q = quote()
    q["source_publication_utc"] = "2026-10-12T03:50:00Z"
    r = research_match([frozen(69.5)], {"quotes": [q]}, observed_utc=OBSERVED)
    x = r["contracts"][0]
    assert r["exact_matches_with_two_editorial_prices"] == 1
    assert x["no_vig_conditional_over_price_reference"] == .5
    assert x["difference_is_not_true_ev"] == .09
    assert x["estimated_true_ev"] is None
    assert x["source_identity_crosswalk_independently_authenticated"] is False
    assert not x["account_executable_price_verified"]
    assert x["decision"] == "PASS"


def test_unknown_second_price_fails_closed():
    q = quote(64.5)
    q["under_american"] = None
    r = research_match([frozen()], {"quotes": [q]}, observed_utc=OBSERVED)
    assert "TWO_SIDED_PRICE_MISSING" in r["contracts"][0]["blockers"]
    assert r["contracts"][0]["difference_is_not_true_ev"] is None


def test_different_player_does_not_use_fuzzy_name_join():
    r = research_match([frozen(69.5, player="OTHER")],
                       {"quotes": [quote()]}, observed_utc=OBSERVED)
    assert r["exact_frozen_contract_matches"] == 0


def test_spoofed_source_and_forged_original_time_rejected():
    q = quote(64.5)
    q["source_url"] = "https://notbetmgm.com/en/blog/article"
    with pytest.raises(ValueError, match="first-party"):
        research_match([frozen()], {"quotes": [q]}, observed_utc=OBSERVED)
    q = quote(64.5)
    q["book_origin_quote_utc"] = OBSERVED
    with pytest.raises(ValueError, match="impersonate"):
        research_match([frozen()], {"quotes": [q]}, observed_utc=OBSERVED)
    q = quote(64.5)
    q["source_publication_utc"] = "2026-10-12T05:00:00Z"
    with pytest.raises(ValueError, match="published after"):
        research_match([frozen()], {"quotes": [q]}, observed_utc=OBSERVED)


def test_duplicate_model_and_source_contracts_fail():
    with pytest.raises(ValueError, match="duplicate original"):
        research_match([frozen(), frozen()],
                       {"quotes": [quote()]}, observed_utc=OBSERVED)
    with pytest.raises(ValueError, match="duplicate sourced"):
        research_match([frozen()], {"quotes": [quote(), quote()]},
                       observed_utc=OBSERVED)


def test_integer_threshold_or_invalid_odds_fail():
    q = quote(69)
    with pytest.raises(ValueError, match="whole-number"):
        research_match([frozen()], {"quotes": [q]}, observed_utc=OBSERVED)
    q = quote(64.5)
    q["over_american"] = 0
    with pytest.raises(ValueError, match="outside valid"):
        research_match([frozen()], {"quotes": [q]}, observed_utc=OBSERVED)


def test_forecast_after_published_quote_cannot_be_backtested():
    q = quote(64.5)
    q["source_publication_utc"] = "2026-10-11T19:00:00Z"
    r = research_match([frozen()], {"quotes": [q]}, observed_utc=OBSERVED)
    x = r["contracts"][0]
    assert "FORECAST_CREATED_AFTER_EDITORIAL_PUBLICATION" in x["blockers"]
    assert x["difference_is_not_true_ev"] is None
    assert r["exact_matches_with_two_editorial_prices"] == 0
    assert r["profitable_edges_verified"] == 0


def test_identity_not_guessed_from_published_name():
    q = quote(64.5)
    q["player_id_crosswalk_verified"] = False
    r = research_match([frozen()], {"quotes": [q]}, observed_utc=OBSERVED)
    assert "GSIS_ID_CROSSWALK_UNVERIFIED" in r["contracts"][0]["blockers"]
    assert r["exact_matches_with_two_editorial_prices"] == 0
    q = quote(64.5)
    q["player_name"] = "Other Player"
    r = research_match([frozen()], {"quotes": [q]}, observed_utc=OBSERVED)
    assert "SOURCE_PLAYER_NAME_DIFFERS_FROM_FROZEN_PLAYER_ID" in r["contracts"][0]["blockers"]


def test_unstamped_publication_blocks_gap_even_with_exact_line():
    r = research_match([frozen()], {"quotes": [quote(64.5)]},
                       observed_utc=OBSERVED)
    x = r["contracts"][0]
    assert x["difference_is_not_true_ev"] is None
    assert "NO_VALID_ORIGINAL_FORECAST_BEFORE_TIMED_EDITORIAL" in x["blockers"]


def test_timed_publication_after_original_forecast_still_passes():
    q = quote(64.5)
    q["source_publication_utc"] = "2026-10-12T03:50:00Z"
    r = research_match([frozen()], {"quotes": [q]}, observed_utc=OBSERVED)
    assert r["contracts"][0]["difference_is_not_true_ev"] == .09
    assert r["decision"] == "PASS"


def test_future_frozen_source_not_used_before_original_time():
    q = quote(64.5)
    r = research_match([frozen()], {"quotes": [q]},
                       observed_utc="2026-10-12T03:35:00Z")
    assert r["exact_frozen_contract_matches"] == 0
    assert r["decision"] == "PASS"
