"""Never convert unattributed collector-time snapshots to account BetMGM wagers."""
from copy import deepcopy
from hashlib import sha256
import json
import pytest
from scripts.nfl_moneyline_forward_public_quote_shadow import pair_originals


def original():
    return {
        "model":"NFL_TWO_WAY_MONEYLINE_ELO_RESEARCH_V1",
        "decision":"PASS","wager":False,
        "bookmaker_account_acceptance":False,
        "game_id":"2026_05_BUF_LA",
        "first_observed_at_utc":"2026-10-12T04:33:00Z",
        "kickoff_utc":"2026-10-13T00:15:00Z",
        "p_home_win":.54797,"p_away_win":.44803,"p_tie_push":.004,
        "conditional_p_home_win_given_no_tie":.54797/.996,
    }


def capture(*,at="2026-10-12T05:00:00Z",home=-185,away=150,
            bookmaker="BetMGM",side_first="home"):
    first=home if side_first=="home" else away
    second=away if side_first=="home" else home
    x={
        "source_provenance":"THIRD_PARTY_PUBLIC_SCOREBOARD_CAPTURE_ONLY",
        "is_account_accepted_betmgm_quote":False,
        "is_authorized_betting_quote":False,
        "event_id":"2026_05_BUF_LA",
        "kickoff_utc":"2026-10-13T00:15:00Z",
        "collector_observed_utc":at,
        "source_url":"https://api.actionnetwork.com/web/v2/scoreboard/nfl",
        "quotes":[{
            "book_as_attributed_by_aggregator":bookmaker,
            "market":"moneyline",
            "first_side":side_first,"second_side":"away" if side_first=="home" else "home",
            "first_line":None,"second_line":None,
            "first_american_odds":first,
            "second_american_odds":second,
            "source_original_book_offer_update_utc":None,
            "offer_origin_proven":False,
            "account_acceptance_verifiable":False,
        }],
    }
    x["snapshot_sha256"]=sha256(json.dumps(
        x,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return x


def test_quote_after_freeze_is_descriptive_not_ev():
    x=pair_originals([original()],[capture()])
    assert x["postmodel_first_capture_game_pairs"]==1
    row=x["same_contract_research_only"][0]
    assert row["book_original_offer_update_time"] is None
    assert row["estimated_true_EV"] is None
    assert row["validated_profit_edge"] is False
    assert x["wager_recommendations"]==[]
    assert x["decision"]=="PASS"
    assert row["aggregator_de_vig_home_conditional"]==pytest.approx(
        (185/285)/((185/285)+(100/250)),abs=1e-8)


def test_ordering_prohibits_quote_before_original_freeze():
    x=pair_originals([original()],[capture(at="2026-10-12T02:51:00Z")])
    assert x["postmodel_first_capture_game_pairs"]==0
    assert x["reject_reason_counts"]["quote_before_or_at_original_model"]==1


def test_postkickoff_and_kickoff_mismatch_rejected():
    x=pair_originals([original()],[capture(at="2026-10-13T02:00:00Z")])
    assert x["postmodel_first_capture_game_pairs"]==0
    q=capture()
    q["kickoff_utc"]="2026-10-14T00:15:00Z"
    q["snapshot_sha256"]=sha256(json.dumps(
        {k:v for k,v in q.items() if k!="snapshot_sha256"},
        sort_keys=True,separators=(",",":")).encode()).hexdigest()
    out=pair_originals([original()],[q])
    assert out["reject_reason_counts"]["source_after_kickoff_or_mismatched"]==1


def test_corrupt_aggregator_payload_rejected():
    q=capture()
    q["quotes"][0]["first_american_odds"]=-900
    with pytest.raises(ValueError,match="not a verified original"):
        pair_originals([original()],[q])


def test_trading_marker_impersonation_rejected_even_with_rehash():
    q=capture()
    q["is_account_accepted_betmgm_quote"]=True
    q["snapshot_sha256"]=sha256(json.dumps(
        {k:v for k,v in q.items() if k!="snapshot_sha256"},
        sort_keys=True,separators=(",",":")).encode()).hexdigest()
    with pytest.raises(ValueError,match="cannot be an authorized"):
        pair_originals([original()],[q])


def test_forecast_corruption_and_duplicate_rejected():
    g=original()
    with pytest.raises(ValueError,match="duplicate"):
        pair_originals([g,g],[])
    y=dict(g,p_home_win=.3)
    with pytest.raises(ValueError,match="probabilities"):
        pair_originals([y],[])


def test_wrong_book_is_not_fabricated_as_betmgm():
    x=pair_originals([original()],[capture(bookmaker="BetRivers")])
    assert x["postmodel_first_capture_game_pairs"]==0


def test_reversed_quote_sides_are_correctly_normalized():
    a=pair_originals([original()],[capture()])
    b=pair_originals([original()],[capture(side_first="away")])
    assert a["same_contract_research_only"][0]["aggregator_de_vig_home_conditional"]==(
        b["same_contract_research_only"][0]["aggregator_de_vig_home_conditional"])


def test_first_snapshot_per_event_not_multiple_pseudogames():
    a=capture(at="2026-10-12T08:00:00Z")
    b=capture(at="2026-10-12T07:00:00Z")
    x=pair_originals([original()],[a,b])
    assert x["postmodel_first_capture_game_pairs"]==1
    assert x["same_contract_research_only"][0]["aggregator_capture_utc"].startswith(
        "2026-10-12T07:00:00")


def test_genuine_unseen_forecast_does_not_get_backfilled():
    q=capture()
    q["event_id"]="2026_06_OTHER"
    q["snapshot_sha256"]=sha256(json.dumps(
        {k:v for k,v in q.items() if k!="snapshot_sha256"},
        sort_keys=True,separators=(",",":")).encode()).hexdigest()
    x=pair_originals([original()],[q])
    assert x["reject_reason_counts"]["no_frozen_forecast"]==1
    assert x["validated_profitable_bets"]==0
