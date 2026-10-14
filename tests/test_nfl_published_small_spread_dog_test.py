"""Predeclared close-underdog research invariants; no betting claims."""
from datetime import datetime,timedelta,timezone
from copy import deepcopy
import pytest
from scripts.nfl_published_small_spread_dog_test import run,paired_quotes

AT=datetime(2026,10,13,21,tzinfo=timezone.utc)
KICK=AT-timedelta(days=8)

def sample(week=4):
    out=[]
    base={"captured_at":(KICK-timedelta(hours=2)).isoformat(),
          "kickoff":KICK.isoformat(),"season":"2026","week":str(week),
          "game_id":"2026_04_ATL_NO","home_team":"NO","away_team":"ATL",
          "provider":"action_network","book":"BetMGM"}
    for market in ("moneyline","spread"):
        row={**base,"market_type":market,"first_side":"home","second_side":"away"}
        if market=="moneyline":
            row.update(first_line="",second_line="",first_american_odds="-135",second_american_odds="115")
        else:
            row.update(first_line="-2.5",second_line="2.5",first_american_odds="-110",second_american_odds="-110")
        out.append(row)
    return out

def score():
    return [{"season":2026,"week":4,"game_type":"REG",
             "game_id":"2026_04_ATL_NO","home_team":"NO","away_team":"ATL",
             "home_score":17,"away_score":20}]

def test_first_available_fixed_source_pair_and_mature_game():
    raw=sample();copy=deepcopy(raw)
    report=run(raw,score(),at=AT)
    assert raw==copy
    assert report["primary_small_spread"]["graded_games"]==1
    assert report["primary_small_spread"]["hypothetical_flat_stake_net_units"]==1.15
    assert report["individual_matured_observations"][0]["dog_american_moneyline"]==115
    assert report["betting_authorized"] is False
    assert report["proven_profitable_edge"] is False
    assert report["retrospective_analysis_not_original_pregame_strategy_decision"] is True

def test_needs_true_same_book_time_provider_and_spread_pair():
    rows=sample();rows[1]["book"]="Different sportsbook"
    report=run(rows,score(),at=AT)
    assert report["primary_small_spread"]["graded_games"]==0
    rows=sample();rows[1]["captured_at"]=(KICK-timedelta(hours=3)).isoformat()
    assert not paired_quotes(rows,at=AT)[0]
    rows=sample();rows[1]["first_line"]="+2.5" # dog has positive, favorite must negative
    assert not paired_quotes(rows,at=AT)[0]

def test_no_lookahead_and_no_pre_maturity_grading():
    rows=sample()
    rows[0]["captured_at"]=(KICK+timedelta(minutes=10)).isoformat()
    assert run(rows,score(),at=AT)["primary_small_spread"]["graded_games"]==0
    q=run(sample(),score(),at=KICK+timedelta(hours=12))
    assert q["primary_small_spread"]["graded_games"]==0
    assert len(q["pending_source_maturity"])==1

def test_no_fabricated_outcome_or_future_data():
    report=run(sample(),[],at=AT)
    assert report["primary_small_spread"]["graded_games"]==0
    assert len(report["pending_source_maturity"])==1
    report=run(sample(),[dict(score()[0],home_team="WRONG")],at=AT)
    assert report["primary_small_spread"]["graded_games"]==0
    report=run(sample(),[dict(score()[0],home_score=20)],at=AT)
    assert report["primary_small_spread"]["graded_games"]==0
