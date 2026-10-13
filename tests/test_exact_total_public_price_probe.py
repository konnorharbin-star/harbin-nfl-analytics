"""Exact archived odds/total accounting, simulated data never certified EV."""
from copy import deepcopy
from datetime import UTC,datetime,timedelta
import pytest

from nfl.exact_total_public_price_probe import compare,_decimal

NOW=datetime(2026,10,13,22,14,tzinfo=UTC)
FROZEN=(NOW-timedelta(hours=2)).isoformat()
QUOTED=(NOW-timedelta(minutes=2)).isoformat()
KICK=(NOW+timedelta(days=2)).isoformat()
GID="2026_06_SEA_DEN"
def data():
    forecast={"event_id":GID,"home_team":"DEN","away_team":"SEA",
              "kickoff_utc":KICK,"captured_utc":FROZEN,
              "content_sha256":"a"*64}
    original={"spec":"NFL_COUPLED_DRIVE_PLAYER_SHADOW_SCAN_V1",
       "season":2026,"week":6,"capture_utc":FROZEN,
       "games":[{"game_id":GID,"status":"UNVALIDATED_JOINT_DRIVE_PLAYER_RESEARCH",
         "draws":2000,"seed":19,"forecast_sha256":"a"*64,
         "frozen_football_score_input":deepcopy(forecast),
         "distribution":{"TOTAL":{"mean":42.0},"HOME_MARGIN":{"mean":-1.0}}}]}
    replay={"spec":"NFL_COUPLED_FROZEN_REPLAY_INPUTS_V1",
      "capture_utc":FROZEN,"season":2026,"week":6,
      "games":[{"game_id":GID,"forecast":forecast,"drives":[],"roles":[]}]}
    quote={"game_id":GID,"market":"NFL_FULL_GAME_TOTAL","period":"REGULATION_AND_OVERTIME",
      "home_team":"DEN","away_team":"SEA","line":40.5,
      "over_american":-110,"under_american":-104,
      "collector_observed_utc":QUOTED,"source_url":"https://www.oddschecker.com/us/football/nfl",
      "source_quote_origin_utc":None,"account_executable":False}
    return original,replay,[quote]

def fake_simulate(frozen,drives,roles,*,asof,draws,seed,queries):
    assert asof==datetime.fromisoformat(FROZEN)
    assert draws==2000 and seed==19
    assert queries==[{"metric":"TOTAL","line":40.5}]
    return {"forecast_sha256":"a"*64,
      "distribution":{"TOTAL":{"mean":42.0},"HOME_MARGIN":{"mean":-1.0}},
      "query_thresholds":[{"line":40.5,"metric":"TOTAL",
        "over":.52,"push":0,"under":.48}]}

def test_model_probability_and_raw_ev_are_research_only_and_not_bets():
    args=data()
    untouched=deepcopy(args)
    report=compare(*args,now=NOW,simulate=fake_simulate)
    assert args==untouched
    assert report["games_examined"]==1
    row=report["quote_pair_observations"][0]
    assert row["decision"]=="PASS_MODEL_NOT_CALIBRATED_AND_PUBLIC_QUOTE_UNEXECUTABLE"
    assert row["sides"]["over"]["model_raw_win_probability"]==.52
    assert row["sides"]["over"]["hypothetical_raw_model_ev_per_unit"]==pytest.approx(.52*(100/110)-.48)
    assert row["sides"]["under"]["hypothetical_raw_model_ev_per_unit"]==pytest.approx(.48*(100/104)-.52)
    assert report["positive_ev_betting_recommendations"]==0
    assert report["betting_authorized"] is False
    assert row["book_origin_and_same_book_pair_verified"] is False

def test_origin_after_kickoff_or_pregame_model_blocks():
    orig,replay,quotes=data()
    quotes[0]["collector_observed_utc"]=(NOW+timedelta(days=3)).isoformat()
    with pytest.raises(ValueError,match="NONPROSPECTIVE"):
        compare(orig,replay,quotes,now=NOW+timedelta(days=4),simulate=fake_simulate)
    orig,replay,quotes=data()
    quotes[0]["home_team"]="SEA"
    with pytest.raises(ValueError,match="WRONG_HOME_AWAY"):
        compare(orig,replay,quotes,now=NOW,simulate=fake_simulate)
    orig,replay,quotes=data()
    replay["games"][0]["forecast"]["content_sha256"]="b"*64
    with pytest.raises(ValueError,match="FROZEN_FORECAST"):
        compare(orig,replay,quotes,now=NOW,simulate=fake_simulate)

def test_quote_source_and_contract_period_not_silent():
    orig,replay,quotes=data()
    quotes[0]["source_quote_origin_utc"]=FROZEN
    with pytest.raises(ValueError,match="PUBLIC_QUOTE"):
        compare(orig,replay,quotes,now=NOW,simulate=fake_simulate)
    orig,replay,quotes=data()
    quotes[0]["period"]="FIRST_HALF"
    with pytest.raises(ValueError,match="MARKET_PERIOD"):
        compare(orig,replay,quotes,now=NOW,simulate=fake_simulate)
    orig,replay,quotes=data()
    quotes[0]["over_american"]=-95
    with pytest.raises(ValueError,match="INVALID_AMERICAN"):
        compare(orig,replay,quotes,now=NOW,simulate=fake_simulate)
    orig,replay,quotes=data()
    quotes[0]["account_executable"]=True
    with pytest.raises(ValueError,match="PUBLIC_QUOTE"):
        compare(orig,replay,quotes,now=NOW,simulate=fake_simulate)

def test_push_and_raw_positive_math_remain_unvalidated():
    def fake_push(*args,**kwargs):
        d=fake_simulate(*args,**kwargs)
        d["query_thresholds"][0].update(over=.65,under=.25,push=.1)
        return d
    orig,replay,quotes=data()
    quotes[0]["line"]=40
    def override(frozen,drives,roles,*,asof,draws,seed,queries):
        assert queries[0]["line"]==40
        return fake_push(frozen,drives,roles,asof=asof,draws=draws,seed=seed,
                         queries=[{"metric":"TOTAL","line":40.5}])
    report=compare(orig,replay,quotes,now=NOW,simulate=override)
    over=report["quote_pair_observations"][0]["sides"]["over"]
    assert over["model_raw_push_probability"]==.1
    assert over["hypothetical_raw_model_ev_per_unit"]==pytest.approx(.65*100/110-.25)
    assert over["hypothetical_raw_model_ev_per_unit"]>0
    assert over["validated_profitable_edge"] is False
    assert report["positive_ev_betting_recommendations"]==0
