"""Pure chronological NFL prior-QB/point-form moneyline challenger safeguards."""
import json
from copy import deepcopy
import pytest
from scripts.nfl_moneyline_qb_efficiency_challenger import (
  FEATURES,RIDGE_ALPHA,VERSION,logistic,prior_features,fit,evaluate,
  freeze_first_seen,run,team
)

def game(gid,year,week,home,away,*,hs=None,aw=None):
  return {"game_id":gid,"season":year,"week":week,"game_type":"REG",
      "home_team":home,"away_team":away,
      "gameday":f"{year}-10-18","gametime":"13:00",
      "location":"Home","home_rest":7,"away_rest":7,
      "home_score":hs,"away_score":aw,
      "home_moneyline":-120,"away_moneyline":110}

def qb(gid,year,week,team_name,epa=10,pid="ID"):
  return {"game_id":gid,"season":year,"week":week,"team":team_name,
    "season_type":"REG","position":"QB","player_id":pid,
    "attempts":30,"sacks_suffered":2,"passing_epa":epa}

def test_current_week_scores_and_qb_stats_cannot_leak():
  s=[game("a",2024,5,"BUF","KC",hs=30,aw=10),
     game("b",2024,6,"BUF","DEN",hs=25,aw=14)]
  p=[qb("a",2024,5,"BUF",epa=9),qb("b",2024,6,"BUF",epa=999)]
  a=prior_features(s,p)
  b=prior_features([s[0],dict(s[1],home_score=100,away_score=0)],
                   [p[0],dict(p[1],passing_epa=-999)])
  assert a[1]["features"]==b[1]["features"]
  assert a[1]["features"][2]>0

def test_prior_result_affects_later_week_not_current_week():
  s=[game("a",2024,5,"BUF","KC",hs=30,aw=10),
     game("b",2024,6,"BUF","DEN",hs=25,aw=14)]
  a=prior_features(s,[])
  changed=prior_features([dict(s[0],home_score=10,away_score=30),s[1]],[])
  assert a[0]["features"]==changed[0]["features"]
  assert a[1]["features"][1] > changed[1]["features"][1]

def test_last_qb_is_proxy_and_first_week_no_qb():
  s=[game("a",2024,5,"BUF","KC",hs=30,aw=10),
     game("b",2024,6,"BUF","DEN")]
  rows=prior_features(s,[qb("a",2024,5,"BUF")])
  assert not rows[0]["home_qb_proxy_known"]
  assert rows[1]["home_qb_proxy_known"]
  assert rows[1]["features"][2] > 0
  assert rows[1]["player_starter_status_verified"] is False

def test_same_week_no_update_when_monday_unfinished():
  s=[game("a",2026,5,"BUF","KC",hs=40,aw=0),
    game("c",2026,5,"DEN","LA",hs=None,aw=None),
    game("b",2026,6,"BUF","LA")]
  rows=prior_features(s,[qb("a",2026,5,"BUF")])
  target=next(r for r in rows if r["game_id"]=="b")
  assert target["features"][1]==0
  assert not target["home_qb_proxy_known"]

def test_old_season_qb_not_assumed_new_starting_qb():
  s=[game("a",2023,17,"BUF","KC",hs=30,aw=10),
    game("b",2024,1,"BUF","KC")]
  rows=prior_features(s,[qb("a",2023,17,"BUF")])
  assert not rows[1]["home_qb_proxy_known"]
  assert rows[1]["features"][1]==0

def test_team_aliases_and_duplicate_game_fails_closed():
  assert team("OAK")=="LV"
  s=game("same",2026,5,"OAK","LAR")
  with pytest.raises(ValueError,match="duplicate"):
    prior_features([s,s],[])

def test_week_duplicate_team_prohibited():
  s=[game("a",2026,5,"BUF","KC"),game("b",2026,5,"BUF","LA")]
  with pytest.raises(ValueError,match="twice"):
    prior_features(s,[])

def training_rows(n=760):
  # Simulate development cohort, date groups irrelevant for direct logistic fit.
  out=[]
  for i in range(n):
    outcome=int(i%3!=0)
    out.append({"game_id":f"d{i}","season":2019+i%4,"week":7,
      "features":[1.,((i%13)-6)/10.,((i%7)-3)/10.,0.,0.,1.],
      "home_score":28 if outcome else 10,
      "away_score":10 if outcome else 28,
      "archive_home_moneyline":-110,"archive_away_moneyline":-110})
  return out

def test_market_free_qb_fit_uses_only_2019_2022():
  rows=training_rows()
  a,n=fit(rows,use_market_offset=False)
  future=[dict(rows[0],season=2023,home_score=0,away_score=99)]
  b,n2=fit(rows+future,use_market_offset=False)
  assert n==n2==len(rows)
  assert a==b
  assert len(a)==len(FEATURES)
  assert all(abs(v)<100 for v in a)

def test_market_anchored_diagnostic_is_distinct_from_market_free():
  rows=training_rows()
  a,_=fit(rows,use_market_offset=False)
  b,_=fit(rows,use_market_offset=True)
  out=evaluate(rows[:10],a,market_offset=False)
  out2=evaluate(rows[:10],b,market_offset=True)
  assert out["n"]==out2["n"]==10
  assert out["market_prices_account_accepted"]==0
  assert out2["approved_profit_edge"] is False

def test_2026_first_seen_frozen_shadow_immutable(tmp_path):
  coeff,_=fit(training_rows(),use_market_offset=False)
  s=[game("2026_06_BUF_KC",2026,6,"BUF","KC")]
  rows=prior_features(s,[])
  path=tmp_path/"original.jsonl"
  x=freeze_first_seen(rows,coefficients=coeff,tie_prior=.005,
     observed_utc="2026-10-12T11:00:00Z",path=path)
  y=freeze_first_seen(rows,coefficients=[a*10 for a in coeff],
     tie_prior=.005,observed_utc="2026-10-12T12:00:00Z",path=path)
  assert x["frozen_new"]==1 and y["frozen_new"]==0
  frozen=json.loads(path.read_text().strip())
  assert frozen["model"]==VERSION
  assert frozen["decision"]=="PASS" and not frozen["wager"]
  assert frozen["market_odds"] is None and frozen["bet_true_EV"] is None
  assert frozen["p_home_win"]+frozen["p_away_win"]+frozen["p_tie_push"]==pytest.approx(1)

def test_no_model_pre_kickoff_backfill(tmp_path):
  row=prior_features([game("a",2026,6,"BUF","KC")],[])
  out=freeze_first_seen(row,coefficients=[0.]*len(FEATURES),
    tie_prior=.005,observed_utc="2026-10-19T00:00:00Z",path=tmp_path/"shadow")
  assert out["frozen_new"]==0

def test_required_development_sample_and_utc():
  with pytest.raises(ValueError,match="insufficient"):
    fit(training_rows(8),use_market_offset=False)
  with pytest.raises(ValueError,match="UTC"):
    freeze_first_seen([],coefficients=[0.]*len(FEATURES),
      tie_prior=.005,observed_utc="2026-10-12T11:00:00",path="/tmp/synthetic-do-not-create")

def test_qb_duplicate_does_not_accidentally_duplicate_output():
  s=[game("a",2024,5,"BUF","KC",hs=30,aw=10),
      game("b",2024,6,"BUF","KC")]
  a=prior_features(s,[qb("a",2024,5,"BUF"),qb("a",2024,5,"BUF")])
  assert len(a)==2


def test_only_forecasts_within_ten_day_freeze_window(tmp_path):
  coeff,_=fit(training_rows(),use_market_offset=False)
  near=prior_features([game("near",2026,6,"BUF","KC")],[])
  far_game=game("far",2026,12,"BUF","KC")
  far_game["gameday"]="2026-11-22"
  far=prior_features([far_game],[])
  out=freeze_first_seen(near+far,coefficients=coeff,tie_prior=.005,
    observed_utc="2026-10-12T11:00:00Z",path=tmp_path/"window.jsonl",
    horizon_days=10)
  assert out["frozen_new"]==1
  assert out["first_seen_game_ids"]==["near"]
