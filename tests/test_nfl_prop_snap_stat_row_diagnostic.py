"""Player-stat-row presence != playing, DNP, prop outcome or void."""
from copy import deepcopy
import json
import pytest
from scripts.nfl_prop_snap_stat_row_diagnostic import (
    FEATURES,VERSION,make_cohorts,fit,predict,_metrics,cluster_interval,freeze,loss
)

def g(gid,year,week,*,played=True):
    return {"game_id":gid,"season":year,"week":week,"game_type":"REG",
        "home_team":"BUF","away_team":"LA",
        "home_score":24 if played else None,
        "away_score":17 if played else None}

def fake_games(raw):
    return [{"game_id":r["game_id"],"season":r["season"],"week":r["week"],
             "completed":r["home_score"] is not None,
             "home":r["home_team"],"away":r["away_team"],
             "kickoff":None} for r in raw]

def snap(gid,week,val,pid="PFR_A"):
    return {"game_type":"REG","game_id":gid,"week":week,"season":2024,
            "team":"BUF","pfr_player_id":pid,"offense_snaps":val}

def pred(gid,week,season=2024):
    return {"game_id":gid,"season":season,"week":week,
       "kickoff_utc":"2026-10-18T17:00:00Z",
       "team":"BUF","player_id":"GSIS_A","player_name":"Synthetic",
       "position":"RB","market":"rushing_yards","challenger_mean":35.,
       "baseline_three_game_mean":30.,"prior_season_games":3,
       "most_recent_seen_week":week-1}

def stub_forecast(games,stats,*,season,week,options,future_only_at=None):
    if week!=5:return []
    return [pred("target",5,season)]

def records(n=520,season=2024):
    return [{"season":season,"game_id":f"{season}-{j//12}",
             "player_id":f"p{j}","observed_stats_row_presence":int(j%7!=0),
             "features":(1.,(j%6+8)/15.,(j%5+8)/15.,.75,0.05)}
            for j in range(n)]

def test_training_only_2024_and_distinct_future_outcomes():
    a=records()
    beta,baseline=fit(a)
    assert len(beta)==len(FEATURES)
    assert 0<baseline<1
    assert 0<predict((1.,1.,1.,.5,0.),beta)<1
    with pytest.raises(ValueError,match="fit season"):
        fit(records(season=2025))
    with pytest.raises(ValueError,match="insufficient"):
        fit(records(n=12))

def test_stat_row_absent_label_not_active_or_void():
    d=records(520)
    beta,base=fit(d)
    out=_metrics(d,beta,base)
    assert out["n_player_games"]==520
    assert not out["snap_model_is_actual_DNP_or_void_probability"]
    assert not out["account_BetMGM_price_verified"]

def test_no_same_week_pfr_input_enters_feature_cohort(monkeypatch):
    import scripts.nfl_prop_snap_stat_row_diagnostic as module
    monkeypatch.setattr(module,"forecast_week",stub_forecast)
    raw=[g("one",2024,1),g("two",2024,2),g("three",2024,3),
         g("four",2024,4),g("target",2024,5)]
    players=[{"game_id":"four","player_id":"GSIS_A","team":"BUF"}]
    snaprows=[snap(k,w,20+10*w) for w,k in enumerate(("one","two","three","four"),1)]
    a,_=make_cohorts(fake_games(raw),players,raw,snaprows,
                      {"GSIS_A":"PFR_A"},season=2024)
    altered=snaprows+[snap("target",5,149)]
    b,_=make_cohorts(fake_games(raw),players,raw,altered,
                      {"GSIS_A":"PFR_A"},season=2024)
    assert len(a)==len(b)==1
    assert a[0]["features"]==b[0]["features"]
    assert a[0]["observed_stats_row_presence"]==0

def test_prior_player_row_missing_not_automatically_zero(monkeypatch):
    import scripts.nfl_prop_snap_stat_row_diagnostic as module
    monkeypatch.setattr(module,"forecast_week",stub_forecast)
    raw=[g(str(x),2024,x) for x in range(1,5)]+[g("target",2024,5)]
    snapped=[snap(str(x),x,35) for x in (1,3,4)]
    snapped+=[snap("2",2,12,pid="PFR_B")]
    a,_=make_cohorts(fake_games(raw),[],raw,snapped,{"GSIS_A":"PFR_A"},season=2024)
    assert len(a)==1
    assert a[0]["last_4_observed_snap_rows"]==3
    assert a[0]["features"][1]==pytest.approx(35/60)

def test_stats_row_observed_whether_or_not_snap_was_in_target_week(monkeypatch):
    import scripts.nfl_prop_snap_stat_row_diagnostic as module
    monkeypatch.setattr(module,"forecast_week",stub_forecast)
    raw=[g(str(x),2024,x) for x in range(1,5)]+[g("target",2024,5)]
    snapped=[snap(str(x),x,35) for x in range(1,5)]
    outcomes=[{"game_id":"target","player_id":"GSIS_A","team":"BUF"}]
    a,_=make_cohorts(fake_games(raw),outcomes,raw,snapped,
                      {"GSIS_A":"PFR_A"},season=2024)
    assert a[0]["observed_stats_row_presence"]==1

def test_2026_original_first_seen_can_never_be_rewritten(tmp_path):
    a=records(520)
    beta,_=fit(a)
    original={"game_id":"2026_06_BUF_LA","player_id":"GSIS_A",
      "team":"BUF","player_name":"Synthetic","season":2026,"week":6,
      "kickoff_utc":"2026-10-18T17:00:00Z",
      "observed_stats_row_presence":None,
      "features":(1.,.6,.8,.75,.15)}
    path=tmp_path/"proxy.jsonl"
    x=freeze([original],beta,observed_utc="2026-10-12T11:40:00Z",path=path)
    y=freeze([dict(original,features=(1.,1.,1.,1.,1.))],[0.]*len(beta),
             observed_utc="2026-10-12T11:41:00Z",path=path)
    assert x["new"]==1 and y["new"]==0
    row=json.loads(path.read_text().strip())
    assert row["version"]==VERSION
    assert row["p_player_active"] is None
    assert row["p_prop_void"] is None
    assert row["estimated_true_EV"] is None and not row["wager"]
    assert row["frozen_feature_vector"]==[1.,.6,.8,.75,.15]

def test_post_kickoff_and_post_ten_days_must_not_freeze(tmp_path):
    original={"game_id":"id","player_id":"GSIS_A","team":"BUF",
      "player_name":"Synthetic","season":2026,"week":6,
      "kickoff_utc":"2026-11-18T17:00:00Z",
      "observed_stats_row_presence":None,
      "features":(1.,.6,.8,.75,.15)}
    path=tmp_path/"none.jsonl"
    out=freeze([original],[0.]*len(FEATURES),
               observed_utc="2026-10-12T12:00:00Z",path=path)
    assert out["new"]==0
    original["kickoff_utc"]="2026-10-11T17:00:00Z"
    assert freeze([original],[0.]*len(FEATURES),
                  observed_utc="2026-10-12T12:00:00Z",path=path)["new"]==0

def test_no_future_target_labels_accidentally_frozen(tmp_path):
    with pytest.raises(ValueError,match="target outcome"):
        freeze([{"game_id":"x","player_id":"p","team":"BUF",
          "season":2026,"week":6,"player_name":"X",
          "kickoff_utc":"2026-10-18T17:00:00Z",
          "observed_stats_row_presence":1,
          "features":(1.,1.,1.,1.,1.)}],
          [0.]*len(FEATURES),observed_utc="2026-10-12T12:00:00Z",
          path=tmp_path/"none")

def test_cluster_interval_deterministic_and_grouped_by_game():
    data=records(520)
    beta,base=fit(data)
    interval=cluster_interval(data,beta,base,draws=30)
    assert interval is not None and len(interval)==2
    assert interval[0]<=interval[1]

def test_original_target_contract_rejects_non_utc(tmp_path):
    with pytest.raises(ValueError,match="UTC"):
        freeze([],[],observed_utc="2026-10-12T12:00:00",
               path=tmp_path/"original")


def test_count_only_restricted_model_does_not_use_snap_magnitude():
    data=records(520)
    b,_=fit(data,allowed_features=(0,3))
    assert b[1]==0 and b[2]==0 and b[4]==0
    first=predict((1.,.1,.5,.75,-.4),b)
    second=predict((1.,2.,1.8,.75,.9),b)
    assert first==pytest.approx(second)

def test_restricted_features_require_intercept():
    with pytest.raises(ValueError,match="intercept"):
        fit(records(),allowed_features=(3,))

def test_game_cluster_incremental_comparison_is_deterministic():
    from scripts.nfl_prop_snap_stat_row_diagnostic import cluster_interval_comparison
    data=records(520)
    complete,_=fit(data)
    simple,_=fit(data,allowed_features=(0,3))
    ci=cluster_interval_comparison(data,complete,simple,draws=25)
    assert ci is not None and ci[0]<=ci[1]


def test_original_2024_beta_catalog_matches_first_2026_shadow(tmp_path):
    from scripts.nfl_prop_snap_stat_row_diagnostic import fixed_training_cohort
    import hashlib
    dev=records()
    beta,mean=fit(dev)
    catalog={"model":VERSION,"development_year":2024,
       "development_unique_player_games":len(dev),
       "development_stat_row_presence_rate":mean,
       "feature_names":FEATURES,"ridge_alpha":50.,
       "full_beta":beta,
       "full_beta_sha256":hashlib.sha256(json.dumps(beta,sort_keys=True).encode()).hexdigest()}
    file=tmp_path/"pinned.json"
    file.write_text(json.dumps(catalog))
    pinned,sha=fixed_training_cohort(beta,mean,dev,path=file)
    assert pinned==beta and sha==catalog["full_beta_sha256"]
    file.write_text(json.dumps(dict(catalog,full_beta=[0.]*len(beta))))
    with pytest.raises(ValueError,match="SHA"):
        fixed_training_cohort(beta,mean,dev,path=file)

def test_revised_training_cohort_cannot_change_original_coefficients(tmp_path):
    from scripts.nfl_prop_snap_stat_row_diagnostic import fixed_training_cohort
    import hashlib
    dev=records()
    beta,mean=fit(dev)
    catalog={"model":VERSION,"development_year":2024,
       "development_unique_player_games":len(dev),
       "development_stat_row_presence_rate":mean,
       "feature_names":FEATURES,"ridge_alpha":50.,
       "full_beta":beta,
       "full_beta_sha256":hashlib.sha256(json.dumps(beta,sort_keys=True).encode()).hexdigest()}
    file=tmp_path/"pinned.json"
    file.write_text(json.dumps(catalog))
    revised=records(n=521)
    newbeta,newmean=fit(revised)
    with pytest.raises(ValueError,match="do NOT silently refit"):
        fixed_training_cohort(newbeta,newmean,revised,path=file)
