from copy import deepcopy
from datetime import UTC, datetime
from types import SimpleNamespace
import numpy as np
import polars as pl
import pytest
import nfl.game_state_workload as mod
from scripts.nfl_game_state_workload_run import load_model


def prior():
    q=dict(n=300,d=160,states={-1:[100,90],0:[100,50],1:[100,20]})
    return mod.profile([{i:deepcopy(q) for i in (1,2,3,4)} for _ in range(4)])

class Mix:
    classes_=np.array([-1,0,1])
    def predict_proba(self,x):
        return np.array([[.1,.1,.8] if x[0][0]>0 else [.8,.1,.1]])

def row(margin=7):
    return dict(game_id='g',week=5,team='A',home=1,fair_margin=margin,prior=prior(),target={})

def test_leading_and_trailing_change_workload_with_valid_conservation():
    lead=mod.predict(row(7),Mix(),32,1)
    trail=mod.predict(row(-7),Mix(),32,1)
    assert lead['challenger_dropback_rate']<lead['baseline_dropback_rate']<trail['challenger_dropback_rate']
    assert 0<lead['challenger_expected_dropbacks']<lead['expected_regulation_opportunities']
    assert all(sum(q['predicted_state_mix'].values())==pytest.approx(1) for q in lead['quarters'])

def test_zero_weight_exactly_reproduces_baseline_and_target_outcome_unused():
    r=row();before=mod.predict(r,Mix(),32,0)
    r['target']={1:dict(n=99,d=99)}
    after=mod.predict(r,Mix(),32,0)
    assert before==after
    assert before['baseline_dropback_rate']==before['challenger_dropback_rate']
    assert before['baseline_expected_dropbacks']==before['challenger_expected_dropbacks']

def test_unknown_state_shrinks_to_baseline_without_inventing_zero():
    r=row()
    for p in r['prior'].values():p['states'][1]=[0,0]
    result=mod.predict(r,Mix(),32,1)
    assert all(q['prior_state_dropback_rates'][1]==q['baseline_dropback_rate'] for q in result['quarters'])
    with pytest.raises(ValueError,match='INSUFFICIENT'):
        mod.profile([])

def test_target_brier_is_play_outcome_proper_score():
    r=row();r['target']={1:dict(n=100,d=60)}
    p=dict(baseline_dropback_rate=.5,challenger_dropback_rate=.6,
           baseline_expected_dropbacks=50,challenger_expected_dropbacks=60)
    score=mod.scores(r,p)
    assert score['baseline_brier']==pytest.approx(.25)
    assert score['challenger_brier']==pytest.approx(.24)
    assert score['challenger_dropback_mae']==0

def test_duplicate_plays_rejected_and_unknown_score_excluded():
    schedules=pl.DataFrame([dict(game_id='g',game_type='REG',home_team='A',away_team='B')])
    r=dict(game_id='g',posteam='A',qtr=1,score_differential=0,qb_dropback=1,
           rush_attempt=0,qb_kneel=0,qb_spike=0,play_id=1,pass_attempt=1,complete_pass=1)
    with pytest.raises(ValueError,match='DUPLICATE'):
        mod.aggregate(schedules,pl.DataFrame([r,r]))
    r['score_differential']=None
    totals,errors=mod.aggregate(schedules,pl.DataFrame([r]))
    assert not totals and errors['missing_state_identity']==1

def test_examples_strictly_prior_mature_weeks(monkeypatch):
    fixtures=[];totals={}
    for w in range(1,8):
        fixtures.append(dict(game_id=f'g{w}',season=2024,week=w,game_type='REG',home_team='A',away_team='B',
                             home_score=24,away_score=20,kickoff_utc=f'2024-09-{w*3:02d}T17:00:00Z'))
        for team in ('A','B'):
            for q in (1,2,3,4):totals[(f'g{w}',team,q)]=dict(n=10,d=5,states={-1:[3,2],0:[4,2],1:[3,1]})
    monkeypatch.setattr(mod,'fit_pregame_fair_score',lambda *a:SimpleNamespace(project=lambda *b:SimpleNamespace(home_margin=3)))
    rows,_=mod.examples(pl.DataFrame(fixtures),totals,seasons=(2024,))
    for r in rows:
        assert r['latest_prior_week']<r['week']
        assert r['game_id'] not in r['source_prior_game_ids']
    prospective,_=mod.examples(pl.DataFrame(fixtures),totals,seasons=(2024,),at=datetime(2024,9,14,tzinfo=UTC))
    assert all(datetime.fromisoformat(r['kickoff_utc'])>datetime(2024,9,14,tzinfo=UTC) for r in prospective)

def test_frozen_numeric_model_roundtrip():
    records=[]
    for i in range(100):
        r=row((i%20)-10)
        r['target']={q:dict(n=20,states={-1:[5,4],0:[5,2],1:[10,3]}) for q in (1,2,3,4)}
        records.append(r)
    model=mod.fit_state_mix(records)
    frozen=dict(classes=list(map(int,model.classes_)),coefficients=model.coef_.tolist(),intercepts=model.intercept_.tolist())
    restored=load_model(frozen)
    assert restored.predict_proba([mod.features(row(),4)])==pytest.approx(model.predict_proba([mod.features(row(),4)]))
