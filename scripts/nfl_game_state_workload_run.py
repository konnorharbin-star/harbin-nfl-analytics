"""Fit/tune the fixed quarter/state hypothesis and retain 2025 diagnostics."""
import argparse
from datetime import UTC,datetime
import hashlib
import json
from pathlib import Path
import random
from statistics import mean
import polars as pl
from nfl.game_state_workload import SPEC,aggregate,examples,fit_state_mix,predict,scores


def interval(scored,reps=3000):
    from collections import defaultdict
    games=defaultdict(list)
    for r in scored:games[r['game_id']].append(r['brier_delta'])
    values=[(sum(v),len(v)) for v in games.values()]
    rng=random.Random(20261014)
    boots=[]
    for _ in range(reps):
        sample=rng.choices(values,k=len(values));boots.append(sum(v[0] for v in sample)/sum(v[1] for v in sample))
    boots.sort()
    return [boots[int(.025*reps)],boots[int(.975*reps)]]


def run(schedules,plays):
    totals,excluded=aggregate(schedules,plays)
    rows,errors=examples(schedules,totals)
    train=[r for r in rows if r['season']==2024 and r['week']<=12]
    tune=[r for r in rows if r['season']==2024 and r['week']>12]
    test=[r for r in rows if r['season']==2025]
    if min(len(tune),len(test))<80:
        raise ValueError('INSUFFICIENT_CHRONOLOGICAL_BLOCK')
    model=fit_state_mix(train)
    choices=[]
    for strength in (32.,128.):
        for blend in (0.,.5,1.):
            scored=[scores(r,predict(r,model,strength,blend)) for r in tune]
            choices.append(dict(strength=strength,blend=blend,tuning_brier=mean(r['challenger_brier'] for r in scored)))
    selected=min(choices,key=lambda x:(x['tuning_brier'],x['blend'],x['strength']))
    model=fit_state_mix(train+tune)
    scored=[scores(r,predict(r,model,selected['strength'],selected['blend'])) for r in test]
    metrics={k:mean(r[k] for r in scored) for k in ('baseline_brier','challenger_brier','brier_delta',
             'baseline_log_loss','challenger_log_loss','baseline_dropback_mae','challenger_dropback_mae')}
    report=dict(spec=SPEC,status='CONDITIONAL_WORKLOAD_RESEARCH_NOT_MARKET_EDGE',
        train_block='2024 weeks5-12',tuning_block='2024 weeks13-18',diagnostic_block='2025 weeks5-18 previously inspected season',
        training_team_games=len(train),tuning_team_games=len(tune),diagnostic_team_games=len(test),
        diagnostic_distinct_games=len({r['game_id'] for r in scored}),
        selected=selected,tuning_candidates=choices,metrics=metrics,
        game_cluster_brier_delta_interval_95=interval(scored),
        source_exclusions=excluded,example_errors=errors,
        retrospective_source_publication_vintages_verified=False,
        sportsbook_prices_used=False,market_profitability_verified=False,estimated_ev=None,
        production_action='PASS',core_coupled_predictions_changed=False,
        interpretation='Regulation dropbacks include sacks and scrambles; not an official passing-attempt prop distribution. Quarter state predictions use own pregame fair margin, not realized target score or market spread.')
    frozen=dict(spec=SPEC,classes=list(map(int,model.classes_)),coefficients=model.coef_.tolist(),
                intercepts=model.intercept_.tolist(),strength=selected['strength'],blend=selected['blend'],
                trained_seasons=[2024],source_training='Retrospective free nflverse snapshots; no publication-vintage claim')
    return report,frozen


def load_model(frozen):
    from sklearn.linear_model import LogisticRegression
    import numpy as np
    model=LogisticRegression();model.classes_=np.array(frozen['classes']);model.coef_=np.array(frozen['coefficients']);model.intercept_=np.array(frozen['intercepts']);model.n_features_in_=model.coef_.shape[1]
    return model


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('study','forecast'))
    p.add_argument('--schedules',type=Path);p.add_argument('--pbp',type=Path)
    p.add_argument('--model',type=Path,default=Path('research/game_state_workload_2024_frozen.json'))
    p.add_argument('--out',type=Path,default=Path('reports/game_state_workload.json'))
    a=p.parse_args()
    import nflreadpy as nfl
    schedules=pl.read_parquet(a.schedules) if a.schedules else nfl.load_schedules([2023,2024,2025] if a.mode=='study' else [2025,2026])
    if a.mode=='study':
        plays=pl.read_parquet(a.pbp) if a.pbp else nfl.load_pbp([2024,2025])
        report,frozen=run(schedules,plays)
        raw=plays.select(sorted(__import__('nfl.game_state_workload',fromlist=['REQUIRED']).REQUIRED)).write_json()
        report['normalized_play_source_sha256']=hashlib.sha256(raw.encode()).hexdigest()
        report['schedule_source_sha256']=hashlib.sha256(schedules.filter(pl.col('season').is_between(2023,2025)).write_json().encode()).hexdigest()
        report['captured_utc']=datetime.now(UTC).isoformat()
        frozen['study_source_sha256']=report['normalized_play_source_sha256']
        a.model.parent.mkdir(parents=True,exist_ok=True);a.model.write_text(json.dumps(frozen,indent=2,sort_keys=True)+'\n')
    else:
        plays=pl.read_parquet(a.pbp) if a.pbp else nfl.load_pbp([2026])
        at=datetime.now(UTC);totals,excluded=aggregate(schedules,plays)
        rows,errors=examples(schedules,totals,seasons=(2026,),at=at)
        frozen=json.loads(a.model.read_text());model=load_model(frozen)
        forecasts=[{k:r[k] for k in ('game_id','team','season','week','kickoff_utc','fair_margin','source_prior_game_ids','latest_prior_week')}|predict(r,model,frozen['strength'],frozen['blend']) for r in rows]
        report=dict(spec=SPEC,captured_utc=at.isoformat(),model_file_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest(),
                    forecasts=forecasts,source_exclusions=excluded,errors=errors,
                    status='UNVALIDATED_WORKLOAD_SIDECAR_NO_PROP_PRICES',production_action='PASS',estimated_ev=None)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('forecasts','example_errors','tuning_candidates')},indent=2))

if __name__=='__main__':main()
