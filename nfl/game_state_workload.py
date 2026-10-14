"""Public-method-inspired pregame quarter/state workload challenger.

Independent fair margin -> expected state mix -> prior conditional dropback
rates. No proprietary Peabody formula, bookmaker input, prop CDF or betting
promotion. All historical examples use only earlier source-mature games.
"""
from collections import Counter, defaultdict
from datetime import timedelta
import math
import numpy as np
from sklearn.linear_model import LogisticRegression
from nfl.coupled_nflverse_asof import kickoff_utc
from nfl.ratings import fit_pregame_fair_score

SPEC = 'NFL_QUARTER_STATE_WORKLOAD_RESEARCH_V1'
STATES = (-1, 0, 1)
REQUIRED = {'game_id','posteam','qtr','score_differential','qb_dropback',
            'rush_attempt','qb_kneel','qb_spike','play_id','pass_attempt','complete_pass'}


def aggregate(schedules, plays):
    """Regulation dropbacks/design runs only; exclude missing state and duplicates."""
    if not REQUIRED <= set(plays.columns):
        raise ValueError('MISSING_PLAY_STATE_COLUMNS')
    fixtures = {g['game_id']: g for g in schedules.to_dicts() if g['game_type']=='REG'}
    totals = defaultdict(lambda: dict(n=0,d=0,attempts=0,completions=0,rushes=0,states={s:[0,0] for s in STATES}))
    seen, exclusions = set(), Counter()
    for r in plays.select(sorted(REQUIRED)).iter_rows(named=True):
        g = fixtures.get(r['game_id'])
        if g is None or r['posteam'] not in (g['home_team'],g['away_team']):
            exclusions['unmatched_game_team'] += 1
            continue
        if r['qb_kneel']==1 or r['qb_spike']==1 or r['qtr'] not in (1,2,3,4):
            exclusions['kneel_spike_or_nonregulation'] += 1
            continue
        if r['qb_dropback']!=1 and r['rush_attempt']!=1:
            continue
        diff = r['score_differential']
        if diff is None or not math.isfinite(float(diff)) or r['play_id'] is None:
            exclusions['missing_state_identity'] += 1
            continue
        key = (r['game_id'],r['play_id'])
        if key in seen:
            raise ValueError('DUPLICATE_ELIGIBLE_PLAY')
        seen.add(key)
        state = 1 if diff>0 else -1 if diff<0 else 0
        t = totals[(r['game_id'],r['posteam'],int(r['qtr']))]
        t['n'] += 1;t['d'] += int(r['qb_dropback']==1)
        t['attempts'] += int(r['pass_attempt']==1)
        t['completions'] += int(r['complete_pass']==1)
        t['rushes'] += int(r['rush_attempt']==1)
        t['states'][state][0] += 1;t['states'][state][1] += int(r['qb_dropback']==1)
    return totals, dict(exclusions)


def profile(priors):
    if len(priors)<4:
        raise ValueError('INSUFFICIENT_PRIOR_TEAM_GAMES')
    result = {}
    for q in (1,2,3,4):
        vals = [p[q] for p in priors if q in p]
        if len(vals)!=len(priors):
            raise ValueError('INCOMPLETE_PRIOR_REGULATION_QUARTERS')
        n = sum(p['n'] for p in vals)
        d = sum(p['d'] for p in vals)
        if n<=0:
            raise ValueError('NO_PRIOR_PLAYS')
        result[q] = dict(n=n,d=d,volume=n/len(vals),
                         states={s:[sum(p['states'][s][0] for p in vals),sum(p['states'][s][1] for p in vals)] for s in STATES})
    return result


def examples(schedules, totals, seasons=(2024,2025), at=None):
    """Features exclude target/future weeks. Current targets require pre-kickoff at."""
    games = [g for g in schedules.to_dicts() if g['game_type']=='REG']
    records, errors, fitted = [], [], {}
    for year,week in sorted({(g['season'],g['week']) for g in games if g['season'] in seasons and g['week']>=5}):
        targets = [g for g in games if g['season']==year and g['week']==week]
        if at is not None:
            targets = [g for g in targets if at<kickoff_utc(g)<=at+timedelta(days=8)]
        else:
            targets = [g for g in targets if g.get('home_score') is not None and g.get('away_score') is not None]
        if not targets:
            continue
        cutoff = min(kickoff_utc(g) for g in targets)-timedelta(seconds=1) if at is None else at
        eligible = [g for g in games if g['season'] in (year-1,year) and
                    (g['season']<year or g['week']<week) and
                    g.get('home_score') is not None and g.get('away_score') is not None and
                    kickoff_utc(g)+timedelta(hours=72)<=cutoff]
        allowed = {g['game_id'] for g in eligible}|{g['game_id'] for g in targets}
        import polars as pl
        model = fit_pregame_fair_score(schedules.filter(pl.col('game_id').is_in(list(allowed))),year,week)
        fitted[(year,week)] = model
        for g in sorted(targets,key=lambda x:x['game_id']):
            fair = model.project(g['home_team'],g['away_team'])
            for team,margin in ((g['home_team'],fair.home_margin),(g['away_team'],-fair.home_margin)):
                previous = sorted([p for p in eligible if p['season']==year and team in (p['home_team'],p['away_team'])],key=kickoff_utc)[-6:]
                try:
                    past = [{q:totals[(p['game_id'],team,q)] for q in (1,2,3,4) if (p['game_id'],team,q) in totals} for p in previous]
                    historical = profile(past)
                    target = {q:totals[(g['game_id'],team,q)] for q in (1,2,3,4) if (g['game_id'],team,q) in totals}
                    if at is None and len(target)!=4:
                        raise ValueError('MISSING_TARGET_QUARTERS')
                    records.append(dict(game_id=g['game_id'],team=team,season=year,week=week,
                        kickoff_utc=kickoff_utc(g).isoformat(),fair_margin=margin,
                        home=int(team==g['home_team']),prior=historical,target=target,
                        source_prior_game_ids=[p['game_id'] for p in previous],
                        latest_prior_week=max(p['week'] for p in previous)))
                except (ValueError,KeyError) as exc:
                    errors.append(dict(game_id=g['game_id'],team=team,reason=str(exc)))
    return records, errors


def features(row,q):
    m=float(row['fair_margin'])/7
    return [m,row['home'],int(q==2),int(q==3),int(q==4),m*int(q==2),m*int(q==3),m*int(q==4)]


def fit_state_mix(records):
    x,y,w = [],[],[]
    for r in records:
        for q,t in r['target'].items():
            for state,(n,_) in t['states'].items():
                if n:
                    x.append(features(r,q));y.append(state);w.append(n/t['n'])
    if len(records)<80 or set(y)!=set(STATES):
        raise ValueError('INSUFFICIENT_STATE_TRAINING')
    model = LogisticRegression(C=1.0,max_iter=400)
    model.fit(x,y,sample_weight=w)
    return model


def predict(row,model,strength,blend):
    if strength<0 or not 0<=blend<=1:
        raise ValueError('INVALID_CHALLENGER_SETTINGS')
    base,challenger,volume = 0.,0.,0.
    quarters = []
    for q,past in sorted(row['prior'].items()):
        baseline = past['d']/past['n']
        occupancy = dict(zip(map(int,model.classes_),model.predict_proba([features(row,q)])[0]))
        rates = {}
        for state,(n,d) in past['states'].items():
            rates[state] = (d+strength*baseline)/(n+strength) if n+strength else baseline
        conditional = sum(occupancy[s]*rates[s] for s in STATES)
        projected = (1-blend)*baseline+blend*conditional
        base += baseline*past['volume'];challenger += projected*past['volume'];volume += past['volume']
        quarters.append(dict(quarter=q,expected_plays=past['volume'],
                             baseline_dropback_rate=baseline,challenger_dropback_rate=projected,
                             predicted_state_mix=occupancy,prior_state_dropback_rates=rates))
    return dict(baseline_dropback_rate=base/volume,challenger_dropback_rate=challenger/volume,
                expected_regulation_opportunities=volume,baseline_expected_dropbacks=base,
                challenger_expected_dropbacks=challenger,quarters=quarters)


def scores(row,pred):
    n=sum(q['n'] for q in row['target'].values());d=sum(q['d'] for q in row['target'].values())
    if n<=0:
        raise ValueError('MISSING_TARGET_OUTCOME')
    result = dict(game_id=row['game_id'],week=row['week'],team=row['team'],n=n,d=d)
    for label in ('baseline','challenger'):
        p=pred[label+'_dropback_rate']
        result[label+'_brier']=(d*(1-p)**2+(n-d)*p*p)/n
        result[label+'_log_loss']=-(d*math.log(max(p,1e-12))+(n-d)*math.log(max(1-p,1e-12)))/n
        result[label+'_dropback_mae']=abs(pred[label+'_expected_dropbacks']-d)
    result['brier_delta']=result['challenger_brier']-result['baseline_brier']
    return result
