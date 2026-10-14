from datetime import UTC, datetime
import json
import pytest
from nfl.independent_publication import build_bundle, write_bundle
from run_nfl_predictions import generate

AT=datetime(2026,10,14,12,tzinfo=UTC)


def dist(mean):
    return dict(mean=mean,p10=0,median=mean,p90=mean+10)


def fixture():
    f=dict(event_id='g',season=2026,week=6,home_team='DEN',away_team='SEA',
           captured_utc='2026-10-14T11:00:00Z',kickoff_utc='2026-10-16T00:15:00Z',
           content_sha256='hash',independent_home_points=23,independent_away_points=20)
    d={'DEN:points':dist(23.125),'SEA:points':dist(20.375),
       'TOTAL':dist(43.5),'HOME_MARGIN':dict(mean=2.75,p10=-10,median=3,p90=17)}
    roles=[]
    for team in ('DEN','SEA'):
        for metric,value in [('completions',10),('passing_yards',100),('rush_attempts',20),('rushing_yards',80)]:
            d[team+':'+metric]=dist(value)
        roles.append(dict(team=team,player_id=team+'p',role='WR'))
        for metric,value in [('receptions',10),('receiving_yards',100),('carries',20),('rushing_yards',80)]:
            d[team+':'+team+'p:'+metric]=dist(value)
    raw=dict(game_id='g',status='UNVALIDATED_JOINT_DRIVE_PLAYER_RESEARCH',
             frozen_football_score_input=f,forecast_sha256='hash',asof_utc=f['captured_utc'],
             distribution=d,home_win=.6,away_win=.35,tie=.05,draws=2000,seed=42)
    scan=dict(spec='NFL_COUPLED_DRIVE_PLAYER_SHADOW_SCAN_V1',capture_utc=f['captured_utc'],season=2026,week=6,games=[raw])
    replay=dict(spec='NFL_COUPLED_FROZEN_REPLAY_INPUTS_V1',capture_utc=f['captured_utc'],games=[dict(game_id='g',forecast=f,roles=roles)])
    return scan,replay


def test_own_scores_spread_total_and_player_means_are_coherent():
    scan,replay=fixture();b=build_bundle(scan,replay,at=AT,names={'DENp':'Receiver'})
    g=b['games'][0]
    assert g['predicted_score']==dict(home=23.125,away=20.375)
    assert g['fair_home_spread']==-2.75
    assert g['predicted_total']==43.5
    assert b['player_props'][0]['player_name']=='Receiver'
    assert b['player_props'][0]['projections']['receiving_yards']['mean']==100
    assert b['accuracy_vs_market']=='UNPROVEN' and b['sportsbook_prices_used'] is False
    assert b['qualified_bets']==0


def test_market_fields_cannot_change_own_predictions():
    scan,replay=fixture();before=build_bundle(scan,replay,at=AT)
    scan['sportsbook_spread']=99;scan['games'][0]['market_total']=300
    assert build_bundle(scan,replay,at=AT)==before


@pytest.mark.parametrize('kind',['score','player','identity','future','incomplete'])
def test_invalid_source_blocks_without_invented_prediction(kind):
    scan,replay=fixture()
    if kind=='score':scan['games'][0]['distribution']['TOTAL']['mean']=50
    if kind=='player':scan['games'][0]['distribution']['DEN:DENp:receptions']['mean']=9
    if kind=='identity':scan['games'][0]['forecast_sha256']='wrong'
    if kind=='future':scan['games'][0]['asof_utc']='2026-10-15T12:00:00Z'
    if kind=='incomplete':del scan['games'][0]['draws']
    b=build_bundle(scan,replay,at=AT)
    assert b['games_forecast']==0 and not b['player_props']
    assert b['games'][0]['predicted_score'] is None


def test_source_block_still_has_game_row_and_no_bogus_zero():
    scan,replay=fixture();scan['games'][0]=dict(game_id='g',status='SOURCE_OR_MODEL_BLOCKED',reason='missing inputs')
    b=build_bundle(scan,replay,at=AT)
    assert b['games_requested']==1 and b['games_forecast']==0
    assert b['games'][0]['fair_home_spread'] is None


def test_other_production_retained_in_accounting_but_not_named_players():
    scan,replay=fixture();d=scan['games'][0]['distribution']
    d['DEN:DENp:receiving_yards']=dist(90);d['DEN:OTHER:receiving_yards']=dist(10)
    b=build_bundle(scan,replay,at=AT)
    assert b['games_forecast']==1 and b['named_player_rows']==2
    assert sum(p['is_unallocated_bucket'] for p in b['player_props'])==1


def test_csv_json_html_outputs_preserve_numbers_and_escape_names(tmp_path):
    scan,replay=fixture();b=build_bundle(scan,replay,at=AT,names={'DENp':'<script>unsafe</script>'})
    out=tmp_path/'independent_forecasts.json';write_bundle(b,out)
    assert json.loads(out.read_text())['games'][0]['predicted_total']==43.5
    html=out.with_suffix('.html').read_text()
    assert '<script>unsafe</script>' not in html and '&lt;script&gt;' in html
    assert (tmp_path/'independent_game_predictions.csv').exists()
    assert (tmp_path/'independent_player_predictions.csv').exists()
    assert out.with_suffix('.png').exists()


def test_command_always_generates_game_and_player_bundle(monkeypatch,tmp_path):
    calls=[]
    monkeypatch.setattr('run_nfl_predictions.subprocess.run',lambda command,**kwargs:calls.append((command,kwargs)))
    generate(2026,6,tmp_path)
    assert len(calls)==3
    assert 'scripts.nfl_coupled_drive_player_research_run' in calls[0][0]
    assert '--week' in calls[0][0]
    assert 'scripts.nfl_independent_predictions' in calls[1][0]
    assert '--resolve-names' in calls[1][0]
    assert 'scripts.nfl_web_model_intake' in calls[2][0]
    assert all(c[1]['check'] for c in calls)


def test_canonical_command_can_preserve_single_json_stdout(monkeypatch,tmp_path):
    import subprocess
    calls=[]
    monkeypatch.setattr('run_nfl_predictions.subprocess.run',lambda command,**kwargs:calls.append(kwargs))
    generate(2026,6,tmp_path,quiet=True)
    assert all(c['stdout']==subprocess.DEVNULL for c in calls)
