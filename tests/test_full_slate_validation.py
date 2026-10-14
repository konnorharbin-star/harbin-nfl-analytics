from copy import deepcopy
from datetime import UTC, datetime
import pytest
from nfl.full_slate_validation import freeze, grade
from nfl.coupled_research_journal import read_journal

AT = datetime(2026,10,14,12,tzinfo=UTC)
MATURE = datetime(2026,10,20,12,tzinfo=UTC)

def card():
    c = dict(game_id='2026_06_SEA_DEN', market='spread', side='away', line=3.5,
             american_odds=-110, opposite_american_odds=-110, book='Book',
             source_url='https://source.example', model_forecast_sha256='a'*64,
             raw_conditional_win_probability=.6, raw_push_probability=0,
             raw_loss_probability=.4, raw_conditional_ev_proxy=.6*100/110-.4,
             model_capture_utc='2026-10-14T11:55:00Z', observed_utc='2026-10-14T11:58:00Z')
    return dict(spec='NFL_FULL_SLATE_CARD_V1', generated_utc='2026-10-14T11:59:00Z',
                ranked_candidates=[c], games=[dict(game_id=c['game_id'], home_team='DEN', away_team='SEA',
                kickoff_utc='2026-10-16T00:15:00Z', best_candidate=c)])

def final(home=24,away=23):
    return dict(game_id='2026_06_SEA_DEN', game_type='REG', home_team='DEN', away_team='SEA',
                gameday='2026-10-15', gametime='20:15', home_score=home, away_score=away)

def capture(root, c=None, at=AT, sha='a'*64):
    return freeze(root, c or card(), at=at,run_id='123',code_sha='a'*40,model_sha=sha)

def test_capture_is_first_seen_and_hash_checked(tmp_path):
    assert capture(tmp_path)['new_primary_games']==1
    path=tmp_path/'primary.jsonl'; original=path.read_bytes()
    c=card();c['games'][0]['best_candidate']['line']=10
    assert capture(tmp_path,c)['repeated_games']==1
    assert path.read_bytes()==original
    path.write_text(path.read_text().replace('3.5','4.5'))
    with pytest.raises(ValueError, match='HASH_MISMATCH'):
        grade(tmp_path,[final()],at=MATURE)

def test_no_immature_outcome_or_economic_profit(tmp_path):
    capture(tmp_path)
    result=grade(tmp_path,[final()],at=AT)
    assert result['counts']['not_mature']==1
    assert result['mean_market_minus_model_brier'] is None
    assert result['economic_roi'] is None

def test_exact_away_spread_grading_and_idempotent_corrections(tmp_path):
    capture(tmp_path)
    result=grade(tmp_path,[final()],at=MATURE)
    assert result['decided_same_contract_market_comparisons']==1
    assert result['mean_market_minus_model_brier']==pytest.approx(.25-.16)
    assert grade(tmp_path,[final()],at=MATURE)['new_or_corrected_grade_events']==0
    assert grade(tmp_path,[final(30,10)],at=MATURE)['new_or_corrected_grade_events']==1
    rows=read_journal(tmp_path/'grades.jsonl')
    assert [r['threshold_outcome'] for r in rows]==['WIN','LOSS']
    assert all(r['economic_roi'] is None for r in rows)

def test_missing_or_identity_mismatched_final_remains_unknown(tmp_path):
    capture(tmp_path)
    assert grade(tmp_path,[],at=MATURE)['currently_graded_games']==0
    f=final();f['home_team']='KC'
    assert grade(tmp_path,[f],at=MATURE)['counts']['invalid_original_or_source']==1
    assert not (tmp_path/'grades.jsonl').exists()

def test_push_excluded_from_binary_comparison(tmp_path):
    c=card();c['games'][0]['best_candidate']['line']=3
    capture(tmp_path,c)
    r=grade(tmp_path,[final(26,23)],at=MATURE)
    assert r['currently_graded_games']==1
    assert r['decided_same_contract_market_comparisons']==0
    assert read_journal(tmp_path/'grades.jsonl')[0]['threshold_outcome']=='PUSH_OR_TIE'

@pytest.mark.parametrize('market,side,line,home,away,expected',[
    ('total','over',45.5,24,23,'WIN'),('total','under',45.5,24,23,'LOSS'),
    ('moneyline','home',None,24,23,'WIN'),('moneyline','away',None,24,23,'LOSS')])
def test_market_orientation(tmp_path,market,side,line,home,away,expected):
    c=card();c['games'][0]['best_candidate'].update(market=market,side=side,line=line)
    capture(tmp_path,c);grade(tmp_path,[final(home,away)],at=MATURE)
    assert read_journal(tmp_path/'grades.jsonl')[0]['threshold_outcome']==expected

def test_postkickoff_future_and_changed_model_blocked(tmp_path):
    assert capture(tmp_path,at=MATURE)['new_primary_games']==0
    c=card();c['generated_utc']='2026-10-15T12:00:00Z'
    with pytest.raises(ValueError,match='FUTURE_CARD'):
        capture(tmp_path,c)
    capture(tmp_path)
    with pytest.raises(ValueError,match='NEW_COHORT'):
        capture(tmp_path,sha='b'*64)

def test_missing_opposite_price_is_not_invented(tmp_path):
    c=card();del c['games'][0]['best_candidate']['opposite_american_odds']
    capture(tmp_path,c)
    r=grade(tmp_path,[final()],at=MATURE)
    assert r['currently_graded_games']==1
    assert r['decided_same_contract_market_comparisons']==0

def test_bad_source_does_not_reuse_old_valid_grade(tmp_path):
    capture(tmp_path);grade(tmp_path,[final()],at=MATURE)
    f=final();f['game_type']='POST'
    r=grade(tmp_path,[f],at=MATURE)
    assert r['decided_same_contract_market_comparisons']==0
