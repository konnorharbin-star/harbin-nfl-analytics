"""Historical NFL underdog rule follows fixed published thresholds."""
from scripts.nfl_small_dog_historical_replication import analyze, parse
from copy import deepcopy

def games():
    sample=[]
    for season in (2022,2023,2024,2025):
        for n in range(20):
            sample.append({"game_id":f"{season}_{n}_A_B",
              "season":season,"week":1+n%18,"game_type":"REG",
              "home_team":"B","away_team":"A",
              "spread_line":2.5 if n<10 else 5,
              "home_moneyline":-130,"away_moneyline":115 if n<10 else 190,
              "home_score":10 if n%2==0 else 20,
              "away_score":20 if n%2==0 else 10})
    return sample

def test_historical_coverage_no_raw_betting_promotion():
    original=games()
    report=analyze(original)
    assert report["total"]["spread_le_3"]["games"]==40
    assert report["total"]["spread_gt_3_le_7"]["games"]==40
    assert report["economic_betting_edge_validated"] is False
    assert report["historical_source_price_offer_time_authenticated"] is False
    assert report["betting_authorized"] is False

def test_missing_and_ambiguous_prices_not_invented():
    cases=games()[:2]
    cases[0]["away_moneyline"]=None
    cases[1]["away_moneyline"]=-100
    assert not parse(cases)

def test_duplicated_event_raises():
    source=games()
    source.append(dict(source[0]))
    try:
        parse(source)
    except ValueError as error:
        assert "DUPLICATE" in str(error)
    else:
        assert False, "Duplicate outcome should fail"

def test_home_underdogs_are_included_with_absolute_spread_and_correct_result():
    source=games()[:2]
    source[0].update(spread_line=-2.5,home_moneyline=140,away_moneyline=-160,
                     home_score=24,away_score=21)
    preserved=deepcopy(source)
    rows=parse(source)
    assert source==preserved
    assert len(rows)==2
    assert rows[0]['underdog_side']=='home'
    assert rows[0]['point_spread']==2.5
    assert rows[0]['outcome']=='win'
    assert rows[0]['hypothetical_net_units']==1.4

def test_nonfinite_scores_fractional_scores_and_wrong_orientation_are_excluded():
    for updates in ({'spread_line':float('nan')},{'home_score':20.5},
                    {'spread_line':-2.5},{'home_score':float('inf')}):
        row=games()[0]
        row.update(updates)
        assert parse([row])==[]

def test_tie_is_not_silently_omitted_or_scored_as_win():
    row=games()[0]
    row.update(home_score=20,away_score=20)
    actual=parse([row])[0]
    assert actual['outcome']=='push' and actual['hypothetical_net_units']==0
