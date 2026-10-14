"""Historical NFL underdog rule follows fixed published thresholds."""
from scripts.nfl_small_dog_historical_replication import analyze, parse

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
