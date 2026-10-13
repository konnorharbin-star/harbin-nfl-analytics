"""External-rule replica: integrity gates and no future earnings assertions."""
from scripts.nfl_published_underdog_replication import analyze
import pytest

def row(year,gid,spread,home_odds,away_odds,home_points,away_points):
    return {"game_type":"REG","season":str(year),"week":"7","game_id":gid,
            "spread_line":str(spread),"home_moneyline":str(home_odds),
            "away_moneyline":str(away_odds),
            "home_score":str(home_points),"away_score":str(away_points)}

def test_short_dog_rule_and_full_game_result():
    values=[
      row(2024,"A",3,-160,140,14,21),
      row(2024,"B",-4,180,-225,10,20),
      row(2025,"C",-2.5,130,-150,20,14),
      row(2025,"D",8,-325,270,20,19),
    ]
    result=analyze(values)
    a=result["years"]["2024"]["spread_0_to_3"]
    assert a["games"]==1 and a["wins"]==1
    assert a["hypothetical_unit_roi"]==pytest.approx(1.4)
    b=result["years"]["2024"]["spread_gt_3_to_7"]
    assert b["games"]==1 and b["losses"]==1
    assert result["years"]["2025"]["spread_0_to_3"]["games"]==1
    assert result["betting_authorized"] is False
    assert result["expected_return_or_profit_in_future"] is None
    assert result["source_quote_origin_verified"] is False

def test_wrong_orientation_rejected_not_fake_edge():
    item=row(2024,"A",3,-120,-140,24,21)
    out=analyze([item])
    assert out["exclusions"]["spread_vs_moneyline_orientation_conflict"]==1
    assert out["distinct_games_included"]==0

def test_duplicate_game_id_fail_closed():
    sample=row(2024,"A",3,-140,120,24,21)
    with pytest.raises(ValueError,match="DUPLICATE"):
        analyze([sample,sample])

def test_pickem_and_missing_source_cannot_be_assumed_market_odds():
    a=row(2025,"A",0,-110,-110,10,14)
    b=row(2025,"B",2,-140,120,11,21)
    b["away_moneyline"]=""
    result=analyze([a,b])
    assert result["distinct_games_included"]==0
    assert result["exclusions"]["missing_or_pickem"]==2
