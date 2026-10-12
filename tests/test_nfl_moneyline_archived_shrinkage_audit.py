"""Market-anchored moneyline probabilities are research proxies, never wagers."""
import pytest
from scripts.nfl_moneyline_archived_shrinkage_audit import (
    _pair, _metrics, _loss, blend, choose_weight, report
)

def base(y=1, p_market=.7, p_elo=.6, season=2020, gid="g"):
    return {"game_id":gid,"season":season,"week":6,"y":y,
        "market":p_market,"elo":p_elo}

def historical(season,gid,*,home=27,away=14,odds_home=-200,odds_away=170,
              elo=.55):
    return {"game_id":gid,"season":season,"week":6,
        "home_score":home,"away_score":away,
        "archive_home_moneyline":odds_home,
        "archive_away_moneyline":odds_away,
        "conditional_p_home_win_given_no_tie":elo}

def test_weight_zero_exactly_matches_market_probabilities():
    g=base()
    assert blend(g,0)==g["market"]
    assert blend(g,1)==g["elo"]
    assert blend(g,.5)==pytest.approx(.65)
    with pytest.raises(ValueError):
        blend(g,1.01)

def test_first_fit_uses_development_seasons_only():
    training=[base(gid=f"d{i}",season=2019, p_market=.95, p_elo=.05)
              for i in range(10)]
    future=[base(gid=f"t{i}",season=2025,p_market=.05,p_elo=.95)
            for i in range(50)]
    w,s=choose_weight(training+future)
    assert w==0 and s["train_games"]==10
    w2,s2=choose_weight(training)
    assert w==w2 and s["dev_selected_logloss"]==s2["dev_selected_logloss"]

def test_elo_can_be_tested_without_promoting_profit():
    training=[base(gid=f"d{i}",season=2022,p_market=.05,p_elo=.95)
              for i in range(10)]
    w,s=choose_weight(training)
    assert w==1
    m=_metrics(training,w)
    assert m["model_minus_market_brier"]<0
    assert m["validated_betting_edge"] is False
    assert m["accepted_BetMGM_US_account_quotes"]==0

def test_paired_rows_exclude_tie_and_unpriced():
    x=[historical(2024,"a"),
       historical(2024,"tie",home=17,away=17),
       historical(2024,"no_odds",odds_home=None)]
    p=_pair(x)
    assert len(p)==1 and p[0]["y"]==1
    assert 0<p[0]["market"]<1

def test_duplicate_exact_game_rejected():
    x=historical(2024,"dupe")
    with pytest.raises(ValueError,match="duplicate"):
        _pair([x,x])

def test_development_data_required():
    with pytest.raises(ValueError,match="no historical"):
        choose_weight([base(season=2025)])
    with pytest.raises(ValueError,match="invalid"):
        choose_weight([base()],weights=(-1,.5))

def test_same_cohort_market_and_candidate_scores():
    g=[base(y=1,p_market=.7,p_elo=.6),
       base(y=0,p_market=.3,p_elo=.4,gid="2")]
    m=_metrics(g,.25)
    assert m["n"]==m["unique_games"]==2
    assert m["model_brier"]==pytest.approx(.105625)
    assert m["market_brier"]==pytest.approx(.09)
    assert m["model_minus_market_brier"]>0

def test_report_validates_chronology_even_on_synthetic_prices():
    dev=[historical(2020,f"d{i}",elo=.52) for i in range(8)]
    validation=[historical(2023,f"v{i}",elo=.52) for i in range(100)]
    inspected=[historical(2024,"h",elo=.52)]
    report_data=report(dev+validation+inspected,min_dev_games=6)
    assert report_data["train"]["train_games"]==8
    assert report_data["2023_validation"]["n"]==100
    assert report_data["archive_market_not_an_executable_BetMGM_quote"]
    assert report_data["candidate_was_fit_on_2024_2025"] is False
    assert report_data["actual_wagers"]==0
    assert report_data["model_wager_recommendations"]==[]
    assert report_data["estimated_real_EV"] is None
    assert report_data["decision"]=="PASS"

def test_report_rejects_small_development_cohort():
    x=[historical(2020,"d",elo=.52)]
    with pytest.raises(ValueError,match="insufficient chronological"):
        report(x,min_dev_games=2)

def test_market_only_likelihood_honored():
    assert _loss(1,.9)<_loss(1,.1)
    assert _loss(0,.1)<_loss(0,.9)
