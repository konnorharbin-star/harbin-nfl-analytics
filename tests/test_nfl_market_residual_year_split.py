"""Check chronological Brier calibration and no fabricated wager authorization."""
from copy import deepcopy
import pytest
from scripts.nfl_market_residual_year_split import analyze

def _row(year,week,num,market):
    return {
      "season":str(year),"week":str(week),"game_id":f"{year}_{market}_{num}",
      "market_type":market,"side":"over" if market=="total" else "home",
      "price_stage":"espn_archived_open","entry_timestamp_verified":"false",
      "model_probability":"0.6","no_vig_probability":"0.5",
      "result":"win" if num%3==0 else "loss",
    }

def data():
    return [_row(year,1+i%17,i,market)
            for year in (2024,2025) for market in ("moneyline","spread","total")
            for i in range(130)]

def test_strict_time_split_with_no_promoted_wager():
    original=data()
    preserved=deepcopy(original)
    report=analyze(original)
    assert original==preserved
    assert report["claimed_profitable_edge"] is False
    assert report["betting_authorized"] is False
    assert report["2025_truly_untouched_confirmatory_holdout"] is False
    for m in ("moneyline","spread","total"):
        r=report["markets"][m]
        assert r["training_games"]==130
        assert r["diagnostic_games"]==130
        assert r["later_year_cluster_uncertainty"]["distinct_week_clusters"]==17
        assert r["weight_selected_from_training_only"] in (-.5,-.25,-.1,0,.1,.25,.5)

def test_archive_duplicate_guard():
    sample=data()
    sample.append(dict(sample[0]))
    with pytest.raises(ValueError,match="DUPLICATE_ARCHIVED"):
        analyze(sample)

def test_unverified_quote_source_cannot_be_reclassified_as_executable():
    sample=data()
    sample[0]["price_stage"]="unverified_custom"
    with pytest.raises(ValueError,match="NOT_SUPPORTED"):
        analyze(sample)

def test_missing_prior_year_cannot_be_a_training_free_edge():
    sample=[x for x in data() if x["season"]=="2025"]
    with pytest.raises(ValueError,match="INSUFFICIENT_ARCHIVE"):
        analyze(sample)
