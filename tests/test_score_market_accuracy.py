"""Research accuracy audit fails closed on missing markets and data leakage."""
import copy
import polars as pl
import pytest
from scripts.score_market_accuracy import attach_archive,fit_market_weight,evaluate

def rows():
    out=[]
    for season in (2023,2024,2025):
        for i in range(160):
            out.append(dict(game_id=f"{season}_{i}",season=season,week=5+i//16,
                model_margin=1.,model_total=45.,actual_margin=float(i%9)-4,
                actual_total=float(35+i%21),market_margin=float(i%5)-2.,
                market_total=float(40+i%9)))
    return out

def test_archive_sign_is_home_margin_not_home_handicap():
    independent=[dict(game_id="A",season=2025,week=4,model_margin=1.,
                      model_total=44.,actual_margin=-2.,actual_total=39.)]
    schedule=pl.DataFrame({"game_id":["A"],"spread_line":[3.5],"total_line":[48.5]})
    a=attach_archive(independent,schedule)
    assert a[0]["market_margin"]==3.5
    assert a[0]["market_total"]==48.5
    assert a[0]["model_margin"]==1.

def test_change_2025_finals_cannot_retrain_blend_weight():
    original=rows()
    a=evaluate(copy.deepcopy(original))
    changed=copy.deepcopy(original)
    for r in changed:
        if r["season"]==2025:
            r["actual_total"]+=500
            r["actual_margin"]-=500
    b=evaluate(changed)
    assert a["market_blend_weight"]==b["market_blend_weight"]
    assert not a["2025_is_untouched_holdout"]
    assert a["decision"]=="PASS"

def test_missing_archive_pairs_rejected():
    original=rows()
    for r in original:
        r["market_total"]=None
    with pytest.raises(ValueError,match="insufficient"):
        evaluate(original)

def test_fit_only_2023_2024_and_bounded_weight():
    samples=[r for r in rows() if r["season"] in (2023,2024)]
    w=fit_market_weight(samples,"total")
    assert 0<=w<=1

def test_duplicates_rejected():
    original=rows()
    with pytest.raises(ValueError,match="duplicate"):
        evaluate(original+[original[0]])

def test_push_reporting_and_no_positive_ev():
    x=evaluate(rows())
    assert x["by_target"]["total"]["diagnostic_2025"]["paired"]==160
    assert x["positive_ev_validated"] is False
    assert x["market_quotes_time_verified"] is False
    assert x["independent_model_consumes_market"] is False
    assert x["normalized_score_and_archive_sha256"]

def test_invalid_archive_fields_fails_closed():
    with pytest.raises(ValueError,match="lacks archive"):
        attach_archive([],pl.DataFrame({"game_id":["A"]}))
