"""Integer spread / total distribution diagnostics must conserve probability."""
import math
import pytest
from scripts.discrete_score_diagnostic import normal_probs,empirical_probs,audit

def synthetic():
    rows=[]
    for yr in (2023,2024,2025):
        for i in range(130):
            k=i%12-4
            rows.append({"season":yr,"week":5+i//13,"game_id":f"{yr}_{i}",
                         "model_margin":1.5,"model_total":42.0,
                         "actual_margin":float(k),"actual_total":float(33+i%21),
                         "market_margin":3.0,"market_total":41.0})
    return rows

def test_integer_three_has_positive_push_mass():
    w,p,l=normal_probs(3.0,3.0,0.,13.)
    assert p>0
    assert math.isclose(w+p+l,1.,abs_tol=1e-10)

def test_half_point_has_zero_push_mass():
    w,p,l=normal_probs(3.0,3.5,0.,13.)
    assert abs(p)<1e-10
    assert math.isclose(w+p+l,1.,abs_tol=1e-10)

def test_negative_spread_threshold():
    w,p,l=normal_probs(-3.0,-3.0,0.,13.)
    assert p>0 and 0<w<1 and 0<l<1

def test_empirical_mass_and_key_number():
    data=synthetic()
    normal=normal_probs(1.5,3.,0.,13.)
    empirical=empirical_probs(data[:260],"margin",1.5,3.,normal)
    assert math.isclose(sum(empirical),1.,abs_tol=1e-10)
    assert empirical[1]>0

def test_2025_does_not_affect_fit():
    data=synthetic()
    x=audit(data)
    changed=[dict(v) for v in data]
    for r in changed:
        if r["season"]==2025:
            r["actual_margin"]+=1000
    y=audit(changed)
    assert x["by_market"]["total"]==y["by_market"]["total"]
    assert not x["diagnostic_pristine"]
    assert x["decision"]=="PASS"
    assert not x["positive_ev_verified"]

def test_missing_archive_prices_fail():
    rows=synthetic()
    for r in rows:
        r["market_total"]=None
    with pytest.raises(ValueError,match="missing archived"):
        audit(rows)

def test_invalid_input_fails():
    with pytest.raises(ValueError):
        normal_probs(1.,3.,0.,0.)
