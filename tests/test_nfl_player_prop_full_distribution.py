"""Full frozen 2024 residual distribution supports offered lines without refitting."""
import json
from copy import deepcopy
from datetime import datetime
import pytest
from scripts.nfl_player_prop_full_distribution import (
    VERSION,CALIBRATION_VERSION,hash_object,group_for,candidate_distributions,
    freeze_first_seen,exact_line_probabilities,load_or_create_calibration,
)
from nfl.prop_probability import over_probability

def catalog():
    d={"calibration_version":CALIBRATION_VERSION,
       "training_year":2024,"vectors":{
           "rushing_yards|RB":[-2+(4*i/199) for i in range(200)],
           "rushing_yards|ALL":[-2+(4*i/299) for i in range(300)],
       }}
    d["sha256"]=hash_object(d)
    return d

def forecast():
    return {"game_id":"2026_06_BUF_KC","season":2026,"week":6,
       "kickoff_utc":"2026-10-18T17:00:00Z","team":"BUF",
       "player_id":"GSIS_EXAMPLE_TEST","player_name":"Synthetic Back",
       "position":"RB","market":"rushing_yards","challenger_mean":64.0}

def original():
    c=catalog()
    s=candidate_distributions([forecast()],c)[0]
    return {"first_observed_at_utc":"2026-10-12T04:40:00Z",**s},c

def test_frozen_distribution_supports_non_grid_exact_halfpoint():
    x,c=original()
    result=exact_line_probabilities(x,c,69.5,observed_utc="2026-10-12T04:45:00Z")
    assert result["line"]==69.5
    assert result["decision"]=="PASS"
    assert result["estimated_true_ev"] is None
    assert result["p_push_conditional_on_official_stat_row"]==0
    assert result["p_over_conditional_on_official_stat_row"]==pytest.approx(
        over_probability(64.,69.5,"rushing_yards",c["vectors"]["rushing_yards|RB"]),abs=1e-9)

def test_multiple_exact_lines_share_unmodified_frozen_snapshot():
    s,c=original()
    a=exact_line_probabilities(s,c,64.5,observed_utc="2026-10-12T04:45:00Z")
    b=exact_line_probabilities(s,c,73.5,observed_utc="2026-10-12T04:45:00Z")
    assert a["p_over_conditional_on_official_stat_row"]>=b["p_over_conditional_on_official_stat_row"]
    assert s["original_point_mean"]==64.
    assert s["residual_catalog_sha256"]==c["sha256"]

def test_whole_number_push_explicit_and_total_probability_one():
    s,c=original()
    x=exact_line_probabilities(s,c,64,observed_utc="2026-10-12T04:45:00Z")
    probs=[x[key] for key in ("p_over_conditional_on_official_stat_row",
        "p_under_conditional_on_official_stat_row","p_push_conditional_on_official_stat_row")]
    assert all(0<=v<=1 for v in probs)
    assert sum(probs)==pytest.approx(1,abs=1e-7)
    assert probs[2]>=0

def test_zero_integer_line_nonnegative_stats_no_under():
    s,c=original()
    x=exact_line_probabilities(s,c,0,observed_utc="2026-10-12T04:45:00Z")
    assert x["p_under_conditional_on_official_stat_row"]==0
    assert x["p_push_conditional_on_official_stat_row"]>=0

def test_late_quote_origin_cannot_backfill_before_frozen_model():
    s,c=original()
    with pytest.raises(ValueError,match="forecast must precede"):
        exact_line_probabilities(s,c,69.5,observed_utc="2026-10-12T04:45:00Z",
                                 original_book_quote_utc="2026-10-11T20:00:00Z")

def test_query_after_kickoff_rejected():
    s,c=original()
    with pytest.raises(ValueError,match="as-of"):
        exact_line_probabilities(s,c,69.5,observed_utc="2026-10-19T00:00:00Z")

def test_no_catalog_hash_swap_possible():
    s,c=original()
    c["vectors"]["rushing_yards|RB"][0]=123.
    c["sha256"]=hash_object({k:v for k,v in c.items() if k!="sha256"})
    with pytest.raises(ValueError,match="does not match"):
        exact_line_probabilities(s,c,69.5,observed_utc="2026-10-12T04:45:00Z")

def test_predicted_stat_scale_swap_detected():
    s,c=original()
    s["original_scale"]=5
    with pytest.raises(ValueError,match="modified stat scale"):
        exact_line_probabilities(s,c,69.5,observed_utc="2026-10-12T04:45:00Z")

def test_invalid_stat_line_rejected():
    s,c=original()
    for line in (-1,69.25,float("nan")):
        with pytest.raises(ValueError,match="line"):
            exact_line_probabilities(s,c,line,observed_utc="2026-10-12T04:45:00Z")

def test_fallback_needs_adequate_2024_sample():
    c=catalog()
    assert group_for("rushing_yards","RB",c)=="rushing_yards|RB"
    assert group_for("rushing_yards","QB",c)=="rushing_yards|ALL"
    assert group_for("receptions","WR",c) is None

def test_freeze_is_first_seen_and_cannot_revise_point_mean(tmp_path):
    s,c=original()
    p=tmp_path/"original.jsonl"
    row={k:v for k,v in s.items() if k!="first_observed_at_utc"}
    at="2026-10-12T04:40:00Z"
    x=freeze_first_seen([row],observed_utc=at,path=p)
    row2=dict(row,original_point_mean=1.)
    y=freeze_first_seen([row2],observed_utc=at,path=p)
    assert x["new"]==1 and y["new"]==0
    saved=json.loads(p.read_text().strip())
    assert saved["original_point_mean"]==64.0

def test_missing_kickoff_no_backfill(tmp_path):
    s,c=original()
    row={k:v for k,v in s.items() if k!="first_observed_at_utc"}
    out=freeze_first_seen([row],observed_utc="2026-10-18T17:00:00Z",
                          path=tmp_path/"none.jsonl")
    assert out["new"]==0

def test_calibration_catalog_first_seen_hash_immutable(tmp_path):
    c=catalog()
    path=tmp_path/"catalog.json"
    path.write_text(json.dumps(c))
    result,created=load_or_create_calibration(path)
    assert not created and result["sha256"]==c["sha256"]
    c["vectors"]["rushing_yards|RB"].append(4.)
    path.write_text(json.dumps(c))
    with pytest.raises(ValueError,match="corrupted"):
        load_or_create_calibration(path)

def test_invalid_participant_does_not_create_wager():
    s,c=original()
    x=exact_line_probabilities(s,c,73.5,observed_utc="2026-10-12T04:45:00Z")
    assert x["participation_or_void_verified"] is False
    assert not x["betmgm_account_quote_accepted"]
    assert x["calibrated_market_edge"] is False
