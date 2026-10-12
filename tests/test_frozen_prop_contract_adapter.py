"""Frozen player shadows can support diagnostic ONLY when same exact contract."""
import csv
import copy
import json
from pathlib import Path

import pytest

from scripts.frozen_prop_contract_adapter import load_frozen_forecasts, join

NOW="2026-10-11T20:08:00Z"
FIELDS=("version,first_observed_at_utc,game_id,season,week,kickoff_utc,team,player_id,"
        "player_name,position,market,synthetic_line,probability_over,probability_under,"
        "residual_sample_count,probability_basis,book_price,book,quoted_at,"
        "model_betting_ev,is_bet_recommendation").split(",")


def frozen():
    return dict(version="prop_empirical_error_cdf_v1",
                first_observed_at_utc="2026-10-09T19:00:00Z",
                game_id="2026_05_BUF_LA",season="2026",week="5",
                kickoff_utc="2026-10-13T00:15:00Z",team="BUF",
                player_id="00-0000001",player_name="SYNTHETIC PLAYER",
                position="WR",market="receiving_yards",synthetic_line="49.5",
                probability_over="0.59",probability_under="0.41",
                residual_sample_count="600",probability_basis="POSITION_2024_EMPIRICAL",
                book_price="",book="",quoted_at="",model_betting_ev="",
                is_bet_recommendation="False")


def write_csv(tmp_path,records):
    path=tmp_path/"predictions.csv"
    with path.open("w",newline="",encoding="utf8") as stream:
        writer=csv.DictWriter(stream,fieldnames=FIELDS)
        writer.writeheader();writer.writerows(records)
    return path


def quotes(line="49.5",player="00-0000001",quote_time="2026-10-11T20:00:00Z"):
    items=[]
    for side in ("OVER","UNDER"):
        items.append(dict(event_id="2026_05_BUF_LA",
          kickoff_utc="2026-10-13T00:15:00Z",
          market="PLAYER_RECEIVING_YARDS",period="FULL_GAME",
          participant=player,line=line,selection=side,
          american_odds="-110",bookmaker="BetMGM",region="US",
          snapshot_id="SYNTHETIC_UNVERIFIED",quote_utc=quote_time,
          captured_utc="2026-10-11T20:04:00Z",
          source_type="PUBLIC_BOOK_PAGE",source_url="https://example.org/artificial"))
    return {"decision_utc":NOW,"markets":[{"quotes":items,"selected":"OVER"}]}


def test_frozen_player_and_exact_line_become_unapproved_diagnostic(tmp_path):
    index=load_frozen_forecasts(write_csv(tmp_path,[frozen()]))
    r=join(quotes(),index)
    assert r["decision"]=="PASS" and not r["recommending_bet"]
    assert r["frozen_prop_matches"][0]["player_id"]=="00-0000001"
    m=r["markets"][0]["independent_probability"]
    assert m["p_win"]==.59
    assert m["p_push"]==0.0
    assert m["p_win_lower_95"] is None
    assert m["model_version_independently_approved"] is False
    assert r["best_verified_edge"] is None


def test_different_player_id_cannot_match(tmp_path):
    index=load_frozen_forecasts(write_csv(tmp_path,[frozen()]))
    r=join(quotes(player="different"),index)
    assert not r["frozen_prop_matches"]
    assert r["markets"][0]["independent_probability"] is None


def test_different_line_cannot_match(tmp_path):
    index=load_frozen_forecasts(write_csv(tmp_path,[frozen()]))
    r=join(quotes(line="50.5"),index)
    assert not r["frozen_prop_matches"]


def test_future_forecast_not_comparable_to_old_quote(tmp_path):
    row=frozen();row["first_observed_at_utc"]="2026-10-11T20:02:00Z"
    index=load_frozen_forecasts(write_csv(tmp_path,[row]))
    r=join(quotes(),index)
    assert not r["frozen_prop_matches"]
    assert r["markets"][0]["independent_probability"] is None


def test_postgame_frozen_record_rejected(tmp_path):
    row=frozen();row["first_observed_at_utc"]="2026-10-13T00:17:00Z"
    with pytest.raises(ValueError,match="post-kickoff"):
        load_frozen_forecasts(write_csv(tmp_path,[row]))


def test_non_normalized_probability_rejected(tmp_path):
    row=frozen();row["probability_under"]="0.9"
    with pytest.raises(ValueError,match="invalid frozen probability"):
        load_frozen_forecasts(write_csv(tmp_path,[row]))


def test_duplicate_player_threshold_rejected(tmp_path):
    row=frozen()
    with pytest.raises(ValueError,match="duplicate frozen"):
        load_frozen_forecasts(write_csv(tmp_path,[row,row]))


def test_cannot_override_injected_independent_model(tmp_path):
    index=load_frozen_forecasts(write_csv(tmp_path,[frozen()]))
    q=quotes()
    q["markets"][0]["independent_model"]={"model_version":"forged"}
    with pytest.raises(ValueError,match="overwrit"):
        join(q,index)


def test_missing_price_quote_means_no_made_up_bet(tmp_path):
    index=load_frozen_forecasts(write_csv(tmp_path,[frozen()]))
    x=join({"decision_utc":NOW,"markets":[]},index)
    assert x["frozen_prop_matches"]==[]
    assert x["best_verified_edge"] is None
    assert x["screened_contracts"]==0


def test_missing_bookmaker_price_does_not_make_bet(tmp_path):
    index=load_frozen_forecasts(write_csv(tmp_path,[frozen()]))
    q=quotes()
    q["markets"][0]["quotes"][0]["american_odds"]=None
    x=join(q,index)
    assert x["markets"][0]["status"]=="BLOCKED"
    assert x["best_verified_edge"] is None

def test_integer_line_fails_closed_instead_of_fake_zero_push(tmp_path):
    index=load_frozen_forecasts(write_csv(tmp_path,[frozen()]))
    result=join(quotes(line="49"),index)
    assert result["decision"]=="PASS"
    assert not result["frozen_prop_matches"]
    assert "no-push lines" in result["frozen_prop_unmatched"][0]["reason"]
    assert result["best_verified_edge"] is None


def test_synthetic_integer_forecast_row_rejected(tmp_path):
    row=frozen()
    row["synthetic_line"]="49"
    with pytest.raises(ValueError,match="no-push lines"):
        load_frozen_forecasts(write_csv(tmp_path,[row]))


def test_nan_synthetic_threshold_rejected(tmp_path):
    row=frozen()
    row["synthetic_line"]="NaN"
    with pytest.raises(ValueError,match="nonfinite"):
        load_frozen_forecasts(write_csv(tmp_path,[row]))
