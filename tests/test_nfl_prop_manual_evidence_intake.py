"""Synthetic manual-source fixtures; no bookmaker prices are authenticated."""
from datetime import datetime, timedelta, timezone
import json

import pytest

from scripts.nfl_prop_manual_evidence_intake import (
    prepare, append_candidates, run, hash_object,
    MAX_CAPTURE_AGE
)
from scripts.nfl_prop_exact_public_market_shadow import create_rows


def example():
    at = datetime(2026, 10, 12, 17, tzinfo=timezone.utc)
    iso = lambda d:d.isoformat()
    catalog={
        "training_year":2024,
        "vectors":{"passing_yards|QB": [-1.0, -0.5, 0.0, 0.5, 1.0]*25},
    }
    catalog["sha256"]=hash_object(catalog)
    f={
        "model":"NFL_PLAYER_PROP_FROZEN_FULL_DISTRIBUTION_V1",
        "game_id":"2026_06_SEA_DEN", "player_id":"00-TEST123",
        "player_name":"S.Darnold", "team":"SEA","market":"passing_yards",
        "position":"QB", "first_observed_at_utc":iso(at-timedelta(hours=2)),
        "kickoff_utc":iso(at+timedelta(days=3)),
        "original_point_mean":225.5, "original_scale":45*.25+.45*225.5,
        "residual_catalog_sha256":catalog["sha256"],
        "residual_group_key":"passing_yards|QB",
        "training_residual_count":125,
    }
    obs={
        "quote_id":"MANUAL_FIXTURE_01",
        "source_url":"https://example.org/allowed-manual-reference",
        "source_kind":"PUBLIC_MANUALLY_VIEWED",
        "source_reuse_permitted":True,
        "source_observed_utc":iso(at-timedelta(minutes=2)),
        "source_quote_origin_utc":None,
        "snapshot_id":"human-transcription-20261012",
        "game_id":"2026_06_SEA_DEN", "team":"SEA",
        "market":"passing_yards","period":"FULL_GAME",
        "player_display_name":"Sam Darnold",
        "bookmaker":"UNVERIFIED_PUBLIC_ATTRIBUTION",
        "book_region":"US-ATTRIBUTED",
        "line":219.5,
        "american_odds_over":-110,
        "american_odds_under":-112,
    }
    return obs,[f],catalog,iso(at)


def test_exact_initial_to_fullname_manual_resolution_is_pre_game_only():
    obs,f,cat,now=example()
    result=prepare([obs],f,now=now)
    assert len(result)==1
    quote=result[0]
    assert quote["player_id"]=="00-TEST123"
    assert quote["source_quote_origin_utc"] is None
    assert quote["source_origin_timestamp_claimed_verified"] is False
    assert quote["capture_method"]=="MANUAL"
    assert quote["validated_candidate_is_executable"] is False
    assert quote["sides"][0]["source_observed_utc"] == quote["sides"][1]["source_observed_utc"]
    assert quote["sides"][0]["line"]==quote["sides"][1]["line"]==219.5
    assert quote["manual_candidate_sha256"]==hash_object(
        {k:v for k,v in quote.items() if k!="manual_candidate_sha256"})
    joined=create_rows(result,f,cat,now=now)[0]
    assert joined["decision"]=="PASS" and joined["estimated_ev"] is None
    assert joined["origin_provenance"]=="CAPTURE_ONLY_ORIGIN_UNKNOWN"


def test_id_required_when_initial_and_surname_ambiguous():
    obs,f,cat,now=example()
    another={**f[0],"player_id":"00-ANOTHER"}
    with pytest.raises(ValueError,match="ambiguous"):
        prepare([obs,fake_observation_same_id(obs)],f+[another],now=now)
    obs["player_id"]="00-TEST123"
    assert prepare([obs],f+[another],now=now)[0]["player_id"]=="00-TEST123"


def fake_observation_same_id(x):
    return {**x,"quote_id":"another-test-id"}


@pytest.mark.parametrize("field,bad",[
    ("team","DEN"),("market","receiving_yards"),
    ("game_id","2026_04_SEA_DEN"),("player_display_name","Different Person"),
    ("player_id","00-WRONG"),
])
def test_wrong_identity_must_not_join(field,bad):
    obs,f,cat,now=example()
    obs[field]=bad
    with pytest.raises(ValueError,match="unresolved or ambiguous"):
        prepare([obs],f,now=now)


def test_no_postgame_or_old_backfill_allowed():
    obs,f,cat,now=example()
    late=(datetime.fromisoformat(obs["source_observed_utc"])+timedelta(hours=3)).isoformat()
    with pytest.raises(ValueError,match="older than ten minutes"):
        prepare([obs],f,now=late)
    future=(datetime.fromisoformat(f[0]["kickoff_utc"])+timedelta(minutes=1)).isoformat()
    with pytest.raises(ValueError,match="older than ten minutes"):
        prepare([obs],f,now=future)


def test_quote_time_after_capture_rejected():
    obs,f,cat,now=example()
    obs["source_quote_origin_utc"]=(datetime.fromisoformat(
        obs["source_observed_utc"])+timedelta(seconds=1)).isoformat()
    with pytest.raises(ValueError,match="origin follows"):
        prepare([obs],f,now=now)


def test_source_reuse_must_be_attested_even_when_publicly_visible():
    obs,f,cat,now=example()
    obs["source_reuse_permitted"]=False
    with pytest.raises(ValueError,match="permission"):
        prepare([obs],f,now=now)


def test_known_disallowed_site_cannot_be_republished():
    obs,f,cat,now=example()
    obs["source_url"]="https://edgelabs.bet/nfl/props"
    with pytest.raises(ValueError,match="forbid reuse"):
        prepare([obs],f,now=now)


def test_host_subdomain_block():
    obs,f,cat,now=example()
    obs["source_url"]="https://www.edgelabs.bet/nfl/props"
    with pytest.raises(ValueError,match="forbid reuse"):
        prepare([obs],f,now=now)


def test_non_https_quote_rejected():
    obs,f,cat,now=example()
    obs["source_url"]="http://example.org/allowed"
    with pytest.raises(ValueError,match="HTTPS"):
        prepare([obs],f,now=now)


def test_no_invented_book_update_time():
    obs,f,cat,now=example()
    assert prepare([obs],f,now=now)[0]["source_quote_origin_utc"] is None


def test_side_prices_are_validated_as_american_odds():
    obs,f,cat,now=example()
    obs["american_odds_over"]=0
    with pytest.raises(ValueError,match="American"):
        prepare([obs],f,now=now)


def test_duplicate_raw_ids_rejected():
    obs,f,cat,now=example()
    with pytest.raises(ValueError,match="duplicate"):
        prepare([obs,obs],f,now=now)


def test_same_quote_cannot_silently_change_first_seen_price(tmp_path):
    obs,f,cat,now=example()
    a=prepare([obs],f,now=now)
    out=tmp_path/"candidate.json"
    assert append_candidates(a,out)["new"]==1
    assert append_candidates(a,out)["new"]==0
    obs["american_odds_over"]=-120
    modified=prepare([obs],f,now=now)
    with pytest.raises(ValueError,match="rewrite"):
        append_candidates(modified,out)
    saved=json.loads(out.read_text())
    saved["records"][0]["line"]=150.5
    out.write_text(json.dumps(saved))
    with pytest.raises(ValueError,match="hash modified"):
        append_candidates([],out)


def test_empty_manual_source_creates_no_quotes(tmp_path):
    inp=tmp_path/"empty.json"
    inp.write_text('{"records":[]}')
    out=tmp_path/"candidate.json"
    report=run(input_path=inp,output_path=out,report_path=tmp_path/"report.json")
    assert report["strict_first_seen_candidates"]=={"previous":0,"new":0,"total":0}
    assert report["live_odds_fetched"] is False
    assert report["decision"]=="PASS" and report["true_ev"] is None
    assert not out.exists()



def test_manual_original_survives_future_game_kickoff_without_repricing(tmp_path):
    obs, frozen, catalog, now=example()
    manual=tmp_path/"manual.json"
    output=tmp_path/"candidates.json"
    f=tmp_path/"frozen.jsonl"
    c=tmp_path/"catalog.json"
    report=tmp_path/"report.json"
    manual.write_text(json.dumps({"records":[obs]}))
    f.write_text(json.dumps(frozen[0])+"\n")
    c.write_text(json.dumps(catalog))
    args=dict(input_path=manual,output_path=output,forecasts_path=f,
              catalog_path=c,report_path=report)
    first=run(**args,now=now)
    original=output.read_text()
    assert first["strict_first_seen_candidates"]["new"]==1
    assert first["new_source_observations_eligible_for_current_cutoff"]==1

    after=(datetime.fromisoformat(frozen[0]["kickoff_utc"])+
           timedelta(minutes=1)).isoformat()
    replay=run(**args,now=after)
    assert replay["strict_first_seen_candidates"]["new"]==0
    assert replay["historical_source_observations_verified_without_repricing"]==1
    assert replay["new_source_observations_eligible_for_current_cutoff"]==0
    assert output.read_text()==original


def test_modified_old_manual_source_rejected_without_changing_saved_record(tmp_path):
    obs, frozen, catalog, now=example()
    manual=tmp_path/"manual.json"
    output=tmp_path/"candidates.json"
    f=tmp_path/"frozen.jsonl"
    c=tmp_path/"catalog.json"
    manual.write_text(json.dumps({"records":[obs]}))
    f.write_text(json.dumps(frozen[0])+"\n")
    c.write_text(json.dumps(catalog))
    args=dict(input_path=manual,output_path=output,forecasts_path=f,
              catalog_path=c,report_path=tmp_path/"report.json")
    run(**args,now=now)
    old=output.read_text()
    obs["american_odds_under"]=-120
    manual.write_text(json.dumps({"records":[obs]}))
    with pytest.raises(ValueError,match="historic original"):
        run(**args,now=(datetime.fromisoformat(frozen[0]["kickoff_utc"])+
                        timedelta(hours=1)).isoformat())
    assert output.read_text()==old


def test_new_quote_cannot_be_inserted_after_kickoff_with_old_registered(tmp_path):
    obs, frozen, catalog, now=example()
    manual=tmp_path/"manual.json"
    output=tmp_path/"candidates.json"
    f=tmp_path/"frozen.jsonl"
    c=tmp_path/"catalog.json"
    manual.write_text(json.dumps({"records":[obs]}))
    f.write_text(json.dumps(frozen[0])+"\n")
    c.write_text(json.dumps(catalog))
    args=dict(input_path=manual,output_path=output,forecasts_path=f,
              catalog_path=c,report_path=tmp_path/"report.json")
    run(**args,now=now)
    old=output.read_text()
    manual.write_text(json.dumps({"records":[obs,{**obs,"quote_id":"new-late-quote"}]}))
    with pytest.raises(ValueError,match="older than ten minutes"):
        run(**args,now=(datetime.fromisoformat(frozen[0]["kickoff_utc"])+
                        timedelta(hours=1)).isoformat())
    assert output.read_text()==old



def test_multiweek_manual_batch_keeps_prior_and_accepts_new_pregame(tmp_path):
    obs, frozen, catalog, now=example()
    manual=tmp_path/"manual.json"
    output=tmp_path/"candidates.json"
    f=tmp_path/"frozen.jsonl"
    c=tmp_path/"catalog.json"
    manual.write_text(json.dumps({"records":[obs]}))
    f.write_text(json.dumps(frozen[0])+"\n")
    c.write_text(json.dumps(catalog))
    args=dict(input_path=manual,output_path=output,forecasts_path=f,
              catalog_path=c,report_path=tmp_path/"report.json")
    run(**args,now=now)
    first=json.loads(output.read_text())["records"][0]
    later=datetime.fromisoformat(frozen[0]["kickoff_utc"])+timedelta(days=1)
    future={**frozen[0], "game_id":"2026_07_NEW_TEST_GAME",
            "player_id":"00-NEWGSIS", 
            "kickoff_utc":(later+timedelta(days=4)).isoformat()}
    new_raw={**obs, "quote_id":"MANUAL_NEW_WEEK_02",
             "game_id":future["game_id"],
             "source_observed_utc":later.isoformat(),
             "snapshot_id":"different-week-original"}
    f.write_text(json.dumps(frozen[0])+"\n"+json.dumps(future)+"\n")
    manual.write_text(json.dumps({"records":[obs,new_raw]}))
    r=run(**args,now=(later+timedelta(minutes=1)).isoformat())
    assert r["historical_source_observations_verified_without_repricing"]==1
    assert r["new_source_observations_eligible_for_current_cutoff"]==1
    assert r["strict_first_seen_candidates"]["new"]==1
    saved=json.loads(output.read_text())["records"]
    assert len(saved)==2
    assert saved[0]==first
    assert saved[1]["player_id"]=="00-NEWGSIS"
