"""Fixed 2024 snap split challenger must fail closed on source and timing."""
import json
import pytest
from scripts.nfl_prop_snap_conditional_cdf import (
    VERSION,bucket,fit_buckets,candidate_residuals,selection,
    score,first_catalog,freeze_future,stat_samples_with_workload
)
from scripts.nfl_player_prop_full_distribution import hash_object

def base_catalog():
    c={"training_year":2024,"vectors":{
        "rushing_yards|RB":[(i-180)/70 for i in range(360)],
        "rushing_yards|ALL":[(i-180)/70 for i in range(400)],
    }}
    c["sha256"]=hash_object(c)
    return c

def train_samples(n=420):
    rows=[]
    for i in range(n):
        b="HIGH_35_PLUS" if i%2 else "LOW_UNDER_35"
        p={"game_id":f"2024_{i//13}","player_id":f"id{i}",
           "market":"rushing_yards","position":"RB",
           "challenger_mean":65.,"season":2024}
        y=65+(i%47)-23+(5 if b=="HIGH_35_PLUS" else -5)
        rows.append((p,float(y),b))
    return rows

def full_original():
    return {
        "model":"NFL_PLAYER_PROP_FROZEN_FULL_DISTRIBUTION_V1",
        "game_id":"2026_06_BUF_LA","season":2026,"week":6,
        "player_id":"GSIS123","player_name":"Synthetic RB","team":"BUF",
        "position":"RB","market":"rushing_yards",
        "original_point_mean":68.,"original_scale":34.,
        "kickoff_utc":"2026-10-18T17:00:00Z",
        "first_observed_at_utc":"2026-10-12T04:36:00Z",
    }

def context():
    return {"game_id":"2026_06_BUF_LA","player_id":"GSIS123",
        "market":"rushing_yards",
        "observed_at_utc":"2026-10-12T11:32:18Z",
        "observed_mean":49.,
        "status":"HISTORICAL_OFFENSIVE_SNAPS_OBSERVED"}

def test_bucket_cutoff_fixed_and_invalid_values():
    assert bucket(34.999)=="LOW_UNDER_35"
    assert bucket(35.)=="HIGH_35_PLUS"
    with pytest.raises(ValueError):bucket(-1)
    with pytest.raises(ValueError):bucket(float("nan"))

def test_buckets_fit_2024_only_and_are_sha_frozen():
    base=base_catalog()
    c=fit_buckets(train_samples(),base)
    assert c["development_year"]==2024
    assert c["2025_outcomes_used_to_fit"] is False
    assert c["base_full_distribution_catalog_sha256"]==base["sha256"]
    assert c["sha256"]==hash_object({k:v for k,v in c.items() if k!="sha256"})
    a,k=candidate_residuals(c,"rushing_yards","RB","HIGH_35_PLUS")
    assert len(a)>=80 and k=="rushing_yards|RB|HIGH_35_PLUS"

def test_low_support_uses_original_empirical_cdf_exactly():
    base=base_catalog()
    c=fit_buckets(train_samples(60),base)
    p=train_samples(1)[0][0]
    old,oldkey,new,newkey,fallback=selection(p,"HIGH_35_PLUS",c,base)
    assert fallback
    assert new is old and newkey==oldkey

def test_2025_paired_probabilities_use_identical_frozen_thresholds():
    base=base_catalog()
    training=train_samples()
    c=fit_buckets(training,base)
    s=score(training[:80],c,base,bootstrap=20)
    x=s["rushing_yards"]
    assert x["source_official_stat_row_player_market_groups"]==80
    assert x["correlated_synthetic_lines"]==80*7
    assert x["unique_nfl_games"]>=6
    assert x["account_BetMGM_offer_verified"] is False
    assert x["validated_market_profit_edge"] is False

def test_first_2024_bucket_catalog_never_overwritten(tmp_path):
    base=base_catalog();rows=train_samples()
    path=tmp_path/"catalog.json"
    _,created=first_catalog(path,rows,base)
    assert created
    frozen=path.read_text()
    _,created=first_catalog(path,rows,base)
    assert not created and path.read_text()==frozen
    with pytest.raises(ValueError,match="do NOT silently refit"):
        first_catalog(path,rows[:-1],base)
    invalid=json.loads(frozen);invalid["fitted_normalized_residual_vectors"]["rushing_yards|RB|HIGH_35_PLUS"][0]=9999
    path.write_text(json.dumps(invalid))
    with pytest.raises(ValueError,match="SHA invalid"):
        first_catalog(path,rows,base)

def test_future_origins_respect_original_forecast_snap_and_book_ban(tmp_path):
    base=base_catalog();c=fit_buckets(train_samples(),base)
    f=full_original();f["residual_catalog_sha256"]=base["sha256"]
    path=tmp_path/"shadow.jsonl"
    at="2026-10-12T12:00:00Z"
    out=freeze_future([f],[context()],c,base,observed_utc=at,path=path)
    assert out["new_originals"]==1
    got=json.loads(path.read_text().strip())
    assert got["model"]==VERSION and got["decision"]=="PASS"
    assert got["frozen_bucket"]=="HIGH_35_PLUS"
    assert got["p_bet_win"] is None and got["p_bet_void"] is None
    assert got["true_EV"] is None and got["BetMGM_US_accepted_quote"] is None
    # Later source revisions cannot rewrite any first-seen original.
    changed=dict(context(),observed_mean=4)
    out=freeze_future([f],[changed],c,base,observed_utc=at,path=path)
    assert out["new_originals"]==0
    assert json.loads(path.read_text().strip())==got

def test_after_kickoff_and_invalid_snap_capture_fail_closed(tmp_path):
    base=base_catalog();c=fit_buckets(train_samples(),base)
    f=full_original();f["residual_catalog_sha256"]=base["sha256"]
    later="2026-10-19T20:00:00Z"
    assert freeze_future([f],[context()],c,base,observed_utc=later,
        path=tmp_path/"none.jsonl")["new_originals"]==0
    corrupted=dict(context(),observed_at_utc="2026-10-12T02:30:00Z")
    with pytest.raises(ValueError,match="captured after model freeze"):
        freeze_future([f],[corrupted],c,base,
            observed_utc="2026-10-12T12:00:00Z",path=tmp_path/"none")

def test_missing_player_context_is_not_imputed_zero_or_void(tmp_path):
    base=base_catalog();c=fit_buckets(train_samples(),base)
    f=full_original();f["residual_catalog_sha256"]=base["sha256"]
    record=freeze_future([f],[],c,base,
        observed_utc="2026-10-12T12:00:00Z",path=tmp_path/"shadow")
    assert record["new_originals"]==0
    assert record["excluded"]["NO_QUALIFIED_PRIOR_SNAP_CONTEXT"]==1

def test_frozen_original_catalog_sha_checked_before_price_evaluation():
    base=base_catalog();c=fit_buckets(train_samples(),base)
    base["vectors"]["rushing_yards|RB"][0]=9999
    with pytest.raises(ValueError,match="corrupted original"):
        score(train_samples(10),c,base)


def test_public_source_retry_preserves_data_and_rejects_permanent_504(monkeypatch):
    import scripts.nfl_prop_snap_conditional_cdf as module
    calls=[]
    class Response:
        def to_dicts(self): return [{"year":2024,"offense_snaps":50}]
    def flaky(seasons):
        calls.append(tuple(seasons))
        if len(calls)==1:raise ConnectionError("504 Gateway Timeout")
        return Response()
    monkeypatch.setattr(module.nfl,"load_snap_counts",flaky)
    rows=module.load_snap_history_with_retry([2024],attempts=2,sleep=lambda _:None)
    assert rows==[{"year":2024,"offense_snaps":50}]
    assert calls==[(2024,),(2024,)]
    def dead(seasons):raise ConnectionError("504 Gateway Timeout")
    monkeypatch.setattr(module.nfl,"load_snap_counts",dead)
    with pytest.raises(ConnectionError,match="unavailable after 2 attempts"):
        module.load_snap_history_with_retry([2024],attempts=2,sleep=lambda _:None)


def test_nfl_player_crosswalk_504_retries_without_stale_identity():
    from scripts.nfl_prop_snap_conditional_cdf import public_download_with_retry
    attempts={"count":0}
    def recover():
        attempts["count"]+=1
        if attempts["count"]==1:raise ConnectionError("HTTP 504")
        return [{"gsis_id":"ABC","pfr_id":"XYZ"}]
    ids=public_download_with_retry(recover,label="players",attempts=3,sleep=lambda _:None)
    assert ids==[{"gsis_id":"ABC","pfr_id":"XYZ"}]
    with pytest.raises(ConnectionError,match="source players unavailable"):
        public_download_with_retry(lambda: (_ for _ in ()).throw(ConnectionError("504")),
                                   label="players",attempts=2,sleep=lambda _:None)
