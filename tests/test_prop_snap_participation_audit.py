"""Real identity joins only; missing stats never assumed a lost under."""
import pytest
from scripts.prop_snap_participation_audit import audit,crosswalk,snap_index

NOW="2026-10-13T08:15:00Z"
def game():
    return {
      "game_id":"2026_05_BUF_LA","season":2026,"week":5,"game_type":"REG",
      "gameday":"2026-10-12","gametime":"20:15",
      "home_team":"LA","away_team":"BUF","home_score":21,"away_score":20}
def prop(pid="00-00001",market="receptions"):
    return {"game_id":"2026_05_BUF_LA","season":"2026","week":"5",
      "first_observed_at_utc":"2026-10-11T00:00:00Z",
      "kickoff_utc":"2026-10-13T00:15:00Z",
      "team":"BUF","player_id":pid,"market":market,"synthetic_line":"4.5"}
def player(gsis="00-00001",pfr="DemoPl01"):
    return {"gsis_id":gsis,"pfr_id":pfr}
def snap(pfr="DemoPl01",team="BUF",offense=10):
    return {"game_id":"2026_05_BUF_LA","team":team,
      "pfr_player_id":pfr,"offense_snaps":offense}

def run(forecasts=None,stats=None,players=None,snaps=None,asof=NOW):
    return audit(forecasts if forecasts is not None else [prop()],
      [game()],stats if stats is not None else [],
      players if players is not None else [player()],
      snaps if snaps is not None else [snap()],
      observed_utc=asof)

def test_gsis_pfr_is_not_name_join():
    r=run()
    assert r["missing_distinct_player_game_market_groups"]==1
    assert r["classification_groups"]["POSITIVE_OFFENSE_SNAPS_BUT_MISSING_STAT_ROW"]==1
    assert not r["can_impute_missing_official_stats_as_zero"]
    assert not r["may_promote_model"]

def test_same_player_multiple_market_thresholds_are_correlated():
    r=run(forecasts=[prop(),dict(prop(),synthetic_line="5.5"),
                     prop(market="receiving_yards")])
    assert r["missing_original_threshold_rows"]==3
    assert r["missing_distinct_player_game_market_groups"]==2

def test_missing_crosswalk_does_not_infer_zero():
    r=run(players=[])
    assert r["classification_groups"]["NO_UNAMBIGUOUS_GSIS_PFR_CROSSWALK"]==1

def test_same_pfr_for_two_gsis_is_ambiguous():
    mapping,n=crosswalk([player("GSIS1"),player("GSIS2")])
    assert mapping=={} and n==2

def test_no_verified_snap_does_not_prove_nonparticipation():
    r=run(snaps=[])
    assert r["classification_groups"]["NO_VERIFIED_PFR_GAME_SNAP_RECORD"]==1

def test_zero_offense_snap_does_not_prove_dnp():
    r=run(snaps=[snap(offense=0)])
    assert r["classification_groups"]["ZERO_OFFENSE_SNAPS_DOES_NOT_PROVE_DNP"]==1

def test_wrong_team_in_pfr_snap_is_not_a_valid_player_join():
    r=run(snaps=[snap(team="LA")])
    assert r["classification_groups"]["PFR_SNAP_TEAM_MISMATCH"]==1

def test_source_different_game_does_not_match():
    s=snap();s["game_id"]="OTHER"
    r=run(snaps=[s])
    assert r["classification_groups"]["NO_VERIFIED_PFR_GAME_SNAP_RECORD"]==1

def test_unknown_game_beyond_schedule_not_linked():
    x=prop();x["game_id"]="wrong"
    r=run(forecasts=[x])
    assert r["classification_groups"]["UNKNOWN_GAME"]==1

def test_future_game_not_postgame_audited():
    r=run(asof="2026-10-13T00:16:00Z")
    assert r["classification_groups"]["NOT_FINAL_OR_TOO_EARLY"]==1

def test_predicted_player_with_official_stats_is_not_missing():
    r=run(stats=[{"game_id":"2026_05_BUF_LA","player_id":"00-00001"}])
    assert r["missing_original_threshold_rows"]==0
    assert r["original_threshold_rows_with_official_stat"]==1

def test_postkickoff_forecast_is_refused():
    x=prop();x["first_observed_at_utc"]="2026-10-13T00:15:00Z"
    with pytest.raises(ValueError,match="before-kickoff"):
        run(forecasts=[x])

def test_duplicate_pfr_rows_are_not_joined():
    index,duplicates=snap_index([snap(),snap()])
    assert not index and duplicates==1

def test_different_source_pfr_id_not_same_player():
    r=run(players=[player(pfr="OtherPF00")])
    assert r["classification_groups"]["NO_VERIFIED_PFR_GAME_SNAP_RECORD"]==1
