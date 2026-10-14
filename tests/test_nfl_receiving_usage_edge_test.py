from copy import deepcopy
import polars as pl
import pytest
from scripts import nfl_receiving_usage_edge_test as research


def schedule():
    return pl.DataFrame([{"game_type":"REG","game_id":"2025_04_A_B",
                          "season":2025,"week":4,"home_team":"B","away_team":"A"}])


def player():
    return {"player_id":"00-1","season_type":"REG","season":2025,
            "week":4,"recent_team":"A","opponent_team":"B"}


def test_weekly_missing_game_id_requires_unique_exact_schedule_identity():
    row=player()
    original=deepcopy(row)
    result=research.normalize_players(schedule(),pl.DataFrame([row]))
    assert row==original
    assert result["game_id"].to_list()==["2025_04_A_B"]
    assert result["team"].to_list()==["A"]
    for changed in ({"game_id":"different"},{"opponent_team":"C"},{"recent_team":"C"}):
        bad={**row,**changed}
        with pytest.raises(ValueError):
            research.normalize_players(schedule(),pl.DataFrame([bad]))
    with pytest.raises(ValueError,match="DUPLICATE_PLAYER"):
        research.normalize_players(schedule(),pl.DataFrame([row,row]))


def test_missing_target_rows_are_unknown_and_cannot_create_zero_residual(monkeypatch):
    forecasts=[{"game_id":"G","player_id":pid,"team":"A","market":"receptions"}
               for pid in ("observed","absent")]
    monkeypatch.setattr(research,"forecast_week",lambda *a,**k: forecasts)
    games=[{"game_id":"G","season":2025,"week":4,"completed":True}]
    stats=[{"game_id":"G","player_id":"observed","team":"A","receptions":0}]
    rows,unknown=research.examples(games,stats,2025)
    assert unknown==1 and len(rows)==1
    assert rows[0][0]["player_id"]=="observed" and rows[0][1]==0


def test_correlated_player_rows_do_not_satisfy_distinct_game_gate():
    rows=[{"game_id":"one_game","brier_delta":-.01} for _ in range(1000)]
    assert research.clustered(rows,"brier_delta",reps=100) is None


def test_residual_pool_respects_position_sample_requirement():
    catalogs={("receptions","WR"):[9]*79,("receptions","ALL"):[1]*160}
    row={"market":"receptions","position":"WR"}
    assert research.vector(catalogs,row)==[1]*160
    catalogs[("receptions","WR")].append(9)
    assert research.vector(catalogs,row)==[9]*80


def test_week_cluster_sensitivity_requires_multiple_independent_weeks():
    one_week=[{"game_id":f"2025_01_{i}","brier_delta":-.01} for i in range(100)]
    assert research.clustered(one_week,"brier_delta",reps=100,by_week=True) is None
    eight_weeks=[{"game_id":f"2025_{w:02d}_A_B","brier_delta":-.01} for w in range(1,9)]
    assert research.clustered(eight_weeks,"brier_delta",reps=100,by_week=True)==pytest.approx([-.01,-.01])
