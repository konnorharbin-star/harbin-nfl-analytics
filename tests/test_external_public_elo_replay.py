"""Synthetic regressions for weekly as-of external public Elo-style replay."""
import copy
import math
import polars as pl
import pytest

from scripts.external_public_elo_replay import (
    pregame_elo_and_scoring, elo_probability, mov_multiplier,
    fit_calibration, evaluate,
)


def games():
    # Two independent fixtures share a week; forecast MUST be frozen before
    # either current-week result is incorporated in ratings or totals.
    return pl.DataFrame([
        dict(game_id="2019_01_A_B",season=2019,week=1,game_type="REG",
             home_team="A",away_team="B",home_score=35,away_score=10,location="Home"),
        dict(game_id="2019_01_C_D",season=2019,week=1,game_type="REG",
             home_team="C",away_team="D",home_score=22,away_score=18,location="Home"),
        dict(game_id="2019_02_A_C",season=2019,week=2,game_type="REG",
             home_team="A",away_team="C",home_score=24,away_score=20,location="Home"),
        dict(game_id="2020_01_A_D",season=2020,week=1,game_type="REG",
             home_team="A",away_team="D",home_score=17,away_score=20,location="Neutral"),
    ])


def test_same_week_cannot_leak_to_other_game():
    original=pregame_elo_and_scoring(games())
    edited=games().with_columns(
        pl.when(pl.col("game_id")=="2019_01_A_B").then(0)
        .otherwise(pl.col("home_score")).alias("home_score"))
    changed=pregame_elo_and_scoring(edited)
    b=next(x for x in original if x["game_id"]=="2019_01_C_D")
    c=next(x for x in changed if x["game_id"]=="2019_01_C_D")
    assert b["elo_delta"]==c["elo_delta"]
    assert b["raw_trailing_total"]==c["raw_trailing_total"]
    # But week TWO should be permitted to use the earlier completed outcome.
    b2=next(x for x in original if x["game_id"]=="2019_02_A_C")
    c2=next(x for x in changed if x["game_id"]=="2019_02_A_C")
    assert b2["elo_delta"]!=c2["elo_delta"]


def test_neutral_field_no_elo_home_bonus():
    x=pregame_elo_and_scoring(games())
    neutral=next(r for r in x if r["season"]==2020)
    assert neutral["elo_delta"]<300
    assert 0<neutral["elo_home_win_probability"]<1


def test_elo_probabilities_and_mov_are_valid():
    assert math.isclose(elo_probability(0),0.5)
    assert elo_probability(100)>0.5
    assert elo_probability(-100)<0.5
    assert mov_multiplier(7,65,1)>0
    assert mov_multiplier(-7,65,0)>0
    assert mov_multiplier(0,65,.5)>0


def seed():
    # Must be 500+ pre-2023 development games.
    rows=[]
    for season in (2019,2020,2021,2022,2023,2024,2025):
        for i in range(140):
            rows.append({
                "game_id":f"{season}_0{i:03d}",
                "season":season,
                "week":5+i//14,
                "actual_margin":float(i%13-6),
                "actual_total":float(36+i%15),
                "elo_delta":float((i%23-11)*18+65),
                "raw_trailing_total":43.,
            })
    return rows


def test_calibration_invariant_to_inspected_future_outcomes():
    records=seed()
    original=fit_calibration(records)
    altered=copy.deepcopy(records)
    for item in altered:
        if item["season"]>=2023:
            item["actual_margin"]=10000.
            item["actual_total"]=0.
    assert fit_calibration(altered)==original


def test_score_report_never_claims_bet_edge():
    records=seed()
    baseline=[
        dict(game_id=r["game_id"],actual_margin=r["actual_margin"],
             actual_total=r["actual_total"],
             projected_home_margin=1.,projected_total=44.)
        for r in records if r["season"]>=2023
    ]
    quote=pl.DataFrame([
        dict(game_id=r["game_id"],spread_line=2.5,total_line=44.5)
        for r in records if r["season"]>=2023
    ])
    report=evaluate(records,baseline,quote)
    assert report["decision"]=="PASS"
    assert not report["model_profit_proven"]
    assert not report["market_prices_executable"]
    assert not report["canonical_modified"]
    assert report["diagnostic_games"]==420
    assert report["by_season"]["2025"]["margin"]["n"]==140


def test_duplicate_same_team_week_is_forbidden():
    original=games()
    duplicate=original.with_columns(
        pl.when(pl.col("game_id")=="2019_01_C_D")
        .then(pl.lit("A")).otherwise(pl.col("home_team")).alias("home_team"))
    with pytest.raises(ValueError,match="simultaneous"):
        pregame_elo_and_scoring(duplicate)


def test_insufficient_calibration_fails():
    with pytest.raises(ValueError,match="insufficient"):
        fit_calibration(seed()[:40])
