"""Residual challenger may identify a signal, but must not manufacture bets."""
import math

from nfl.market_residual_challenger import (
    evaluate_residual_challenger, residual_prediction,
)


def row(market="moneyline",season=2024,model=.8,no_vig=.5,y=0,
        game=0,odds=2.0):
    return {
        "market":market,"season":season,"model":model,
        "market_p":no_vig,"win":y,"game_id":f"{season}-{game}",
        "week":1+game%18,"decimal":odds,
        "entry_timestamp_verified":False,
    }


def test_market_only_alpha_zero_exactly_matches_market():
    assert math.isclose(residual_prediction(.8,.4,0),.4)
    assert residual_prediction(.8,.4,-.4)<.4
    assert math.isclose(residual_prediction(.8,.4,1),.8)


def test_development_only_selects_negative_weight_if_model_anti_predictive():
    # 2024 extreme synthetic example: model always predicts winner backwards.
    # 2025 diagnostic outcomes are deliberately reversed; they must NOT
    # influence 2024-fitted alpha or be treated as pristine validation.
    train = [
        row(season=2024,model=.8 if i%2 else .2,
            y=0 if i%2 else 1,game=i)
        for i in range(200)
    ]
    diag = [
        row(season=2025,model=.8 if i%2 else .2,
            y=1 if i%2 else 0,game=i)
        for i in range(200)
    ]
    report=evaluate_residual_challenger(train+diag)["by_market"]["moneyline"]
    assert report["alpha_selected_on_2024"]<0
    assert report["2025_residual_candidate"]["log_loss"] > report["2025_market_only"]["log_loss"]
    assert report["deployment_authorized"] is False
    assert report["positive_archived_model_edge_proven"] is False


def test_no_model_or_season_coverage_is_fail_closed():
    result=evaluate_residual_challenger([])["by_market"]
    assert all(x["status"]=="INSUFFICIENT_RESEARCH_SAMPLE" for x in result.values())


def test_2025_diagnostic_data_does_not_change_development_selection():
    train = [
        row(season=2024,model=.7 if i%2 else .3,
            y=i%2,game=i)
        for i in range(170)
    ]
    a=[row(season=2025,model=.7 if i%2 else .3,
           y=i%2,game=i) for i in range(170)]
    b=[{**x,"win":1-x["win"]} for x in a]
    first=evaluate_residual_challenger(train+a)["by_market"]["moneyline"]
    second=evaluate_residual_challenger(train+b)["by_market"]["moneyline"]
    assert first["alpha_selected_on_2024"] == second["alpha_selected_on_2024"]
