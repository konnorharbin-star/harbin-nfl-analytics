"""Original v1 NFL prop benchmark capture remains research-only."""
import pytest
from scripts.auto_freeze_legacy_prop_baseline import freeze_baseline

AT="2026-10-12T03:45:00Z"


def future(week,date,id):
    return dict(season=2026,week=week,game_id=id,game_type="REG",
      gameday=date,gametime="20:15",
      home_team="LA",away_team="BUF",home_score=None,away_score=None)


def test_legacy_baseline_preflight_schedules_frozen_future_only():
    weeks=[]
    def runner(*,client,season,week,as_of):
        weeks.append(week)
        return {
          "status":"RESEARCH_ONLY_UNPRICED_PLAYER_FORECASTS",
          "actual_sportsbook_edge_proven":False,
          "pregame_player_market_forecasts":90,
          "player_prop_probability_experiment":{
              "status":"SYNTHETIC_LINES_DIAGNOSTIC_RESEARCH",
              "forward_probability_rows":630,
          },
          "forward_ledger":{"new_entries":90},
        }
    scheduled=[future(5,"2026-10-11","already"),
               future(5,"2026-10-12","monday"),
               future(6,"2026-10-15","thursday"),
               future(6,"2026-10-18","sunday")]
    result=freeze_baseline(schedules=scheduled,observed_utc=AT,
                           season=2026,horizon_days=10,runner=runner)
    assert weeks==[5,6]
    assert result["future_weeks"]==2
    assert result["screened_weeks"][0]["v1_calibrated_synthetic_thresholds_in_capture"]==630
    assert result["approved_betting_signals"]==0
    assert result["no_wager_placed"]


def test_baseline_calibration_failure_must_abort_not_reuse_stale_probs():
    def invalid(**kw):
        return {
            "status":"RESEARCH_ONLY_UNPRICED_PLAYER_FORECASTS",
            "actual_sportsbook_edge_proven":False,
            "player_prop_probability_experiment":{
                "status":"BLOCKED_PROBABILITY_EXPERIMENT",
            }
        }
    with pytest.raises(ValueError,match="calibration blocked"):
        freeze_baseline(schedules=[future(6,"2026-10-15","thursday")],
                        observed_utc=AT,season=2026,horizon_days=10,
                        runner=invalid)


def test_no_future_games_never_refits():
    def danger(**kwargs):
        raise AssertionError("no game, no source call")
    result=freeze_baseline(schedules=[future(5,"2026-10-11","started")],
        observed_utc=AT,season=2026,horizon_days=10,runner=danger)
    assert result["future_weeks"]==0


def test_v1_signal_never_approved():
    assert freeze_baseline(schedules=[],observed_utc=AT,season=2026,
        horizon_days=10)["reference_prices_or_independent_profitability_verified"] is False
