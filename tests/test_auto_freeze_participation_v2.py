"""Automatic v2 frozen NFL week discovery: only never-played future games."""
import pytest
from scripts.auto_freeze_participation_v2 import future_weeks,freeze_upcoming

AT="2026-10-12T03:45:00Z"


def game(week,date,time,gid,*,complete=False,typ="REG"):
    return dict(season=2026,week=week,game_id=gid,game_type=typ,
        gameday=date,gametime=time,
        home_team="LA",away_team="BUF",
        home_score=20 if complete else None,
        away_score=18 if complete else None)


def fixtures():
    return [
        game(5,"2026-10-11","20:20","played"),
        game(5,"2026-10-12","20:15","monday"),
        game(6,"2026-10-15","20:15","thursday"),
        game(6,"2026-10-18","20:25","sunday"),
        game(7,"2026-10-22","20:15","next_thursday"),
        game(7,"2026-10-25","20:15","out_of_horizon"),
        game(6,"2026-10-18","20:15","other_done",complete=True),
        game(6,"2026-10-18","20:15","preseason",typ="PRE"),
    ]


def test_auto_discovers_multiple_future_regular_weeks():
    result=future_weeks(fixtures(),observed_utc=AT,horizon_days=10)
    assert [x["week"] for x in result]==[5,6]
    assert result[0]["future_games"]==1
    assert result[1]["future_games"]==2


def test_excludes_future_week_outside_short_horizon():
    weeks=future_weeks(fixtures(),observed_utc=AT,horizon_days=2)
    assert [x["week"] for x in weeks]==[5]


def test_no_upcoming_games_is_valid_no_bet():
    assert future_weeks([],observed_utc=AT)==[]


def test_disallows_registering_2027_with_2024_calibrator():
    with pytest.raises(ValueError,match="2026"):
        future_weeks([],observed_utc=AT,season=2027)


def test_disallows_unbounded_future_research_horizon():
    with pytest.raises(ValueError,match="horizon"):
        future_weeks([],observed_utc=AT,horizon_days=180)


def test_frozen_runner_called_only_on_future_game_weeks_and_fail_closed():
    observed=[]
    def fake_run(*,season,week,freeze_path,report_path):
        observed.append((season,week))
        return {
            "decision":"PASS","approved_betting_model":False,
            "2026_not_used_for_fit":True,
            "version":"PROP_PARTICIPATION_CONDITIONAL_V2_RESEARCH",
            "current_account_betmgm_prices":None,
            "forward_frozen_records":{"new":7 if week==5 else 14},
        }
    r=freeze_upcoming(schedules=fixtures(),observed_utc=AT,season=2026,
          horizon_days=10,freeze_path="synthetic-do-not-write",runner=fake_run)
    assert observed==[(2026,5),(2026,6)]
    assert r["first_seen_unique_synthetic_forecasts_added"]==21
    assert r["eligible_upcoming_weeks"]==2
    assert r["decision"]=="PASS" and r["actual_bet_count"]==0
    assert r["approved_betting_model"] is False


def test_unapproved_incorrect_run_provenance_rejected():
    def tainted(**kwargs):
        return {"decision":"PASS","2026_not_used_for_fit":False,
                "approved_betting_model":False}
    with pytest.raises(ValueError,match="contract"):
        freeze_upcoming(schedules=fixtures(),observed_utc=AT,season=2026,
             horizon_days=10,freeze_path="irrelevant",runner=tainted)


def test_no_runner_called_when_all_past():
    def bomb(**kwargs):raise AssertionError("must not freeze after kickoff")
    r=freeze_upcoming(schedules=[game(5,"2026-10-11","20:15","old")],
        observed_utc=AT,season=2026,horizon_days=10,
        freeze_path="nothing",runner=bomb)
    assert r["eligible_upcoming_weeks"]==0 and r["first_seen_unique_synthetic_forecasts_added"]==0
