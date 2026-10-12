"""Moneyline Elo guards: chronological, tie-aware, never a bookmaker offer."""
from copy import deepcopy
import json
from datetime import datetime, timezone
import pytest

from scripts.nfl_moneyline_elo_research import (
    VERSION, american_implied, market_home_no_vig, elo_home_probability,
    pregame_elo, prior_tie_rate, freeze_future, _score,
)

def game(gid, week, home, away, *, h=None, a=None, year=2026,
         neutral=False, line_home=-110, line_away=-110):
    return {
        "game_id":gid, "season":year, "week":week, "game_type":"REG",
        "home_team":home,"away_team":away,
        "location":"Neutral" if neutral else "Home",
        "home_score":h,"away_score":a,
        "gameday":"2026-10-18" if week == 6 else "2026-10-25",
        "gametime":"13:00", "home_moneyline":line_home,
        "away_moneyline":line_away,
    }

def test_week_leakage_scores_do_not_modify_weekmate_prediction():
    g1=game("A",6,"BUF","LA",h=60,a=0)
    g2=game("B",6,"KC","DEN",h=0,a=20)
    baseline=pregame_elo([g1,g2],tie_rate=.005)
    changed=pregame_elo([dict(g1,home_score=0,away_score=60),g2],tie_rate=.005)
    assert baseline[1]["p_home_win"] == changed[1]["p_home_win"]
    assert baseline[0]["p_home_win"] == changed[0]["p_home_win"]
    assert baseline[1]["p_tie_push"] == .005

def test_next_week_updates_after_completed_week_not_before():
    rows=[
        game("A",6,"BUF","LA",h=42,a=10),
        game("B",7,"BUF","KC",h=None,a=None),
    ]
    out=pregame_elo(rows,tie_rate=.005)
    assert out[1]["p_home_win"] > out[0]["p_home_win"]

def test_half_finished_week_does_not_update_any_ratings():
    rows=[
        game("A",6,"BUF","LA",h=42,a=10),
        game("B",6,"KC","DEN",h=None,a=None),
        game("C",7,"BUF","KC",h=None,a=None),
    ]
    out=pregame_elo(rows,tie_rate=.005)
    assert out[2]["elo_home_delta"] == 65.

def test_neutral_venue_and_team_aliases():
    x=pregame_elo([game("A",6,"OAK","LAR",neutral=True)],tie_rate=.005)[0]
    assert x["home_team"]=="LV" and x["away_team"]=="LA"
    assert x["conditional_p_home_win_given_no_tie"] == .5

def test_american_to_implied_and_novig():
    assert american_implied(-110)==pytest.approx(110/210)
    assert american_implied(100)==pytest.approx(.5)
    assert market_home_no_vig(-110,-110)==pytest.approx(.5)
    assert market_home_no_vig(0,-110) is None
    assert market_home_no_vig(-1000,1000) == pytest.approx(.5)
    assert market_home_no_vig(None,-110) is None

def test_prior_tie_pre2024_only():
    dev=[game(str(i),6,"BUF","LA",h=24,a=24 if i == 0 else 3,
              year=2017) for i in range(310)]
    future=[game("end",6,"BUF","LA",h=24,a=24,year=2025)]
    p,n,t=prior_tie_rate(dev+future)
    assert n==310 and t==1
    assert p==pytest.approx(1.5/311)

def test_duplicate_games_teams_cannot_hide_input_problems():
    with pytest.raises(ValueError,match="duplicate schedule"):
        pregame_elo([game("A",6,"BUF","LA"),game("A",7,"BUF","KC")],tie_rate=.005)
    with pytest.raises(ValueError,match="duplicate team"):
        pregame_elo([game("A",6,"BUF","LA"),game("B",6,"BUF","KC")],tie_rate=.005)

def test_two_way_uses_push_mass_but_no_vig_conditional_on_nontie():
    r=pregame_elo([game("A",6,"BUF","LA")],tie_rate=.02)[0]
    assert sum(r[k] for k in ("p_home_win","p_away_win","p_tie_push")) == pytest.approx(1)
    assert r["p_home_win"] < r["conditional_p_home_win_given_no_tie"]

def test_forward_first_seen_cannot_be_overwritten(tmp_path):
    r=pregame_elo([game("A",6,"BUF","LA")],tie_rate=.005)[0]
    path=tmp_path/"moneyline.jsonl"
    now="2026-10-12T04:00:00Z"
    a=freeze_future([r],observed_utc=now,path=path)
    b=freeze_future([dict(r,p_home_win=.01)],observed_utc=now,path=path)
    assert a["added"]==1 and b["added"]==0
    written=json.loads(path.read_text().strip())
    assert written["model"]==VERSION and written["decision"]=="PASS"
    assert written["payout_estimated_ev"] is None
    assert written["p_home_win"]==r["p_home_win"]
    assert not written["bookmaker_account_acceptance"]

def test_kickoff_after_capture_required(tmp_path):
    r=pregame_elo([game("A",6,"BUF","LA")],tie_rate=.005)[0]
    assert freeze_future([r],observed_utc="2026-10-20T04:00:00Z",
                         path=tmp_path/"none")["added"]==0

def test_archived_odds_comparison_marked_unexecutable():
    g=game("A",6,"BUF","LA",h=24,a=17)
    r=pregame_elo([g],tie_rate=.005)
    result=_score(r)
    assert result["archive_novig_matched"]["n"]==1
    assert result["profitable_edge_proven"] is False
    assert result["not_an_account_quote"] is True

def test_invalid_tie_and_utc_rejected(tmp_path):
    with pytest.raises(ValueError,match="tie prior"):
        pregame_elo([],tie_rate=.2)
    with pytest.raises(ValueError,match="UTC"):
        freeze_future([],observed_utc="2026-10-12T04:00:00",
                      path=tmp_path/"none")
