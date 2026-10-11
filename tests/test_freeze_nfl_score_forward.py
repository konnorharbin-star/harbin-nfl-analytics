"""Test that first-seen NFL scores remain frozen and can never be postgame."""
import copy
import json
from pathlib import Path

import pytest

from scripts.freeze_nfl_score_forward import freeze, _load_history


def game(**overrides):
    row = dict(game_id="2026_05_BUF_LA",season=2026,week=5,game_type="REG",
               home_team="LA",away_team="BUF",gameday="2026-10-12",
               gametime="20:15",home_score=None,away_score=None)
    row.update(overrides)
    return row


def projection(**overrides):
    row = dict(game_id="2026_05_BUF_LA",season=2026,week=5,
               home_team="LA",away_team="BUF",
               baseline_home_margin=0.5,baseline_total=50.0,
               home_qb_proxy_name="Home previous-week",
               away_qb_proxy_name="Away previous-week")
    row.update(overrides)
    return dict(status="READY",games=[row])


NOW="2026-10-11T23:40:00Z"
SHA="abcdef1234567890"


def run(proj=None, schedule=None, existing=None, time=NOW):
    return freeze(proj or projection(), [schedule or game()],
                  [] if existing is None else existing,
                  captured_utc=time,source_sha=SHA)


def test_future_first_seen_prediction_and_no_ev():
    saved,r=run()
    assert r["new_records"]==["2026_05_BUF_LA"]
    assert r["decision"]=="PASS" and not r["positive_ev_proven"]
    x=saved[0]
    assert x["kickoff_utc"]=="2026-10-13T00:15:00+00:00"
    assert x["independent_projected_home_margin"]==0.5
    assert x["independent_projected_total"]==50
    assert x["independent_home_points"]==25.25
    assert x["independent_away_points"]==24.75
    assert x["p_win"] is None and x["estimated_ev"] is None
    assert not x["odds_verified"] and not x["bet_executed"]
    assert x["content_sha256"]


def test_first_snapshot_is_never_replaced_by_new_score():
    prior,_=run()
    changed=projection(baseline_home_margin=12,baseline_total=66)
    updated,r=run(proj=changed,existing=prior)
    assert updated==prior
    assert r["skipped_existing"]==["2026_05_BUF_LA"]
    assert r["new_records"]==[]


def test_after_kickoff_without_final_is_not_pregame():
    records,r=run(time="2026-10-13T00:16:00Z")
    assert records==[]
    assert r["skipped_not_pregame"]==["2026_05_BUF_LA"]


def test_final_score_in_schedule_blocks_first_seen():
    records,r=run(schedule=game(home_score=27,away_score=24))
    assert records==[] and r["decision"]=="PASS"
    assert r["skipped_not_pregame"]


def test_future_schedule_kickoff_required():
    with pytest.raises(ValueError,match="kickoff unavailable"):
        run(schedule=game(gametime=None))


def test_mismatch_schedule_identity_rejected():
    with pytest.raises(ValueError,match="identity mismatch"):
        run(schedule=game(home_team="ATL"))


def test_duplicate_source_games_rejected():
    p=projection()
    p["games"].append(copy.deepcopy(p["games"][0]))
    with pytest.raises(ValueError,match="duplicate"):
        run(proj=p)


def test_missing_or_nonfinite_score_fail_closed():
    for margin in (None, float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            run(proj=projection(baseline_home_margin=margin))


def test_immutable_history_duplicate_rejected():
    history,_=run()
    with pytest.raises(ValueError,match="duplicate"):
        run(existing=history+history)


def test_blocked_engine_does_not_fabricate_record():
    records,r=run(proj=dict(status="BLOCKED",reason="missing quarterback data",games=[]))
    assert records==[] and r["status"]=="BLOCKED"
    assert r["blockers"] and r["decision"]=="PASS"


def test_jsonl_roundtrip(tmp_path):
    history,_=run()
    path=tmp_path/"history.jsonl"
    path.write_text(json.dumps(history[0])+"\n",encoding="utf8")
    assert _load_history(path)==history


def test_unknown_source_sha_rejected():
    with pytest.raises(ValueError,match="source Git SHA"):
        freeze(projection(),[game()],[],captured_utc=NOW,source_sha="")
