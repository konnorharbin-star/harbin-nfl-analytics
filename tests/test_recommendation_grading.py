from datetime import UTC, datetime

import pytest

from scripts.grade_recommendations import final_result, grade
from scripts.recommendation_ledger import archive_decision


def receipt(tmp_path, market="spread", side="home", line=-3.5):
    row = {
        "sport": "nfl",
        "game_id": "g1",
        "home_team": "BUF",
        "away_team": "NE",
        "kickoff": "2026-10-11T18:00:00Z",
        "quoted_at": "2026-10-10T11:30:00Z",
        "quote_observed_at": "2026-10-10T11:31:00Z",
        "market": market,
        "side": side,
        "line": line,
        "american_odds": -110,
        "book": "fixture",
        "model_version": "test",
        "source_url": "https://example.com",
        "confidence": "test",
        "risks": ["test"],
        "model_probability": 0.60,
        "conservative_probability": 0.56,
        "decision": "BET_READY",
        "bet_approved": True,
    }
    return archive_decision(
        row,
        gate={"production_eligible": True},
        root=tmp_path,
        observed_at=datetime(2026, 10, 10, 12, tzinfo=UTC),
    )


@pytest.mark.parametrize(
    "market,side,line,scores,outcome,profit",
    [
        ("spread", "home", -3.5, (24, 20), "WIN", 100 / 110),
        ("spread", "away", 3.5, (24, 20), "LOSS", -1),
        ("spread", "home", -3, (23, 20), "PUSH", 0),
        ("total", "under", 44, (24, 20), "PUSH", 0),
        ("total", "over", 43.5, (24, 20), "WIN", 100 / 110),
        ("moneyline", "away", None, (20, 24), "WIN", 100 / 110),
    ],
)
def test_one_unit_settlement(tmp_path, market, side, line, scores, outcome, profit):
    r = receipt(tmp_path, market, side, line)
    result = grade(r, {"home_score": scores[0], "away_score": scores[1]})
    assert result["result"] == outcome
    assert result["profit_units"] == pytest.approx(profit)
    assert result["clv"] is None


def test_final_score_requires_identity_time_and_completed_status(tmp_path):
    row = receipt(tmp_path)["original_board_row"]
    event = {
        "id": "1",
        "date": row["kickoff"],
        "status": {"type": {"completed": True, "name": "STATUS_FINAL"}},
        "competitions": [
            {
                "competitors": [
                    {"homeAway": "home", "score": "24", "team": {"abbreviation": "BUF"}},
                    {"homeAway": "away", "score": "20", "team": {"abbreviation": "NE"}},
                ]
            }
        ],
    }
    assert final_result(row, {"events": [event]})["home_score"] == 24
    assert final_result(row, {"events": [event, event]}) is None
    event["status"]["type"]["completed"] = False
    assert final_result(row, {"events": [event]}) is None
