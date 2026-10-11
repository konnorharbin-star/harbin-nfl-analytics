"""Tests for frozen professional-pick research ledger and faithful main-market grading."""
from datetime import datetime, timezone
from pathlib import Path
import csv
import tempfile

import pytest

from scripts.public_bettor_tracking import grade, read_picks, review, theoretical_units

ROOT = Path(__file__).resolve().parents[1]
PICKS = ROOT / "research" / "public_expert_picks_2026.csv"


def test_real_frozen_source_entries_are_pregame_and_distinct():
    picks = read_picks(PICKS)
    assert len(picks) == 5
    assert len(set(x["record_id"] for x in picks)) == len(picks)
    assert all(x["entry_status"] == "OBSERVED_EDITORIAL_NOT_EXECUTABLE" for x in picks)


def test_key_number_three_allows_both_sides_nonloss():
    game = {"gameId": "2026_05_BAL_ATL", "homeTeam": "ATL",
            "awayTeam": "BAL", "homeScore": 27, "awayScore": 24}
    picks = read_picks(PICKS)
    stuckey = next(p for p in picks if p["analyst"] == "Stuckey")
    ward = next(p for p in picks if p["analyst"] == "Billy Ward")
    assert grade(stuckey, game) == "W"
    assert grade(ward, game) == "P"


def test_total_does_not_infer_player_boxscore():
    game = {"gameId": "2026_05_BAL_ATL", "homeTeam": "ATL",
            "awayTeam": "BAL", "homeScore": 20, "awayScore": 17}
    picks = read_picks(PICKS)
    assert grade(next(p for p in picks if p["analyst"] == "John Lanfranca"), game) == "W"
    assert grade(next(p for p in picks if p["analyst"] == "Sean Koerner"), game) == "NEEDS_PLAYER_BOX"


def test_no_pregame_published_picks_are_graded_before_final():
    picks = read_picks(PICKS)
    rep = review(picks, [{"gameId": "2026_05_BAL_ATL", "homeScore": None,
                          "awayScore": None}])
    assert all(row["grade"] == "PENDING" for row in rep["picks"])
    assert all(row["account_quote_verified"] is False for row in rep["picks"])
    assert rep["decision"] == "PASS"


def test_article_returns_labelled_hypothetical():
    assert theoretical_units("W", -110) == pytest.approx(100 / 110)
    assert theoretical_units("P", -110) == 0
    assert theoretical_units("L", -110) == -1
    assert theoretical_units("PENDING", -110) is None


def test_temporal_correction_rejected(tmp_path):
    src = PICKS.read_text(encoding="utf8")
    original = "2026-10-11T22:42:34Z"
    assert original in src
    late = tmp_path / "late.csv"
    late.write_text(src.replace(original, "2026-10-12T03:00:00Z"), encoding="utf8")
    with pytest.raises(ValueError, match="pre-kickoff"):
        read_picks(late)


def test_other_nfl_team_codes_are_valid(tmp_path):
    source = PICKS.read_text(encoding="utf8")
    assert ",spread,BAL,BAL,+3.5," in source
    tmp = tmp_path / "other_teams.csv"
    tmp.write_text(source.replace(",spread,BAL,BAL,+3.5,",
                                  ",spread,BUF,BUF,+3.5,"),
                   encoding="utf8")
    picks = read_picks(tmp)
    assert picks[0]["selection"] == "BUF"


def test_duplicate_rejected(tmp_path):
    source = PICKS.read_text(encoding="utf8")
    tmp = tmp_path / "dups.csv"
    tmp.write_text(source + source.splitlines()[1] + "\n", encoding="utf8")
    with pytest.raises(ValueError, match="duplicate"):
        read_picks(tmp)
