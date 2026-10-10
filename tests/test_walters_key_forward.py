"""Frozen 2026 key-number shadow must not tune to outcomes or place wagers."""
import csv
from datetime import UTC, datetime, timedelta

import pytest

from scripts.walters_key_forward import (
    VERSION, _threeway_score, capture, grade, make_snapshots,
)

NOW = datetime(2026, 10, 10, 19, 0, tzinfo=UTC)
KICKOFF = NOW + timedelta(hours=24)


def _hist(path, sport):
    fields = {
        "nfl": ("projected_home_margin", "actual_home_margin", (2022, 2023, 2024, 2025)),
        "cfb": ("pred_margin_home", "actual_margin_home", (2023, 2024, 2025)),
    }
    pred, actual, years = fields[sport]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["season", "week", "game_id", pred, actual]
        )
        writer.writeheader()
        for year in years:
            for i in range(110):
                writer.writerow({
                    "season": year, "week": 1 + i // 8,
                    "game_id": f"{sport}{year}_{i}",
                    pred: (i % 11 - 5) * 1.5,
                    actual: (i % 13 - 6) * 3,
                })


def _board(path, sport, kickoff=KICKOFF, score=2.5):
    field = "kickoff" if sport == "nfl" else "date"
    fields = ["game_id", "season", "week", "home_team", "away_team",
              "model_margin_home", field]
    row = {
        "game_id": "2026_05_BAL_ATL" if sport == "nfl" else "401860002",
        "season": "2026", "week": "5", "home_team": "ATL" if sport == "nfl" else "Atlanta",
        "away_team": "BAL" if sport == "nfl" else "Baltimore",
        "model_margin_home": score, field: kickoff.isoformat(),
    }
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)


def _fixture(tmp_path, sport):
    archive = tmp_path / f"{sport}-historical.csv"
    board = tmp_path / f"{sport}-board.csv"
    _hist(archive, sport)
    _board(board, sport)
    return archive, board


@pytest.mark.parametrize("sport", ["nfl", "cfb"])
def test_real_forecast_contract_frozen_before_kickoff(tmp_path, sport):
    archive, board = _fixture(tmp_path, sport)
    rows = make_snapshots(board, archive, sport, now=NOW)
    assert len(rows) == 1
    r = rows[0]
    assert r["spec"] == VERSION
    assert r["captured_at"] < r["kickoff"]
    assert r["fixed_alpha"] == 1.0
    assert r["sportsbook_odds_used"] is False
    assert r["wager_authorized"] is False
    assert len(r["diagnostic_spreads"]) == 9
    assert r["historical_source_sha256"]
    for item in r["diagnostic_spreads"]:
        assert sum(item["baseline"]) == pytest.approx(1.0)
        assert sum(item["key_number"]) == pytest.approx(1.0)
    assert next(v for v in r["diagnostic_spreads"]
                if v["home_spread"] == -3.0)["baseline"][1] > 0


def test_first_snapshot_never_overwritten_even_if_board_changes(tmp_path):
    archive, board = _fixture(tmp_path, "nfl")
    first = make_snapshots(board, archive, "nfl", now=NOW)
    root = tmp_path / "forecasts"
    assert capture(first, root)["newly_frozen"] == 1
    saved = next(root.glob("*.json")).read_bytes()
    _board(board, "nfl", score=-28.0)
    second = make_snapshots(board, archive, "nfl", now=NOW)
    assert second[0]["independent_home_margin"] == -28.0
    assert capture(second, root) == {
        "newly_frozen": 0, "already_frozen": 1
    }
    assert next(root.glob("*.json")).read_bytes() == saved


def test_post_kickoff_and_last_minute_backfill_are_not_captured(tmp_path):
    archive, board = _fixture(tmp_path, "nfl")
    assert make_snapshots(board, archive, "nfl", now=KICKOFF) == []
    assert make_snapshots(board, archive, "nfl",
                          now=KICKOFF - timedelta(minutes=4)) == []
    _board(board, "nfl", kickoff=NOW - timedelta(hours=1))
    assert make_snapshots(board, archive, "nfl", now=NOW) == []


def test_market_odds_do_not_influence_frozen_football_distribution(tmp_path):
    archive, board = _fixture(tmp_path, "nfl")
    with board.open() as fh:
        rows = list(csv.DictReader(fh))
        fields = list(rows[0])
    first = make_snapshots(board, archive, "nfl", now=NOW)
    fields += ["quant_odds", "market_spread_home", "book"]
    rows[0].update(quant_odds="9999", market_spread_home="-19", book="Synthetic")
    with board.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    second = make_snapshots(board, archive, "nfl", now=NOW)
    assert first[0]["diagnostic_spreads"] == second[0]["diagnostic_spreads"]
    assert first[0]["independent_home_margin"] == second[0]["independent_home_margin"]
    assert first[0]["live_board_sha256"] != second[0]["live_board_sha256"]


def _scoreboard(kickoff=KICKOFF):
    return {
        "events": [{
            "id": "123",
            "date": kickoff.isoformat(),
            "status": {"type": {"completed": True, "name": "STATUS_FINAL"}},
            "competitions": [{"competitors": [
                {"homeAway": "home", "team": {
                    "abbreviation": "ATL", "location": "Atlanta"
                }, "score": "27"},
                {"homeAway": "away", "team": {
                    "abbreviation": "BAL", "location": "Baltimore"
                }, "score": "24"},
            ]}],
        }]
    }


def test_forward_grading_only_after_proven_git_commit_and_final(tmp_path):
    archive, board = _fixture(tmp_path, "nfl")
    source = make_snapshots(board, archive, "nfl", now=NOW)
    root = tmp_path / "forecasts"
    capture(source, root)

    def fetch(_sport, _day):
        return _scoreboard(), "https://scoreboard.example/fixtures"

    missing = grade(
        root, now=KICKOFF + timedelta(hours=6),
        commit_lookup=lambda _p: None, scoreboard_fetch=fetch
    )
    assert missing["graded_games"] == 0
    assert next(iter(missing["ungraded"].values())) == "NO_VERIFIED_PREGAME_PUBLICATION"
    fine = grade(
        root, now=KICKOFF + timedelta(hours=6),
        commit_lookup=lambda _p: NOW + timedelta(minutes=4),
        scoreboard_fetch=fetch,
    )
    assert fine["graded_games"] == 1
    assert fine["scores"]["baseline"]["games"] == 1
    assert fine["roi"] is None
    assert fine["betting_authorized"] is False
    assert fine["status"] == "FORWARD_INSUFFICIENT_SAMPLE"


def test_same_week_game_not_graded_before_kickoff(tmp_path):
    archive, board = _fixture(tmp_path, "nfl")
    root = tmp_path / "forecasts"
    capture(make_snapshots(board, archive, "nfl", now=NOW), root)
    report = grade(
        root, now=KICKOFF - timedelta(minutes=30),
        commit_lookup=lambda _p: NOW + timedelta(minutes=4),
        scoreboard_fetch=lambda *_: (_scoreboard(), "url"),
    )
    assert report["graded_games"] == 0
    assert "PENDING_KICKOFF" in report["ungraded"].values()


def test_paper_result_does_not_reweight_forecast_probabilities(tmp_path):
    archive, board = _fixture(tmp_path, "nfl")
    r = make_snapshots(board, archive, "nfl", now=NOW)[0]
    x = _threeway_score(3, r)
    y = _threeway_score(-18, r)
    assert x != y
    assert r["diagnostic_spreads"] == make_snapshots(
        board, archive, "nfl", now=NOW
    )[0]["diagnostic_spreads"]


def test_invalid_duplicate_game_or_naive_time_fails_closed(tmp_path):
    archive, board = _fixture(tmp_path, "nfl")
    original = board.read_text()
    # Repeated identical rows are expected: NFL dashboard markets share a game.
    board.write_text(original + original.splitlines()[-1] + "\n")
    assert len(make_snapshots(board, archive, "nfl", now=NOW)) == 1
    # Conflicting game-level projections must never silently pick one market row.
    changed = original.splitlines()[-1].replace(",2.5,", ",5.5,")
    board.write_text(original + changed + "\n")
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        make_snapshots(board, archive, "nfl", now=NOW)
    board.write_text(original.replace("+00:00", ""))
    with pytest.raises(ValueError, match="Invalid timestamp"):
        make_snapshots(board, archive, "nfl", now=NOW)


def test_forecast_source_omission_fails_before_generation(tmp_path):
    archive, board = _fixture(tmp_path, "nfl")
    data = archive.read_text()
    archive.write_text(data.replace(",2022,", ",2050,"))
    # The canonical historical set is still internally valid; require all
    # expected prior seasons or refuse to publish an incomplete model.
    rows = list(csv.DictReader(archive.open()))
    rows[:] = [r for r in rows if int(r["season"]) > 2022]
    with archive.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(ValueError, match="prior season"):
        make_snapshots(board, archive, "nfl", now=NOW)
