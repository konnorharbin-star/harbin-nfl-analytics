"""Regression tests for historical score-error decomposition."""
import csv
from pathlib import Path

import pytest

from scripts.score_error_diagnostics import (
    extreme_games,
    load_games,
    metrics,
    report,
)


def _sample(path: Path, sport: str = "nfl") -> Path:
    if sport == "nfl":
        names = [
            "game_id", "season", "week", "home_team", "away_team",
            "projected_home_margin", "projected_total",
            "actual_home_margin", "actual_total",
        ]
    else:
        names = [
            "game_id", "season", "week", "home_team", "away_team",
            "pred_margin_home", "pred_total",
            "actual_margin_home", "actual_total",
        ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerow(dict(zip(names, [
            "g1", 2025, 5, "H", "A", 4, 44, -8, 40
        ], strict=True)))
        writer.writerow(dict(zip(names, [
            "g2", 2025, 6, "H", "B", -7, 45, -7, 45
        ], strict=True)))
    return path


@pytest.mark.parametrize("sport", ["nfl", "cfb"])
def test_home_away_decomposition_and_winner(sport: str, tmp_path: Path) -> None:
    rows = load_games(_sample(tmp_path / "games.csv", sport), sport)
    a = rows[0]
    assert a["projected_home"] == 24
    assert a["projected_away"] == 20
    assert a["actual_home"] == 16
    assert a["actual_away"] == 24
    assert a["home_error"] == 8
    assert a["away_error"] == -4
    assert a["margin_error"] == 12
    assert a["total_error"] == 4
    stats = metrics(rows)
    assert stats["margin_mae"] == 6
    assert stats["total_mae"] == 2
    assert stats["winner_accuracy_excluding_ties"] == 0.5
    result = report(rows, sport=sport, source_sha256="fixture")
    assert result["overall"]["games"] == 2
    assert result["production_model_adjustment_points"] == 0
    assert extreme_games(rows)[0]["game_id"] == "g1"


def test_duplicate_and_invalid_game_fail_closed(tmp_path: Path) -> None:
    path = _sample(tmp_path / "games.csv")
    with path.open("a") as handle:
        handle.write("g1,2025,7,H,C,2,42,1,41\n")
    with pytest.raises(ValueError, match="Duplicate"):
        load_games(path, "nfl")

    path = _sample(tmp_path / "new.csv")
    text = path.read_text().replace("g2,2025,6,H,B,-7,45,-7,45",
                                     "g2,2025,6,H,B,-7,45,-7,-2")
    path.write_text(text)
    with pytest.raises(ValueError, match="Impossible"):
        load_games(path, "nfl")
