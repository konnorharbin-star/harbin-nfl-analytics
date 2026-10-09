"""Prevent team priors from being misread as prospective EA player evidence."""
import csv
import json
from datetime import datetime, timezone

import pytest

from research.ea_team_prior.audit import audit, canonical, read_games, read_ratings


def test_nfl_crosswalk_and_cfb_abbreviations():
    assert canonical("Buccaneers", "nfl") == "TB"
    assert canonical("Rams", "nfl") == "LA"
    assert canonical("Western Michigan", "cfb") == "w michigan"
    assert canonical("Miami (OH)", "cfb") == "miami oh"
    assert canonical("Miami", "cfb") == "miami"
    assert canonical("North Dakota State", "cfb") == "ndsu"


def test_source_validation_and_duplicate_fails(tmp_path):
    source = tmp_path / "team.csv"
    meta = tmp_path / "source.json"
    source.write_text("team,offense,defense,overall\nCowboys,87,80,84\n")
    meta.write_text(json.dumps({
        "source_url": "https://www.ea.com/example",
        "published_date": "2026-07-31", "season": 2026,
        "ratings_level": "TEAM_NOT_PLAYER",
    }))
    ratings, proof = read_ratings(source, meta, "nfl")
    assert ratings["DAL"] == (87, 80, 84)
    assert proof["article_revision_not_independently_archived"] is True
    source.write_text(source.read_text() + "Cowboys,87,80,84\n")
    with pytest.raises(ValueError, match="Duplicate"):
        read_ratings(source, meta, "nfl")


def test_prepublication_excluded_and_calibration_is_disabled():
    ratings = {"DAL": (87, 80, 84), "TB": (83, 79, 81)}
    def game(kickoff):
        return {
            "game_id": "g1", "kickoff": datetime.fromisoformat(kickoff),
            "observed": datetime(2026, 10, 6, tzinfo=timezone.utc),
            "home": "DAL", "away": "TB",
            "projected_margin": 3, "actual_margin": -8,
            "projected_total": 50, "actual_total": 40,
        }
    result, rows = audit(
        ratings, [game("2026-10-09T00:15:00+00:00")],
        "2026-07-31", "nfl",
    )
    assert result["matched_games"] == 1
    assert rows[0]["talent_matchup_diff_units"] == 5
    assert rows[0]["base_margin_error"] == 11
    assert result["candidate_margin_mae"] is None
    assert result["approved_prediction_adjustment"] == 0
    result, rows = audit(
        ratings, [game("2026-07-31T22:00:00+00:00")],
        "2026-07-31", "nfl",
    )
    assert result["matched_games"] == 0
    assert result["exclusions"]["kickoff_before_or_on_publication_date"] == 1


def test_read_games_requires_point_in_time_snapshot(tmp_path):
    games = tmp_path / "games.csv"
    with games.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "season", "game_id", "home_team", "away_team", "kickoff",
            "captured_at", "baseline_home_margin", "baseline_total",
            "home_score", "away_score",
        ])
        writer.writerow([
            "2026", "g1", "DAL", "TB", "2026-10-09T00:15:00+00:00",
            "2026-10-10T00:00:00+00:00", "3", "50", "16", "24",
        ])
    valid, excluded = read_games(games, "nfl")
    assert valid == []
    assert excluded["late_or_nonpregame_snapshot"] == 1


def test_official_data_coverage():
    from pathlib import Path

    for sport, count in (("nfl", 32), ("cfb", 138)):
        # Runs in both repositories against their respective local sport.
        directory = Path("research/ea_team_prior")
        meta_path = directory / "source_metadata.json"
        meta = json.loads(meta_path.read_text())
        if meta["game"].startswith("Madden") != (sport == "nfl"):
            continue
        ratings, _ = read_ratings(
            directory / "team_ratings_2026.csv", meta_path, sport
        )
        assert len(ratings) == count
