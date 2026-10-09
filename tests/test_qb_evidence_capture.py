"""QB evidence capture: source timing, starter and backup identity."""
import csv
from pathlib import Path

import pytest

from research.qb_evidence.capture import build, choose_input


NOW = "2026-10-09T17:00:00+00:00"


def _write_csv(path: Path, columns: list[str], rows: list[list[str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def _pred(tmp_path: Path, sport: str) -> Path:
    file = tmp_path / "model.csv"
    if sport == "nfl":
        _write_csv(
            file,
            ["game_id", "season", "kickoff", "home_team", "away_team",
             "home_expected_qb_id", "away_expected_qb_id"],
            [["game1", "2026", "2026-10-11T20:00:00+00:00",
              "DAL", "TB", "qb1", "qb-away"]],
        )
    else:
        _write_csv(
            file,
            ["game_id", "season", "date", "home_team", "away_team"],
            [["game1", "2026", "2026-10-11T20:00:00+00:00",
              "Dallas", "Tampa Bay"]],
        )
    return file


def _independent(tmp_path: Path, home="DAL", obtained="2026-10-09T16:00:00+00:00") -> Path:
    path = tmp_path / "independent.csv"
    _write_csv(
        path,
        ["game_id", "team", "model_player_id", "role", "depth_rank",
         "observed_at", "obtained_at", "source_url", "verified"],
        [
            ["game1", home, "qb1", "starter", "1",
             "2026-10-09T15:00:00+00:00", obtained,
             "https://example.org/verified", "true"],
            ["game1", home, "qb2", "reserve", "2",
             "2026-10-09T15:00:00+00:00", obtained,
             "https://example.org/verified", "true"],
        ],
    )
    return path


@pytest.mark.parametrize("sport", ["nfl", "cfb"])
def test_model_is_not_source_verification(tmp_path: Path, sport: str) -> None:
    rows, report = build(_pred(tmp_path, sport), sport=sport, capture_at=NOW)
    assert len(rows) == 2
    assert report["upcoming_games"] == 1
    assert report["validated_replacement_ratings_count"] == 0
    assert all(not r["eligible_for_talent_backtest"] for r in rows)
    assert all(r["ea_player_gap_rating_units"] is None for r in rows)
    assert rows[0]["independent_evidence_status"] == "NO_INDEPENDENT_LINEUP"


def test_named_starter_backup_source(tmp_path: Path) -> None:
    records, _ = build(
        _pred(tmp_path, "nfl"),
        sport="nfl", capture_at=NOW,
        verified_lineups=_independent(tmp_path),
    )
    home = next(r for r in records if r["team"] == "DAL")
    assert home["independent_evidence_status"] == "STARTER_AND_BACKUP_CONFIRMED"
    assert home["independent_starter_id"] == "qb1"
    assert home["independent_backup_id"] == "qb2"
    assert home["ea_player_gap_rating_units"] is None


def test_unconfirmed_model_qb_conflict(tmp_path: Path) -> None:
    source = _independent(tmp_path)
    source.write_text(source.read_text().replace(
        "qb1,starter,1", "qb3,starter,1"
    ))
    records, _ = build(
        _pred(tmp_path, "nfl"), sport="nfl",
        capture_at=NOW, verified_lineups=source,
    )
    assert records[0]["independent_evidence_status"] == (
        "STARTER_CONFLICT_WITH_MODEL"
    )


def test_postcapture_information_is_blocked(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="after cutoff"):
        build(
            _pred(tmp_path, "nfl"), sport="nfl", capture_at=NOW,
            verified_lineups=_independent(
                tmp_path, obtained="2026-10-10T16:00:00+00:00"
            ),
        )


def test_kickoff_already_passed_excluded(tmp_path: Path) -> None:
    file = _pred(tmp_path, "nfl")
    file.write_text(file.read_text().replace(
        "2026-10-11T20:00:00+00:00",
        "2026-10-09T12:00:00+00:00",
    ))
    records, summary = build(file, sport="nfl", capture_at=NOW)
    assert records == []
    assert summary["excluded"]["already_kicked_off"] == 1


def test_choose_latest_named_cfb_week(tmp_path: Path) -> None:
    a = tmp_path / "cfb_model_2026_week6.csv"
    b = tmp_path / "cfb_model_2026_week10.csv"
    a.write_text("data\n")
    b.write_text("data\n")
    assert choose_input(str(tmp_path / "cfb_model_*.csv")) == b


def test_no_fake_timezones(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="explicit UTC offset"):
        build(
            _pred(tmp_path, "nfl"), sport="nfl",
            capture_at="2026-10-09T17:00:00",
        )


def test_market_duplicates_collapse_only_when_qb_identity_matches(tmp_path: Path) -> None:
    path = _pred(tmp_path, "nfl")
    original = path.read_text().splitlines()
    path.write_text("\n".join([*original, original[1]]) + "\n")
    rows, summary = build(path, sport="nfl", capture_at=NOW)
    assert summary["model_game_rows"] == 2
    assert summary["unique_game_rows"] == 1
    assert summary["excluded"]["duplicate_market_rows"] == 1
    assert len(rows) == 2

    path.write_text(path.read_text().replace(
        "qb1,qb-away\n", "different,qb-away\n", 1
    ))
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        build(path, sport="nfl", capture_at=NOW)


