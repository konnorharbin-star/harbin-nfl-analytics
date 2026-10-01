from __future__ import annotations

import polars as pl

from nfl.data import NFLDataClient


def test_modern_depth_chart_schema_derives_nfl_season(monkeypatch, tmp_path) -> None:
    frame = pl.DataFrame(
        [
            {
                "dt": "2026-09-30",
                "team": "BUF",
                "player_name": "Quarterback One",
                "gsis_id": "qb-1",
                "pos_abb": "QB",
                "pos_rank": 1,
            },
            {
                "dt": "2027-01-10",
                "team": "BUF",
                "player_name": "Quarterback Two",
                "gsis_id": "qb-2",
                "pos_abb": "QB",
                "pos_rank": 2,
            },
        ]
    )

    monkeypatch.setattr("nfl.data.nfl.load_depth_charts", lambda years: frame)
    client = NFLDataClient(cache_dir=tmp_path)
    loaded = client.load_depth_charts([2026], refresh=True)

    assert loaded.get_column("team").to_list() == ["BUF", "BUF"]
    assert loaded.get_column("season").to_list() == [2026, 2026]
    assert "dt" in loaded.columns


def test_legacy_depth_chart_schema_maps_club_code(monkeypatch, tmp_path) -> None:
    frame = pl.DataFrame(
        [
            {
                "season": 2024,
                "week": 4,
                "club_code": "NE",
                "football_name": "Legacy Player",
                "gsis_id": "legacy-1",
                "position": "WR",
                "depth_team": 1,
            }
        ]
    )

    monkeypatch.setattr("nfl.data.nfl.load_depth_charts", lambda years: frame)
    client = NFLDataClient(cache_dir=tmp_path)
    loaded = client.load_depth_charts([2024], refresh=True)

    assert loaded.get_column("team").to_list() == ["NE"]
    assert loaded.get_column("season").to_list() == [2024]
    assert "club_code" not in loaded.columns
