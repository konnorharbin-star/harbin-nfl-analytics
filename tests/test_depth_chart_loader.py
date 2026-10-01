from __future__ import annotations

import polars as pl
import pytest

from nfl import data
from nfl.contracts import DataContractError
from nfl.data import NFLDataClient, _normalize_depth_chart_season_frame


def _depth_frame(*, dt: str, team: str = "ARI") -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "dt": dt,
                "team": team,
                "player_name": "Example Player",
                "gsis_id": "00-0000001",
                "pos_abb": "QB",
                "pos_rank": 1,
            }
        ]
    )


def test_depth_loader_uses_one_source_request_per_season_and_never_dt_year() -> None:
    calls: list[list[int]] = []

    def fake_load_depth_charts(years: list[int]) -> pl.DataFrame:
        calls.append(list(years))
        assert len(years) == 1
        season = years[0]
        if season == 2025:
            # The 2025 NFL season legitimately extends into calendar year 2026.
            return _depth_frame(dt="2026-03-14T07:32:09Z")
        if season == 2026:
            return _depth_frame(dt="2026-10-01T14:25:58Z", team="ATL")
        raise AssertionError(f"unexpected season {season}")

    original = data.nfl.load_depth_charts
    data.nfl.load_depth_charts = fake_load_depth_charts
    try:
        client = NFLDataClient(cache_dir="/tmp/harbin_depth_test_scope")
        frame = client.load_depth_charts([2025, 2026], refresh=True)
    finally:
        data.nfl.load_depth_charts = original

    assert calls == [[2025], [2026]]
    assert frame.height == 2
    by_team = {row["team"]: row for row in frame.iter_rows(named=True)}
    assert by_team["ARI"]["season"] == 2025
    assert by_team["ARI"]["dt"].startswith("2026-")
    assert by_team["ATL"]["season"] == 2026


def test_depth_loader_rebuilds_legacy_cache_missing_season(tmp_path, monkeypatch) -> None:
    client = NFLDataClient(cache_dir=tmp_path)
    cache = client._cache_path("depth_charts", [2026])
    _depth_frame(dt="2026-09-01T12:00:00Z").write_parquet(cache)

    calls: list[list[int]] = []

    def fake_load_depth_charts(years: list[int]) -> pl.DataFrame:
        calls.append(list(years))
        return _depth_frame(dt="2026-10-01T14:25:58Z")

    monkeypatch.setattr(data.nfl, "load_depth_charts", fake_load_depth_charts)
    frame = client.load_depth_charts(2026)

    assert calls == [[2026]]
    assert frame.get_column("season").unique().to_list() == [2026]
    cached = pl.read_parquet(cache)
    assert "season" in cached.columns
    assert cached.get_column("season").unique().to_list() == [2026]


def test_depth_loader_reuses_normalized_cache_without_source_call(
    tmp_path,
    monkeypatch,
) -> None:
    client = NFLDataClient(cache_dir=tmp_path)
    cache = client._cache_path("depth_charts", [2026])
    _depth_frame(dt="2026-10-01T14:25:58Z").with_columns(
        pl.lit(2026).cast(pl.Int32).alias("season")
    ).write_parquet(cache)

    def fail_load(_years: list[int]) -> pl.DataFrame:
        raise AssertionError("normalized cache should have been reused")

    monkeypatch.setattr(data.nfl, "load_depth_charts", fail_load)
    frame = client.load_depth_charts(2026)

    assert frame.height == 1
    assert frame.get_column("season").to_list() == [2026]


def test_native_depth_season_mismatch_fails_closed() -> None:
    frame = _depth_frame(dt="2026-10-01T14:25:58Z").with_columns(
        pl.lit(2025).alias("season")
    )
    with pytest.raises(DataContractError, match="unexpected season"):
        _normalize_depth_chart_season_frame(frame, 2026)
