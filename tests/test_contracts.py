import polars as pl
import pytest

from nfl.contracts import DataContractError, normalize_seasons, require_columns, require_unique


def test_normalize_seasons_sorts_and_deduplicates() -> None:
    assert normalize_seasons([2025, 2023, 2025, 2024]) == [2023, 2024, 2025]


def test_normalize_seasons_rejects_empty() -> None:
    with pytest.raises(DataContractError):
        normalize_seasons([])


def test_require_columns_fails_closed() -> None:
    frame = pl.DataFrame({"season": [2025]})
    with pytest.raises(DataContractError, match="week"):
        require_columns(frame, {"season", "week"}, "example")


def test_require_unique_detects_duplicate_key() -> None:
    frame = pl.DataFrame({"game_id": ["a", "a"]})
    with pytest.raises(DataContractError, match="duplicated"):
        require_unique(frame, ["game_id"], "example")
