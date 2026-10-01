from datetime import datetime, timedelta

import math
import polars as pl
import pytest

from nfl.contracts import DataContractError
from nfl.market_backtest import (
    compare_selected_market_snapshots,
    grade_market_comparisons,
    summarize_qualified_bets,
)
from nfl.market_history import select_market_snapshots_asof, validate_market_history
from nfl.probability import GaussianScoreDistribution


def _distribution() -> GaussianScoreDistribution:
    rows = 140
    projected_margin = [float((index % 13) - 6) for index in range(rows)]
    projected_total = [42.0 + float(index % 9) for index in range(rows)]
    frame = pl.DataFrame(
        {
            "projected_home_margin": projected_margin,
            "projected_total": projected_total,
            "actual_home_margin": [
                projected_margin[index] + float(((index * 7) % 25) - 12)
                for index in range(rows)
            ],
            "actual_total": [
                projected_total[index] + float(((index * 5) % 29) - 14)
                for index in range(rows)
            ],
        }
    )
    return GaussianScoreDistribution().fit(frame)


def _row(
    *,
    snapshot_id: str,
    captured_at: datetime,
    side: str,
    line: float,
    odds: int,
) -> dict[str, object]:
    return {
        "game_id": "2025_05_AAA_BBB",
        "market_type": "spread",
        "side": side,
        "line": line,
        "american_odds": odds,
        "provider": "fixture",
        "book": "book-a",
        "captured_at": captured_at,
        "snapshot_id": snapshot_id,
        "source_event_id": "event-123",
    }


def test_asof_selection_never_mixes_sides_across_snapshots() -> None:
    noon = datetime(2025, 10, 1, 12, 0)
    history = pl.DataFrame(
        [
            _row(snapshot_id="s1", captured_at=noon, side="home", line=-3.5, odds=-110),
            _row(snapshot_id="s1", captured_at=noon, side="away", line=3.5, odds=-110),
            _row(
                snapshot_id="s2",
                captured_at=noon + timedelta(hours=1),
                side="home",
                line=-4.0,
                odds=-105,
            ),
            _row(
                snapshot_id="s3",
                captured_at=noon + timedelta(hours=2),
                side="home",
                line=-4.5,
                odds=-110,
            ),
            _row(
                snapshot_id="s3",
                captured_at=noon + timedelta(hours=2),
                side="away",
                line=4.5,
                odds=-110,
            ),
        ]
    )
    decisions = pl.DataFrame(
        {
            "game_id": ["2025_05_AAA_BBB"],
            "decision_time": [noon + timedelta(hours=1, minutes=30)],
        }
    )

    selected = select_market_snapshots_asof(history, decisions)

    assert selected.height == 2
    assert set(selected.get_column("snapshot_id").to_list()) == {"s1"}
    assert set(selected.get_column("side").to_list()) == {"home", "away"}


def test_asof_selection_can_reject_stale_quotes() -> None:
    noon = datetime(2025, 10, 1, 12, 0)
    history = pl.DataFrame(
        [
            _row(snapshot_id="s1", captured_at=noon, side="home", line=-3.5, odds=-110),
            _row(snapshot_id="s1", captured_at=noon, side="away", line=3.5, odds=-110),
        ]
    )
    decisions = pl.DataFrame(
        {
            "game_id": ["2025_05_AAA_BBB"],
            "decision_time": [noon + timedelta(hours=3)],
        }
    )

    selected = select_market_snapshots_asof(history, decisions, max_quote_age_minutes=60)

    assert selected.is_empty()


def test_market_history_requires_reproducible_provenance() -> None:
    noon = datetime(2025, 10, 1, 12, 0)
    history = pl.DataFrame(
        [
            _row(snapshot_id="", captured_at=noon, side="home", line=-3.5, odds=-110),
            _row(snapshot_id="", captured_at=noon, side="away", line=3.5, odds=-110),
        ]
    )

    with pytest.raises(DataContractError, match="empty snapshot_id"):
        validate_market_history(history)


def test_point_in_time_comparison_preserves_provenance_and_grades_price() -> None:
    noon = datetime(2025, 10, 1, 12, 0)
    history = pl.DataFrame(
        [
            _row(snapshot_id="s1", captured_at=noon, side="home", line=-3.5, odds=120),
            _row(snapshot_id="s1", captured_at=noon, side="away", line=3.5, odds=-140),
        ]
    )
    decisions = pl.DataFrame(
        {
            "game_id": ["2025_05_AAA_BBB"],
            "decision_time": [noon + timedelta(minutes=20)],
        }
    )
    selected = select_market_snapshots_asof(history, decisions)
    projections = pl.DataFrame(
        {
            "game_id": ["2025_05_AAA_BBB"],
            "projected_home_margin": [6.0],
            "projected_total": [46.0],
        }
    )

    comparisons = compare_selected_market_snapshots(projections, selected, _distribution())
    outcomes = pl.DataFrame(
        {
            "game_id": ["2025_05_AAA_BBB"],
            "actual_home_margin": [7.0],
            "actual_total": [44.0],
        }
    )
    graded = grade_market_comparisons(comparisons, outcomes)
    home = graded.filter(pl.col("side") == "home").row(0, named=True)

    assert home["provider"] == "fixture"
    assert home["snapshot_id"] == "s1"
    assert home["captured_at"] == noon
    assert home["decision_time"] == noon + timedelta(minutes=20)
    assert home["result"] == "win"
    assert math.isclose(home["net_units"], 1.2)


def test_summary_requires_explicit_edge_and_ev_thresholds() -> None:
    frame = pl.DataFrame(
        {
            "result": ["win", "loss", "push"],
            "net_units": [1.0, -1.0, 0.0],
            "probability_edge": [0.06, 0.04, 0.08],
            "expected_value_per_unit": [0.05, 0.03, 0.07],
        }
    )

    summary = summarize_qualified_bets(
        frame,
        min_probability_edge=0.05,
        min_expected_value_per_unit=0.04,
    )

    assert summary.bets == 2
    assert summary.wins == 1
    assert summary.losses == 0
    assert summary.pushes == 1
    assert math.isclose(summary.net_units, 1.0)
    assert math.isclose(summary.roi_per_unit_staked, 0.5)
