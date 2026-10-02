"""Whole-week chronological validation primitives for NFL model research."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import polars as pl

from .contracts import DataContractError, require_columns


@dataclass(frozen=True)
class TemporalSpan:
    rows: int
    start: tuple[int, int] | None
    end: tuple[int, int] | None


@dataclass(frozen=True)
class WholeWeekPartitionMetadata:
    core: TemporalSpan
    tune: TemporalSpan
    calibration: TemporalSpan
    evaluation: TemporalSpan
    whole_week_boundaries: bool
    selection_uses_evaluation: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def sort_chronologically(frame: pl.DataFrame) -> pl.DataFrame:
    """Return a stable season/week/game ordering without consulting outcomes."""

    require_columns(frame, {"season", "week"}, "temporal_validation")
    sort_columns = ["season", "week"]
    if "gameday" in frame.columns:
        sort_columns.append("gameday")
    if "game_id" in frame.columns:
        sort_columns.append("game_id")
    return frame.sort(sort_columns)


def _week_key(frame: pl.DataFrame, *, first: bool) -> tuple[int, int] | None:
    if frame.is_empty():
        return None
    row = frame.row(0 if first else -1, named=True)
    return int(row["season"]), int(row["week"])


def temporal_span(frame: pl.DataFrame) -> TemporalSpan:
    ordered = sort_chronologically(frame)
    return TemporalSpan(
        rows=ordered.height,
        start=_week_key(ordered, first=True),
        end=_week_key(ordered, first=False),
    )


def assert_strict_prior(earlier: pl.DataFrame, later: pl.DataFrame) -> None:
    """Require the last earlier week to precede the first later week."""

    if earlier.is_empty() or later.is_empty():
        raise DataContractError("temporal partitions must not be empty")
    earlier_end = _week_key(sort_chronologically(earlier), first=False)
    later_start = _week_key(sort_chronologically(later), first=True)
    if earlier_end is None or later_start is None or not earlier_end < later_start:
        raise DataContractError(
            f"temporal leakage: earlier partition ends {earlier_end}, "
            f"later partition begins {later_start}"
        )


def _week_blocks(frame: pl.DataFrame) -> list[tuple[int, int, int]]:
    ordered = sort_chronologically(frame)
    grouped = (
        ordered.group_by(["season", "week"], maintain_order=True)
        .len()
        .rename({"len": "rows"})
    )
    return [
        (int(row["season"]), int(row["week"]), int(row["rows"]))
        for row in grouped.iter_rows(named=True)
    ]


def _choose_boundary(
    cumulative_rows: list[int],
    *,
    target: float,
    low: int,
    high: int,
) -> int:
    eligible = [value for value in cumulative_rows if low <= value <= high]
    if not eligible:
        raise DataContractError(
            f"cannot place a whole-week boundary within row bounds [{low}, {high}]"
        )
    return min(eligible, key=lambda value: (abs(value - target), value))


def whole_week_partition(
    frame: pl.DataFrame,
    *,
    min_core_rows: int = 500,
    min_section_rows: int = 90,
) -> tuple[
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    WholeWeekPartitionMetadata,
]:
    """Split history into core/tune/calibration/evaluation on complete NFL weeks.

    The approximate row targets mirror the NCAA validation architecture: 58% core,
    14% tune, 14% calibration, 14% untouched evaluation. The exact cut moves to the
    nearest valid whole-week boundary while respecting minimum section sizes.
    """

    if min_core_rows <= 0 or min_section_rows <= 0:
        raise ValueError("minimum partition sizes must be positive")

    ordered = sort_chronologically(frame)
    n = ordered.height
    required_rows = int(min_core_rows) + 3 * int(min_section_rows)
    if n < required_rows:
        raise DataContractError(
            f"whole-week validation requires at least {required_rows} rows; got {n}"
        )

    blocks = _week_blocks(ordered)
    if len(blocks) < 8:
        raise DataContractError(
            f"whole-week validation requires at least 8 season/week blocks; got {len(blocks)}"
        )

    cumulative: list[int] = []
    running = 0
    for _, _, rows in blocks:
        running += rows
        cumulative.append(running)

    b1 = _choose_boundary(
        cumulative,
        target=0.58 * n,
        low=int(min_core_rows),
        high=n - 3 * int(min_section_rows),
    )
    b2 = _choose_boundary(
        cumulative,
        target=0.72 * n,
        low=b1 + int(min_section_rows),
        high=n - 2 * int(min_section_rows),
    )
    b3 = _choose_boundary(
        cumulative,
        target=0.86 * n,
        low=b2 + int(min_section_rows),
        high=n - int(min_section_rows),
    )

    core = ordered.slice(0, b1)
    tune = ordered.slice(b1, b2 - b1)
    calibration = ordered.slice(b2, b3 - b2)
    evaluation = ordered.slice(b3, n - b3)

    assert_strict_prior(core, tune)
    assert_strict_prior(tune, calibration)
    assert_strict_prior(calibration, evaluation)

    metadata = WholeWeekPartitionMetadata(
        core=temporal_span(core),
        tune=temporal_span(tune),
        calibration=temporal_span(calibration),
        evaluation=temporal_span(evaluation),
        whole_week_boundaries=True,
        selection_uses_evaluation=False,
    )
    return core, tune, calibration, evaluation, metadata
