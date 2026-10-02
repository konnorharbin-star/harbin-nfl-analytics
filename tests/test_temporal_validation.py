import polars as pl

from nfl.temporal_validation import whole_week_partition


def _frame() -> pl.DataFrame:
    rows = []
    game_id = 0
    for season in (2021, 2022, 2023, 2024):
        for week in range(5, 19):
            for game in range(12):
                game_id += 1
                rows.append(
                    {
                        "season": season,
                        "week": week,
                        "game_id": f"g{game_id}",
                        "value": float(game),
                    }
                )
    return pl.DataFrame(rows)


def test_partition_keeps_every_week_wholly_inside_one_section() -> None:
    frame = _frame()
    core, tune, calibration, evaluation, metadata = whole_week_partition(
        frame,
        min_core_rows=300,
        min_section_rows=60,
    )

    sections = [core, tune, calibration, evaluation]
    seen: dict[tuple[int, int], int] = {}
    for section_index, section in enumerate(sections):
        for season, week in section.select(["season", "week"]).unique().iter_rows():
            key = (int(season), int(week))
            assert key not in seen
            seen[key] = section_index

    assert sum(section.height for section in sections) == frame.height
    assert metadata.whole_week_boundaries is True
    assert metadata.selection_uses_evaluation is False
    assert metadata.core.end < metadata.tune.start
    assert metadata.tune.end < metadata.calibration.start
    assert metadata.calibration.end < metadata.evaluation.start
