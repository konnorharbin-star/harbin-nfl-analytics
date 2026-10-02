"""Combined point-in-time NFL feature matrix for nonlinear residual research.

The Stage 22 matrix joins four independently leakage-safe walk-forward datasets:
core PBP efficiency, situational PBP, drive efficiency, and quarterback state.
Sportsbook and postgame target fields are never admitted as model features.
"""

from __future__ import annotations

from collections.abc import Iterable

import polars as pl

from .contracts import DataContractError
from .dataset import build_walkforward_dataset
from .drive_dataset import build_drive_walkforward_dataset
from .qb_dataset import build_qb_walkforward_dataset
from .situational_dataset import build_situational_walkforward_dataset

JOIN_KEYS = ("season", "week", "game_id")
IDENTIFIER_COLUMNS = {
    "season",
    "week",
    "game_id",
    "gameday",
    "home_team",
    "away_team",
    "home_qb_proxy_id",
    "away_qb_proxy_id",
}
TARGET_OR_LEAKAGE_FRAGMENTS = (
    "actual_",
    "target_",
    "_residual",
    "market",
    "odds",
    "moneyline",
    "closing",
    "close_",
    "_close",
    "clv",
    "stake",
    "postgame",
    "result_",
)
SUPPLEMENTAL_DROP_PREFIXES = (
    "baseline_",
    "actual_",
)
SUPPLEMENTAL_DROP_COLUMNS = {
    "margin_residual",
    "total_residual",
    "gameday",
    "home_team",
    "away_team",
}


def _safe_numeric_columns(
    frame: pl.DataFrame,
    *,
    exclude: Iterable[str] = (),
) -> list[str]:
    excluded = set(exclude) | IDENTIFIER_COLUMNS
    columns: list[str] = []
    for name, dtype in frame.schema.items():
        lower = name.lower()
        if name in excluded:
            continue
        if any(fragment in lower for fragment in TARGET_OR_LEAKAGE_FRAGMENTS):
            continue
        if not dtype.is_numeric():
            continue
        columns.append(name)
    return sorted(columns)


def ensemble_feature_columns(frame: pl.DataFrame) -> tuple[str, ...]:
    """Return numeric pregame-only features admitted to the nonlinear model."""

    columns = _safe_numeric_columns(frame)
    if not columns:
        raise DataContractError("ensemble feature matrix contains no safe numeric features")
    return tuple(columns)


def assert_no_feature_leakage(columns: Iterable[str]) -> None:
    """Fail closed if a proposed model feature name violates the pregame contract."""

    bad: list[str] = []
    for name in columns:
        lower = str(name).lower()
        if name in IDENTIFIER_COLUMNS or any(
            fragment in lower for fragment in TARGET_OR_LEAKAGE_FRAGMENTS
        ):
            bad.append(str(name))
    if bad:
        raise DataContractError(
            "ensemble feature contract contains leakage/identifier columns: "
            + ", ".join(sorted(bad))
        )


def _supplemental_columns(frame: pl.DataFrame) -> list[str]:
    columns: list[str] = []
    for name, dtype in frame.schema.items():
        if name in JOIN_KEYS or name in SUPPLEMENTAL_DROP_COLUMNS:
            continue
        if any(name.startswith(prefix) for prefix in SUPPLEMENTAL_DROP_PREFIXES):
            continue
        if not dtype.is_numeric():
            continue
        lower = name.lower()
        if any(fragment in lower for fragment in TARGET_OR_LEAKAGE_FRAGMENTS):
            continue
        columns.append(name)
    return sorted(columns)


def _join_namespace(
    base: pl.DataFrame,
    extra: pl.DataFrame,
    *,
    prefix: str,
) -> pl.DataFrame:
    if extra.is_empty():
        raise DataContractError(f"{prefix} feature dataset is empty")
    columns = _supplemental_columns(extra)
    renamed = {
        column: f"{prefix}__{column}"
        for column in columns
        if column not in JOIN_KEYS
    }
    selected = extra.select(
        [*JOIN_KEYS, *columns]
    ).rename(renamed)
    joined = base.join(
        selected,
        on=list(JOIN_KEYS),
        how="inner",
        validate="1:1",
    )
    if joined.height != base.height or joined.height != extra.height:
        raise DataContractError(
            f"{prefix} feature join changed coverage: "
            f"base={base.height}, extra={extra.height}, joined={joined.height}"
        )
    return joined


def build_ensemble_walkforward_dataset(
    schedules: pl.DataFrame,
    pbp: pl.DataFrame,
    player_stats: pl.DataFrame,
    season: int,
    *,
    start_week: int = 5,
    end_week: int = 18,
    score_ridge: float = 8.0,
) -> pl.DataFrame:
    """Build one season of joined point-in-time NFL residual features."""

    core = build_walkforward_dataset(
        schedules,
        pbp,
        season,
        start_week=start_week,
        end_week=end_week,
        ridge=score_ridge,
    )
    situational = build_situational_walkforward_dataset(
        schedules,
        pbp,
        season,
        start_week=start_week,
        end_week=end_week,
        score_ridge=score_ridge,
    )
    drive = build_drive_walkforward_dataset(
        schedules,
        pbp,
        season,
        start_week=start_week,
        end_week=end_week,
        score_ridge=score_ridge,
    )
    qb = build_qb_walkforward_dataset(
        schedules,
        player_stats.filter(pl.col("season") == season),
        season,
        start_week=start_week,
        end_week=end_week,
        score_ridge=score_ridge,
    )

    combined = _join_namespace(core, situational, prefix="situ")
    combined = _join_namespace(combined, drive, prefix="drive")
    combined = _join_namespace(combined, qb, prefix="qb")
    features = ensemble_feature_columns(combined)
    assert_no_feature_leakage(features)
    return combined.sort(list(JOIN_KEYS))
