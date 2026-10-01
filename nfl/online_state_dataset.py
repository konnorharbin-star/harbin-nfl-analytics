"""Chronological dataset for NCAA-style NFL online-state residual research."""

from __future__ import annotations

import polars as pl

from .contracts import DataContractError
from .online_state import OnlineStateConfig, build_online_state_features
from .recency import build_score_walkforward


def build_online_state_dataset(
    schedules: pl.DataFrame,
    config: OnlineStateConfig,
    *,
    start_season: int = 2022,
    end_season: int = 2025,
    start_week: int = 5,
    end_week: int = 18,
    score_ridge: float = 8.0,
) -> pl.DataFrame:
    """Join strict pre-week online state to the canonical independent score baseline."""

    if start_season > end_season:
        raise ValueError("start_season must be <= end_season")

    state = build_online_state_features(schedules, config).filter(
        (pl.col("season") >= start_season)
        & (pl.col("season") <= end_season)
        & (pl.col("week") >= start_week)
        & (pl.col("week") <= end_week)
    )
    if state.is_empty():
        raise DataContractError("online-state feature frame is empty")

    baseline_frames: list[pl.DataFrame] = []
    for season in range(start_season, end_season + 1):
        frame = build_score_walkforward(
            schedules,
            season,
            start_week=start_week,
            end_week=end_week,
            ridge=score_ridge,
        ).select(
            [
                "season",
                "week",
                "game_id",
                pl.col("projected_home_margin").alias("baseline_home_margin"),
                pl.col("projected_total").alias("baseline_total"),
            ]
        )
        baseline_frames.append(frame)

    baseline = pl.concat(baseline_frames, how="vertical_relaxed")
    out = state.join(
        baseline,
        on=["season", "week", "game_id"],
        how="inner",
        validate="1:1",
    )
    if out.height != state.height:
        raise DataContractError(
            f"online-state/baseline join lost {state.height - out.height} game rows"
        )

    return out.with_columns(
        (pl.col("actual_home_margin") - pl.col("baseline_home_margin")).alias(
            "margin_residual"
        ),
        (pl.col("actual_total") - pl.col("baseline_total")).alias("total_residual"),
        pl.lit(config.name).alias("state_config"),
    ).sort(["season", "week", "game_id"])
