"""Out-of-time 2026 shadow audit for the frozen recent-form totals candidate.

The candidate structure was selected only from 2022-2025 rolling-origin development
folds. This module freezes that structure, refits its coefficients on completed
2022-2025 development data, and evaluates 2026 without retuning. It is not part of the
canonical fair-score path and cannot earn promotion evidence before the minimum shadow
sample is reached.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import polars as pl

from .data import NFLDataClient
from .oa_residuals import FeatureRidgeModel
from .recent_form import recent_feature_columns
from .recent_form_dataset import build_recent_form_walkforward_dataset
from .shadow_gate import TargetShadowEvidence, paired_bootstrap_evidence

FROZEN_RECENT_ALPHA = 0.20
FROZEN_FEATURE_SET = ("epa_per_play", "success_rate")
FROZEN_RIDGE_ALPHA = 10.0
FROZEN_BLEND_WEIGHT = 0.50
FROZEN_SELECTION_SEASONS = (2022, 2023, 2024, 2025)
FROZEN_TOTAL_FEATURES = recent_feature_columns(
    FROZEN_FEATURE_SET,
    target="total_residual",
)


@dataclass(frozen=True)
class RecentFormShadowAudit:
    training_seasons: tuple[int, ...]
    shadow_season: int
    training_games: int
    shadow_games: int
    shadow_start_week: int
    shadow_end_week: int
    recent_alpha: float
    ridge_alpha: float
    blend_weight: float
    feature_set: tuple[str, ...]
    baseline_total_mae: float
    adjusted_total_mae: float
    baseline_total_rmse: float
    adjusted_total_rmse: float
    total_mae_improvement: float
    total_rmse_improvement: float
    total_evidence: TargetShadowEvidence
    release_state: str
    promotion_eligible: bool
    meaning: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def fit_frozen_recent_total(training: pl.DataFrame) -> FeatureRidgeModel:
    """Fit the frozen totals residual structure on pre-2026 development rows."""

    if training.is_empty():
        raise ValueError("recent-form shadow training data is empty")
    return FeatureRidgeModel(FROZEN_TOTAL_FEATURES, FROZEN_RIDGE_ALPHA).fit(
        training,
        "total_residual",
    )


def apply_frozen_recent_total(
    model: FeatureRidgeModel,
    frame: pl.DataFrame,
) -> pl.DataFrame:
    """Add frozen recent-form totals shadow columns without altering canonical totals."""

    correction = model.predict(frame)
    adjusted = np.asarray(frame.get_column("baseline_total"), dtype=float) + (
        FROZEN_BLEND_WEIGHT * correction
    )
    return frame.with_columns(
        pl.Series("recent_form_total_adjustment", FROZEN_BLEND_WEIGHT * correction),
        pl.Series("recent_form_shadow_total", adjusted),
        pl.lit("SHADOW").alias("recent_form_total_release_state"),
    )


def _error_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> tuple[float, float]:
    error = predicted - actual
    return float(np.mean(np.abs(error))), float(np.sqrt(np.mean(np.square(error))))


def run_recent_form_shadow_audit(
    *,
    training_seasons: tuple[int, ...] = FROZEN_SELECTION_SEASONS,
    shadow_season: int = 2026,
    training_start_week: int = 5,
    shadow_start_week: int = 3,
    score_ridge: float = 8.0,
    bootstrap_iterations: int = 5000,
    confidence: float = 0.95,
    minimum_shadow_games: int = 128,
    client: NFLDataClient | None = None,
    refresh: bool = False,
) -> RecentFormShadowAudit:
    """Refit on frozen development seasons and score completed 2026 games only."""

    if training_seasons != FROZEN_SELECTION_SEASONS:
        raise ValueError(
            "recent-form shadow training seasons are frozen to 2022-2025"
        )
    if max(training_seasons) >= shadow_season:
        raise ValueError("all recent-form training seasons must precede shadow season")

    source = client or NFLDataClient()
    schedule_seasons = sorted({min(training_seasons) - 1, *training_seasons, shadow_season})
    schedules = source.load_schedules(schedule_seasons, refresh=refresh)
    pbp_seasons = sorted({*training_seasons, shadow_season})
    pbp = source.load_pbp(pbp_seasons, refresh=refresh)

    training_frames = [
        build_recent_form_walkforward_dataset(
            schedules,
            pbp.filter(pl.col("season") == season),
            season,
            recent_alpha=FROZEN_RECENT_ALPHA,
            start_week=training_start_week,
            end_week=18,
            score_ridge=score_ridge,
        )
        for season in training_seasons
    ]
    training = pl.concat(training_frames, how="vertical_relaxed").sort(
        ["season", "week", "game_id"]
    )

    shadow = build_recent_form_walkforward_dataset(
        schedules,
        pbp.filter(pl.col("season") == shadow_season),
        shadow_season,
        recent_alpha=FROZEN_RECENT_ALPHA,
        start_week=shadow_start_week,
        end_week=None,
        score_ridge=score_ridge,
    )
    model = fit_frozen_recent_total(training)
    scored = apply_frozen_recent_total(model, shadow)

    actual = np.asarray(scored.get_column("actual_total"), dtype=float)
    baseline = np.asarray(scored.get_column("baseline_total"), dtype=float)
    adjusted = np.asarray(scored.get_column("recent_form_shadow_total"), dtype=float)
    baseline_mae, baseline_rmse = _error_metrics(actual, baseline)
    adjusted_mae, adjusted_rmse = _error_metrics(actual, adjusted)
    evidence = paired_bootstrap_evidence(
        actual,
        baseline,
        adjusted,
        iterations=bootstrap_iterations,
        confidence=confidence,
        minimum_games=minimum_shadow_games,
        seed=20261001,
    )
    release_state = (
        "PROMOTION_EVIDENCE"
        if evidence.status == "PROMOTION_EVIDENCE"
        else "SHADOW"
    )
    return RecentFormShadowAudit(
        training_seasons=training_seasons,
        shadow_season=shadow_season,
        training_games=training.height,
        shadow_games=shadow.height,
        shadow_start_week=int(shadow.get_column("week").min()),
        shadow_end_week=int(shadow.get_column("week").max()),
        recent_alpha=FROZEN_RECENT_ALPHA,
        ridge_alpha=FROZEN_RIDGE_ALPHA,
        blend_weight=FROZEN_BLEND_WEIGHT,
        feature_set=FROZEN_FEATURE_SET,
        baseline_total_mae=baseline_mae,
        adjusted_total_mae=adjusted_mae,
        baseline_total_rmse=baseline_rmse,
        adjusted_total_rmse=adjusted_rmse,
        total_mae_improvement=baseline_mae - adjusted_mae,
        total_rmse_improvement=baseline_rmse - adjusted_rmse,
        total_evidence=evidence,
        release_state=release_state,
        promotion_eligible=evidence.status == "PROMOTION_EVIDENCE",
        meaning=(
            "Frozen recent-form totals shadow evaluation. The specification is not "
            "retuned on 2026 outcomes and remains outside the canonical score path."
        ),
    )
