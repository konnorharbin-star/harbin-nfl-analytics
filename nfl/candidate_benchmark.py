"""Rolling-origin scoreboard for NFL-native fair-score research candidates.

This module compares existing research layers on common chronological folds without
changing the canonical projection. 2023-2025 are development evidence because those
seasons have already informed repository design. Any selected candidate is eligible
only for a future 2026 SHADOW experiment; baseline/zero adjustment is always a valid
selection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .data import NFLDataClient
from .oa_audit import OAAudit, run_oa_audit
from .prior_audit import PriorAudit, run_prior_audit
from .qb_audit import QBAudit, run_qb_audit
from .recency_audit import RecencyAudit, run_recency_audit

DEFAULT_FOLDS: tuple[tuple[int, int], ...] = ((2023, 2024), (2024, 2025))


@dataclass(frozen=True)
class TargetFoldEvidence:
    validation_season: int
    holdout_season: int
    games: int
    baseline_mae: float
    adjusted_mae: float | None
    baseline_rmse: float
    adjusted_rmse: float | None
    mae_improvement: float
    rmse_improvement: float
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateEvidence:
    name: str
    margin_folds: tuple[TargetFoldEvidence, ...]
    total_folds: tuple[TargetFoldEvidence, ...]
    margin_eligible: bool
    total_eligible: bool
    margin_robustness_score: float
    total_robustness_score: float

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["margin_folds"] = [value.to_dict() for value in self.margin_folds]
        result["total_folds"] = [value.to_dict() for value in self.total_folds]
        return result


def _target_fold(
    *,
    validation_season: int,
    holdout_season: int,
    games: int,
    baseline_mae: float,
    adjusted_mae: float | None,
    baseline_rmse: float,
    adjusted_rmse: float | None,
    passed: bool,
    detail: str,
) -> TargetFoldEvidence:
    mae_improvement = 0.0 if adjusted_mae is None else baseline_mae - adjusted_mae
    rmse_improvement = 0.0 if adjusted_rmse is None else baseline_rmse - adjusted_rmse
    return TargetFoldEvidence(
        validation_season=validation_season,
        holdout_season=holdout_season,
        games=games,
        baseline_mae=baseline_mae,
        adjusted_mae=adjusted_mae,
        baseline_rmse=baseline_rmse,
        adjusted_rmse=adjusted_rmse,
        mae_improvement=mae_improvement,
        rmse_improvement=rmse_improvement,
        passed=bool(passed),
        detail=detail,
    )


def _score(folds: tuple[TargetFoldEvidence, ...]) -> float:
    """Weighted mean relative error reduction across held-out folds."""

    total_games = sum(value.games for value in folds)
    if total_games <= 0:
        return 0.0
    weighted = 0.0
    for value in folds:
        mae_relative = value.mae_improvement / value.baseline_mae
        rmse_relative = value.rmse_improvement / value.baseline_rmse
        weighted += value.games * ((mae_relative + rmse_relative) / 2.0)
    return weighted / total_games


def _candidate(
    name: str,
    margin_folds: list[TargetFoldEvidence],
    total_folds: list[TargetFoldEvidence],
) -> CandidateEvidence:
    margin = tuple(margin_folds)
    total = tuple(total_folds)
    margin_score = _score(margin)
    total_score = _score(total)
    return CandidateEvidence(
        name=name,
        margin_folds=margin,
        total_folds=total,
        margin_eligible=bool(margin) and all(value.passed for value in margin) and margin_score > 0,
        total_eligible=bool(total) and all(value.passed for value in total) and total_score > 0,
        margin_robustness_score=margin_score,
        total_robustness_score=total_score,
    )


def _recency_fold(audit: RecencyAudit) -> tuple[TargetFoldEvidence, TargetFoldEvidence]:
    enabled = audit.selected_half_life_weeks is not None
    detail = (
        f"half_life_weeks={audit.selected_half_life_weeks}"
        if enabled
        else "validation selected baseline / zero recency"
    )
    # Recency is one shared score-model change, so it must clear its complete nested
    # candidate gate before either target can count as passed.
    passed = bool(audit.candidate_pass)
    return (
        _target_fold(
            validation_season=audit.validation_season,
            holdout_season=audit.holdout_season,
            games=audit.holdout_games,
            baseline_mae=audit.holdout_baseline_margin_mae,
            adjusted_mae=audit.holdout_recency_margin_mae,
            baseline_rmse=audit.holdout_baseline_margin_rmse,
            adjusted_rmse=audit.holdout_recency_margin_rmse,
            passed=passed,
            detail=detail,
        ),
        _target_fold(
            validation_season=audit.validation_season,
            holdout_season=audit.holdout_season,
            games=audit.holdout_games,
            baseline_mae=audit.holdout_baseline_total_mae,
            adjusted_mae=audit.holdout_recency_total_mae,
            baseline_rmse=audit.holdout_baseline_total_rmse,
            adjusted_rmse=audit.holdout_recency_total_rmse,
            passed=passed,
            detail=detail,
        ),
    )


def _prior_fold(audit: PriorAudit) -> tuple[TargetFoldEvidence, TargetFoldEvidence]:
    detail = f"prior_weight={audit.selected_prior_weight}"
    passed = bool(audit.candidate_pass)
    return (
        _target_fold(
            validation_season=audit.validation_season,
            holdout_season=audit.holdout_season,
            games=audit.holdout_games,
            baseline_mae=audit.holdout_baseline_margin_mae,
            adjusted_mae=audit.holdout_prior_margin_mae,
            baseline_rmse=audit.holdout_baseline_margin_rmse,
            adjusted_rmse=audit.holdout_prior_margin_rmse,
            passed=passed,
            detail=detail,
        ),
        _target_fold(
            validation_season=audit.validation_season,
            holdout_season=audit.holdout_season,
            games=audit.holdout_games,
            baseline_mae=audit.holdout_baseline_total_mae,
            adjusted_mae=audit.holdout_prior_total_mae,
            baseline_rmse=audit.holdout_baseline_total_rmse,
            adjusted_rmse=audit.holdout_prior_total_rmse,
            passed=passed,
            detail=detail,
        ),
    )


def _oa_fold(audit: OAAudit) -> tuple[TargetFoldEvidence, TargetFoldEvidence]:
    return (
        _target_fold(
            validation_season=audit.validation_season,
            holdout_season=audit.holdout_season,
            games=0,
            baseline_mae=audit.margin_baseline_mae,
            adjusted_mae=audit.margin_adjusted_mae,
            baseline_rmse=audit.margin_baseline_rmse,
            adjusted_rmse=audit.margin_adjusted_rmse,
            passed=audit.margin_candidate_pass,
            detail=(
                f"feature_set={audit.margin_feature_set}; ridge_alpha={audit.margin_alpha}; "
                f"pbp_ridge={audit.pbp_ridge}"
            ),
        ),
        _target_fold(
            validation_season=audit.validation_season,
            holdout_season=audit.holdout_season,
            games=0,
            baseline_mae=audit.total_baseline_mae,
            adjusted_mae=audit.total_adjusted_mae,
            baseline_rmse=audit.total_baseline_rmse,
            adjusted_rmse=audit.total_adjusted_rmse,
            passed=audit.total_candidate_pass,
            detail=(
                f"feature_set={audit.total_feature_set}; ridge_alpha={audit.total_alpha}; "
                f"pbp_ridge={audit.pbp_ridge}"
            ),
        ),
    )


def _qb_fold(audit: QBAudit) -> tuple[TargetFoldEvidence, TargetFoldEvidence]:
    return (
        _target_fold(
            validation_season=audit.validation_season,
            holdout_season=audit.holdout_season,
            games=0,
            baseline_mae=audit.margin_baseline_mae,
            adjusted_mae=audit.margin_adjusted_mae,
            baseline_rmse=audit.margin_baseline_rmse,
            adjusted_rmse=audit.margin_adjusted_rmse,
            passed=audit.margin_candidate_pass,
            detail=(
                f"feature_set={audit.margin_feature_set}; ridge_alpha={audit.margin_alpha}; "
                f"qb_prior_dropbacks={audit.qb_prior_dropbacks}"
            ),
        ),
        _target_fold(
            validation_season=audit.validation_season,
            holdout_season=audit.holdout_season,
            games=0,
            baseline_mae=audit.total_baseline_mae,
            adjusted_mae=audit.total_adjusted_mae,
            baseline_rmse=audit.total_baseline_rmse,
            adjusted_rmse=audit.total_adjusted_rmse,
            passed=audit.total_candidate_pass,
            detail=(
                f"feature_set={audit.total_feature_set}; ridge_alpha={audit.total_alpha}; "
                f"qb_prior_dropbacks={audit.qb_prior_dropbacks}"
            ),
        ),
    )


def select_target_candidate(
    candidates: tuple[CandidateEvidence, ...],
    *,
    target: str,
) -> dict[str, object]:
    """Select a development SHADOW candidate, with baseline as the default winner."""

    if target not in {"margin", "total"}:
        raise ValueError("target must be margin or total")
    eligible_key = f"{target}_eligible"
    score_key = f"{target}_robustness_score"
    eligible = [value for value in candidates if bool(getattr(value, eligible_key))]
    if not eligible:
        return {
            "selected": "baseline",
            "score_adjustment_enabled": False,
            "robustness_score": 0.0,
            "release_state": "DISABLED",
            "reason": "no candidate cleared every chronological development fold",
        }
    eligible.sort(key=lambda value: (-float(getattr(value, score_key)), value.name))
    selected = eligible[0]
    return {
        "selected": selected.name,
        "score_adjustment_enabled": False,
        "robustness_score": float(getattr(selected, score_key)),
        "release_state": "SHADOW_CANDIDATE",
        "reason": (
            "development-fold winner only; canonical score remains unchanged until "
            "independent 2026 forward evidence clears a separate gate"
        ),
    }


def run_candidate_benchmark(
    *,
    folds: tuple[tuple[int, int], ...] = DEFAULT_FOLDS,
    start_week: int = 5,
    end_week: int = 18,
    client: NFLDataClient | None = None,
    refresh: bool = False,
) -> dict[str, object]:
    """Run a common rolling-origin scoreboard across NFL-native research layers."""

    if not folds:
        raise ValueError("folds must not be empty")
    if any(validation >= holdout for validation, holdout in folds):
        raise ValueError("each validation season must precede its holdout season")
    source = client or NFLDataClient()

    recency_margin: list[TargetFoldEvidence] = []
    recency_total: list[TargetFoldEvidence] = []
    prior_margin: list[TargetFoldEvidence] = []
    prior_total: list[TargetFoldEvidence] = []
    oa_margin: list[TargetFoldEvidence] = []
    oa_total: list[TargetFoldEvidence] = []
    qb_margin: list[TargetFoldEvidence] = []
    qb_total: list[TargetFoldEvidence] = []

    for index, (validation, holdout) in enumerate(folds):
        refresh_fold = refresh and index == 0
        recency = run_recency_audit(
            validation_season=validation,
            holdout_season=holdout,
            start_week=start_week,
            client=source,
            refresh=refresh_fold,
        )
        margin, total = _recency_fold(recency)
        recency_margin.append(margin)
        recency_total.append(total)

        prior = run_prior_audit(
            validation_season=validation,
            holdout_season=holdout,
            start_week=start_week,
            client=source,
            refresh=False,
        )
        margin, total = _prior_fold(prior)
        prior_margin.append(margin)
        prior_total.append(total)

        seasons = tuple(range(validation - 2, holdout + 1))
        oa = run_oa_audit(
            seasons=seasons,
            validation_season=validation,
            holdout_season=holdout,
            start_week=start_week,
            end_week=end_week,
            client=source,
            refresh=False,
        )
        margin, total = _oa_fold(oa)
        # OA audit exposes aggregate row count, not holdout count. Use the common
        # regular-season holdout sample size from the score-model audit for weighting.
        margin = TargetFoldEvidence(**{**margin.to_dict(), "games": recency.holdout_games})
        total = TargetFoldEvidence(**{**total.to_dict(), "games": recency.holdout_games})
        oa_margin.append(margin)
        oa_total.append(total)

        qb = run_qb_audit(
            seasons=seasons,
            validation_season=validation,
            holdout_season=holdout,
            start_week=start_week,
            end_week=end_week,
            client=source,
            refresh=False,
        )
        margin, total = _qb_fold(qb)
        margin = TargetFoldEvidence(**{**margin.to_dict(), "games": recency.holdout_games})
        total = TargetFoldEvidence(**{**total.to_dict(), "games": recency.holdout_games})
        qb_margin.append(margin)
        qb_total.append(total)

    candidates = (
        _candidate("score_recency", recency_margin, recency_total),
        _candidate("prior_season", prior_margin, prior_total),
        _candidate("opponent_adjusted_pbp", oa_margin, oa_total),
        _candidate("quarterback_state", qb_margin, qb_total),
    )
    return {
        "status": "DEVELOPMENT_ONLY",
        "folds": [
            {"validation_season": validation, "holdout_season": holdout}
            for validation, holdout in folds
        ],
        "start_week": start_week,
        "end_week": end_week,
        "candidates": {value.name: value.to_dict() for value in candidates},
        "margin_selection": select_target_candidate(candidates, target="margin"),
        "total_selection": select_target_candidate(candidates, target="total"),
        "canonical_score_adjustment_enabled": False,
        "promotion_eligible": False,
        "reserved_forward_season": 2026,
        "meaning": (
            "2023-2025 are rolling development evidence, not a pristine promotion holdout. "
            "Baseline remains canonical. Any selected nonzero layer is only a candidate for "
            "separately persisted 2026 forward SHADOW evaluation."
        ),
    }
