from nfl.candidate_benchmark import (
    CandidateEvidence,
    TargetFoldEvidence,
    select_target_candidate,
)


def _fold(*, passed: bool, improvement: float, season: int) -> TargetFoldEvidence:
    return TargetFoldEvidence(
        validation_season=season - 1,
        holdout_season=season,
        games=200,
        baseline_mae=10.0,
        adjusted_mae=10.0 - improvement,
        baseline_rmse=13.0,
        adjusted_rmse=13.0 - improvement,
        mae_improvement=improvement,
        rmse_improvement=improvement,
        passed=passed,
        detail="fixture",
    )


def _candidate(
    name: str,
    *,
    margin_passes: tuple[bool, bool],
    margin_improvement: float,
    total_passes: tuple[bool, bool],
    total_improvement: float,
) -> CandidateEvidence:
    margin_folds = tuple(
        _fold(passed=value, improvement=margin_improvement, season=season)
        for value, season in zip(margin_passes, (2024, 2025), strict=True)
    )
    total_folds = tuple(
        _fold(passed=value, improvement=total_improvement, season=season)
        for value, season in zip(total_passes, (2024, 2025), strict=True)
    )
    margin_score = sum(
        ((fold.mae_improvement / fold.baseline_mae) + (fold.rmse_improvement / fold.baseline_rmse))
        / 2.0
        for fold in margin_folds
    ) / len(margin_folds)
    total_score = sum(
        ((fold.mae_improvement / fold.baseline_mae) + (fold.rmse_improvement / fold.baseline_rmse))
        / 2.0
        for fold in total_folds
    ) / len(total_folds)
    return CandidateEvidence(
        name=name,
        margin_folds=margin_folds,
        total_folds=total_folds,
        margin_eligible=all(margin_passes) and margin_score > 0,
        total_eligible=all(total_passes) and total_score > 0,
        margin_robustness_score=margin_score,
        total_robustness_score=total_score,
    )


def test_baseline_is_selected_when_no_candidate_clears_every_fold() -> None:
    candidates = (
        _candidate(
            "a",
            margin_passes=(True, False),
            margin_improvement=0.3,
            total_passes=(False, False),
            total_improvement=0.2,
        ),
        _candidate(
            "b",
            margin_passes=(False, True),
            margin_improvement=0.4,
            total_passes=(True, False),
            total_improvement=0.1,
        ),
    )

    selection = select_target_candidate(candidates, target="margin")

    assert selection["selected"] == "baseline"
    assert selection["score_adjustment_enabled"] is False
    assert selection["release_state"] == "DISABLED"


def test_stronger_all_fold_candidate_is_shadow_only() -> None:
    candidates = (
        _candidate(
            "smaller",
            margin_passes=(True, True),
            margin_improvement=0.1,
            total_passes=(True, True),
            total_improvement=0.1,
        ),
        _candidate(
            "larger",
            margin_passes=(True, True),
            margin_improvement=0.4,
            total_passes=(True, True),
            total_improvement=0.3,
        ),
    )

    margin = select_target_candidate(candidates, target="margin")
    total = select_target_candidate(candidates, target="total")

    assert margin["selected"] == "larger"
    assert total["selected"] == "larger"
    assert margin["release_state"] == "SHADOW_CANDIDATE"
    assert total["release_state"] == "SHADOW_CANDIDATE"
    assert margin["score_adjustment_enabled"] is False
    assert total["score_adjustment_enabled"] is False
