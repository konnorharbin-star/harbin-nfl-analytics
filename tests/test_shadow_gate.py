import numpy as np

from nfl.shadow_gate import paired_bootstrap_evidence


def test_small_forward_sample_cannot_earn_promotion_status() -> None:
    actual = np.arange(16, dtype=float)
    baseline = actual + 2.0
    adjusted = actual + 1.0

    evidence = paired_bootstrap_evidence(
        actual,
        baseline,
        adjusted,
        iterations=1000,
        minimum_games=128,
        seed=7,
    )

    assert evidence.mae.point_improvement > 0
    assert evidence.rmse.point_improvement > 0
    assert evidence.status == "SHADOW_INSUFFICIENT_SAMPLE"


def test_large_consistent_improvement_can_earn_promotion_evidence() -> None:
    actual = np.linspace(-20.0, 20.0, 200)
    baseline = actual + np.where(np.arange(200) % 2 == 0, 4.0, -4.0)
    adjusted = actual + np.where(np.arange(200) % 2 == 0, 1.0, -1.0)

    evidence = paired_bootstrap_evidence(
        actual,
        baseline,
        adjusted,
        iterations=1000,
        minimum_games=128,
        seed=11,
    )

    assert evidence.mae.lower > 0
    assert evidence.rmse.lower > 0
    assert evidence.status == "PROMOTION_EVIDENCE"


def test_large_negative_point_result_is_shadow_failing() -> None:
    actual = np.linspace(-10.0, 10.0, 160)
    baseline = actual + 1.0
    adjusted = actual + 2.0

    evidence = paired_bootstrap_evidence(
        actual,
        baseline,
        adjusted,
        iterations=1000,
        minimum_games=128,
        seed=17,
    )

    assert evidence.mae.point_improvement < 0
    assert evidence.rmse.point_improvement < 0
    assert evidence.status == "SHADOW_FAILING"
