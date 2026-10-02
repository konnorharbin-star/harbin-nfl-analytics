from __future__ import annotations

from nfl.forward_shadow import build_forward_shadow_summary


def test_phase5_fails_closed_with_no_forward_evidence() -> None:
    summary = build_forward_shadow_summary()

    assert summary["phase_status"] == "ACCUMULATING_FORWARD_EVIDENCE"
    assert summary["forward_betting_gate_passed"] is False
    assert summary["production_release_authority"] is False
    assert summary["canonical_model_change_enabled"] is False
    assert summary["candidate_promotion_evidence"] == []


def test_live_forward_gate_requires_sample_roi_and_clv() -> None:
    summary = build_forward_shadow_summary(
        live_report={
            "evidence_source": "portfolio_decisions_v1",
            "portfolio_verified": True,
            "graded_bets": 300,
            "overall": {
                "bets": 300,
                "roi": 0.01,
                "avg_clv": 0.02,
                "max_drawdown": 8.0,
                "roi_ci_95": [-0.03, 0.05],
            },
        }
    )

    assert summary["canonical_betting"]["status"] == "FORWARD_EVIDENCE_READY"
    assert summary["forward_betting_gate_passed"] is True
    assert summary["phase_status"] == "READY_FOR_RELEASE_REVIEW"
    assert summary["production_release_authority"] is False


def test_reported_candidate_promotion_is_rejected_before_minimum_sample() -> None:
    summary = build_forward_shadow_summary(
        recent_form_report={
            "status": "PROMOTION_EVIDENCE",
            "ledger_rows": 40,
            "graded_games": 40,
            "minimum_games": 128,
            "promotion_eligible": True,
        }
    )

    candidate = summary["candidates"]["recent_form_total"]
    assert candidate["status"] == "ACCUMULATING_FORWARD_EVIDENCE"
    assert candidate["promotion_eligible"] is False
    assert summary["candidate_promotion_evidence"] == []


def test_independent_shadow_candidate_can_earn_promotion_evidence() -> None:
    summary = build_forward_shadow_summary(
        qb_total_report={
            "status": "PROMOTION_EVIDENCE",
            "ledger_rows": 140,
            "graded_games": 128,
            "minimum_games": 128,
            "promotion_eligible": True,
            "canonical_score_adjustment_enabled": False,
            "spec_version": "stage15_fixed_quality_v1",
        }
    )

    candidate = summary["candidates"]["qb_total"]
    assert candidate["promotion_eligible"] is True
    assert candidate["canonical_change_enabled"] is False
    assert summary["candidate_promotion_evidence"] == ["qb_total"]
    assert summary["canonical_model_change_enabled"] is False


def test_probability_targets_remain_independent() -> None:
    summary = build_forward_shadow_summary(
        probability_report={
            "status": "READY",
            "ledger_rows": 140,
            "graded_games": 130,
            "non_tied_games": 128,
            "minimum_games": 128,
            "spec_version": "stage23_nested_probability_v1",
            "canonical_probability_change_enabled": False,
            "home_win": {
                "status": "PROMOTION_EVIDENCE",
                "promotion_eligible": True,
                "baseline_brier": 0.23,
                "shadow_brier": 0.22,
            },
            "total_distribution": {
                "status": "SHADOW_FAILING",
                "promotion_eligible": False,
                "baseline_nll": 4.0,
                "shadow_nll": 4.1,
            },
        }
    )

    home = summary["candidates"]["probability_home_win"]
    total = summary["candidates"]["probability_total_distribution"]
    assert home["promotion_eligible"] is True
    assert total["promotion_eligible"] is False
    assert summary["candidate_promotion_evidence"] == ["probability_home_win"]


def test_positive_candidate_evidence_does_not_open_production() -> None:
    summary = build_forward_shadow_summary(
        recent_form_report={
            "status": "PROMOTION_EVIDENCE",
            "ledger_rows": 128,
            "graded_games": 128,
            "minimum_games": 128,
            "promotion_eligible": True,
        },
        live_report={
            "portfolio_verified": True,
            "graded_bets": 10,
            "overall": {"bets": 10, "roi": 0.5, "avg_clv": 0.5},
        },
    )

    assert summary["candidate_promotion_evidence_count"] == 1
    assert summary["forward_betting_gate_passed"] is False
    assert summary["phase_status"] == "ACCUMULATING_FORWARD_EVIDENCE"
    assert summary["production_release_authority"] is False
