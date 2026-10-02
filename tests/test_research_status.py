from nfl.research_status import build_research_status


def test_research_status_reconciles_historical_and_forward_evidence() -> None:
    candidate = {
        "status": "DEVELOPMENT_ONLY",
        "margin_selection": {
            "selected": "quarterback_state",
            "release_state": "SHADOW_CANDIDATE",
            "robustness_score": 0.006,
        },
        "total_selection": {
            "selected": "baseline",
            "release_state": "DISABLED",
            "robustness_score": 0.0,
        },
        "canonical_score_adjustment_enabled": False,
    }
    market = {
        "status": "RESEARCH_ONLY",
        "incremental_model_value_markets": 0,
        "canonical_market_probability_change_enabled": False,
        "betting_policy_change_enabled": False,
        "markets": {
            "moneyline": {
                "alpha": 0.0,
                "incremental_model_value_vs_market": False,
            },
            "spread": {
                "alpha": 0.0,
                "incremental_model_value_vs_market": False,
            },
            "total": {
                "alpha": 0.0,
                "incremental_model_value_vs_market": False,
            },
        },
    }
    forward = {
        "phase_status": "ACCUMULATING_FORWARD_EVIDENCE",
        "canonical_model_change_enabled": False,
        "production_release_authority": False,
        "canonical_betting": {
            "status": "ACCUMULATING_FORWARD_EVIDENCE",
            "graded_bets": 0,
            "minimum_bets": 300,
            "forward_betting_gate_passed": False,
        },
        "candidates": {
            "qb_total": {
                "status": "ACCUMULATING_FORWARD_EVIDENCE",
                "graded_games": 0,
                "minimum_games": 128,
                "promotion_eligible": False,
            }
        },
        "candidate_promotion_evidence_count": 0,
    }

    report = build_research_status(
        candidate_benchmark=candidate,
        market_shrinkage=market,
        forward_shadow=forward,
    )

    assert report["canonical_model_change_enabled"] is False
    assert report["canonical_market_change_enabled"] is False
    assert report["historical_model_change_ready"] is False
    assert report["market_edge_change_ready"] is False
    assert report["forward_model_change_ready"] is False
    assert report["active_forward_candidates"] == ["qb_total"]
    assert report["market_edge_research"]["selected_alpha"] == {
        "moneyline": 0.0,
        "spread": 0.0,
        "total": 0.0,
    }
    assert (
        report["historical_fair_score_research"]["fixed_qb_margin_enabled"]
        is False
    )
    assert (
        report["historical_fair_score_research"]["fixed_qb_total_shadow_enabled"]
        is True
    )


def test_missing_research_inputs_fail_closed() -> None:
    report = build_research_status()

    assert report["historical_model_change_ready"] is False
    assert report["market_edge_change_ready"] is False
    assert report["forward_model_change_ready"] is False
    assert report["active_forward_candidates"] == []
    assert report["promotion_ready_forward_candidates"] == []
    assert report["canonical_model_change_enabled"] is False
