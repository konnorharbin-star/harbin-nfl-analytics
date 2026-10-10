from __future__ import annotations

import json
from datetime import UTC, datetime

import polars as pl

from nfl.monitoring import build_live_monitoring
from nfl.release_gate import build_release_gate


def _current() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_id": [f"game-{index // 3}" for index in range(12)],
            "model_margin_home": [2.0, 2.0, 2.0, -1.0, -1.0, -1.0] * 2,
            "model_total": [44.0, 44.0, 44.0, 47.0, 47.0, 47.0] * 2,
            "calibrated_home_probability": [0.56] * 12,
        }
    )


def _meta(now: datetime) -> dict[str, object]:
    return {
        "generated_at": now.isoformat(),
        "market_coverage": {
            "games": 4,
            "moneyline": 4,
            "spread": 4,
            "total": 4,
        },
        "market_intelligence": {"multi_book_coverage": 0.0},
        "probability": {
            "home_win_brier": 0.20,
            "margin_80_coverage": 0.80,
            "total_80_coverage": 0.80,
        },
        "current_context": {
            "coverage": 1.0,
            "qb_coverage": 1.0,
            "components": {
                "quarterback": 1.0,
                "injuries_personnel": 1.0,
                "rest_travel": 1.0,
                "weather_stadium": 1.0,
            },
            "injury_feed_freshness": {"status": "FRESH"},
        },
    }


def _write_policy(tmp_path, *, mode: str = "production"):
    path = tmp_path / "production_policy.json"
    path.write_text(
        json.dumps(
            {
                "deployment_mode": mode,
                "source": "fixture nested chronological calibration",
                "split": "development/tune/evaluation",
                "markets": {
                    "moneyline": {"enabled": True},
                    "spread": {"enabled": True},
                    "total": {"enabled": False},
                },
                "regime_reliability": {
                    "status": "READY",
                    "operational_ready": True,
                    "fail_closed": True,
                    "market_status": {
                        "moneyline": "RELIABLE",
                        "spread": "RELIABLE",
                        "total": "UNRELIABLE",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_engineering_readiness_excludes_multibook_penalty(tmp_path) -> None:
    now = datetime.now(UTC)
    report = build_live_monitoring(
        _current(),
        _meta(now),
        reports_dir=tmp_path,
        now=now,
    )

    assert report["scores"]["multi_book"] == 0.0
    assert report["live_readiness_score"] < 90.0
    assert report["engineering_readiness_score"] >= 90.0


def test_release_gate_uses_engineering_readiness_not_multibook_score(tmp_path) -> None:
    now = datetime.now(UTC)
    meta = _meta(now)
    monitor = build_live_monitoring(
        _current(),
        meta,
        reports_dir=tmp_path,
        now=now,
    )
    gate = build_release_gate(
        meta,
        monitor,
        {"status": "OK"},
        evidence_path=tmp_path / "missing-evidence.json",
        live_path=tmp_path / "missing-live.json",
    )

    checks = {check["name"]: check for check in gate["checks"]}
    assert checks["live_monitoring"]["passed"] is True
    assert checks["multi_book_consensus"]["passed"] is False
    assert gate["engineering_ready"] is True
    assert gate["release_state"] == "PAPER"
    assert gate["production_eligible"] is False


def test_release_gate_fails_when_injury_personnel_context_is_stale(
    tmp_path,
) -> None:
    now = datetime.now(UTC)
    meta = _meta(now)
    meta["current_context"]["components"]["injuries_personnel"] = 0.75
    meta["current_context"]["injury_feed_freshness"] = {
        "status": "STALE",
        "reason": "no current-week injury reports",
    }

    gate = build_release_gate(
        meta,
        {"engineering_readiness_score": 95},
        {"status": "OK"},
        evidence_path=tmp_path / "missing-evidence.json",
        live_path=tmp_path / "missing-live.json",
    )
    checks = {check["name"]: check for check in gate["checks"]}

    assert checks["injury_personnel_freshness"]["passed"] is False
    assert gate["engineering_ready"] is False
    assert gate["release_state"] == "RESEARCH"


def test_explicit_probability_reliability_failure_forces_research(
    tmp_path,
) -> None:
    now = datetime.now(UTC)
    meta = _meta(now)
    meta["probability"] = {
        "home_win_brier": 0.22,
        "home_win_ece": 0.11,
        "margin_80_coverage": 0.80,
        "total_80_coverage": 0.80,
        "mid_confidence_games": 40,
        "mid_confidence_gap": 0.12,
        "reliability_ready": False,
        "model_family": "conditional_student_t",
    }

    gate = build_release_gate(
        meta,
        {"engineering_readiness_score": 95},
        {"status": "OK"},
        evidence_path=tmp_path / "missing-evidence.json",
        live_path=tmp_path / "missing-live.json",
    )
    checks = {check["name"]: check for check in gate["checks"]}

    assert checks["probability_calibration"]["passed"] is False
    assert gate["engineering_ready"] is False
    assert gate["release_state"] == "RESEARCH"


def test_release_gate_requires_broad_historical_clv_coverage(tmp_path) -> None:
    meta = _meta(datetime.now(UTC))
    meta["market_intelligence"] = {"multi_book_coverage": 1.0}

    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                "status": "ROBUST",
                "overall": {"roi_ci_95": [-0.10, -0.02]},
                "promotion_sample": {
                    "entry_quote_verified": True,
                    "verified_bets": 1200,
                    "verified_timestamped_provider_bets": 1200,
                    "verified_roi_ci_95": [0.01, 0.05],
                    "avg_verified_clv_proxy": 0.02,
                    "verified_clv_samples": 1068,
                    "verified_clv_coverage": 0.89,
                    "positive_markets": 2,
                    "positive_seasons": 2,
                },
            }
        ),
        encoding="utf-8",
    )
    live_path = tmp_path / "live.json"
    live_path.write_text(
        json.dumps(
            {
                "evidence_source": "portfolio_decisions_v1",
                "portfolio_verified": True,
                "graded_bets": 300,
                "roi": 0.01,
                "avg_clv": 0.02,
                "entry_quote_coverage": 1.0,
                "execution_ready_coverage": 1.0,
                "clv_samples": 270,
                "clv_coverage": 0.90,
            }
        ),
        encoding="utf-8",
    )

    gate = build_release_gate(
        meta,
        {"engineering_readiness_score": 95},
        {"status": "OK"},
        evidence_path=evidence_path,
        live_path=live_path,
        policy_path=_write_policy(tmp_path),
    )
    checks = {check["name"]: check for check in gate["checks"]}

    assert checks["historical_clv_coverage"]["passed"] is False
    assert gate["historical_edge_ready"] is False
    assert gate["release_state"] == "PAPER"
    assert gate["production_eligible"] is False


def test_release_gate_cannot_claim_production_with_paper_policy(tmp_path) -> None:
    meta = _meta(datetime.now(UTC))
    meta["market_intelligence"] = {"multi_book_coverage": 1.0}

    evidence_path = tmp_path / "policy-evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                "status": "ROBUST",
                "promotion_sample": {
                    "entry_quote_verified": True,
                    "verified_bets": 1200,
                    "verified_timestamped_provider_bets": 1200,
                    "verified_roi_ci_95": [0.01, 0.05],
                    "avg_verified_clv_proxy": 0.02,
                    "verified_clv_samples": 1080,
                    "verified_clv_coverage": 0.90,
                    "positive_markets": 2,
                    "positive_seasons": 2,
                },
            }
        ),
        encoding="utf-8",
    )
    live_path = tmp_path / "policy-live.json"
    live_path.write_text(
        json.dumps(
            {
                "evidence_source": "portfolio_decisions_v1",
                "portfolio_verified": True,
                "graded_bets": 300,
                "roi": 0.01,
                "avg_clv": 0.02,
                "entry_quote_coverage": 1.0,
                "execution_ready_coverage": 1.0,
                "clv_samples": 270,
                "clv_coverage": 0.90,
            }
        ),
        encoding="utf-8",
    )

    gate = build_release_gate(
        meta,
        {"engineering_readiness_score": 95},
        {"status": "OK"},
        evidence_path=evidence_path,
        live_path=live_path,
        policy_path=_write_policy(tmp_path, mode="paper"),
    )
    checks = {check["name"]: check for check in gate["checks"]}

    assert gate["historical_edge_ready"] is True
    assert gate["live_evidence_ready"] is True
    assert gate["production_policy_ready"] is False
    assert checks["production_policy"]["passed"] is False
    assert gate["release_state"] == "SHADOW"
    assert gate["production_eligible"] is False


def test_release_gate_blocks_unreliable_market_regimes(tmp_path) -> None:
    meta = _meta(datetime.now(UTC))
    meta["market_intelligence"] = {"multi_book_coverage": 1.0}

    evidence_path = tmp_path / "regime-evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                "status": "ROBUST",
                "promotion_sample": {
                    "entry_quote_verified": True,
                    "verified_bets": 1200,
                    "verified_timestamped_provider_bets": 1200,
                    "verified_roi_ci_95": [0.01, 0.05],
                    "avg_verified_clv_proxy": 0.02,
                    "verified_clv_samples": 1080,
                    "verified_clv_coverage": 0.90,
                    "positive_markets": 2,
                    "positive_seasons": 2,
                },
            }
        ),
        encoding="utf-8",
    )
    live_path = tmp_path / "regime-live.json"
    live_path.write_text(
        json.dumps(
            {
                "evidence_source": "portfolio_decisions_v1",
                "portfolio_verified": True,
                "graded_bets": 300,
                "roi": 0.01,
                "avg_clv": 0.02,
                "entry_quote_coverage": 1.0,
                "execution_ready_coverage": 1.0,
                "clv_samples": 270,
                "clv_coverage": 0.90,
            }
        ),
        encoding="utf-8",
    )
    policy_path = _write_policy(tmp_path)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["regime_reliability"]["operational_ready"] = False
    policy["regime_reliability"]["market_status"] = {
        "moneyline": "UNRELIABLE",
        "spread": "UNRELIABLE",
        "total": "UNRELIABLE",
    }
    policy_path.write_text(json.dumps(policy), encoding="utf-8")

    gate = build_release_gate(
        meta,
        {"engineering_readiness_score": 95},
        {"status": "OK"},
        evidence_path=evidence_path,
        live_path=live_path,
        policy_path=policy_path,
    )
    checks = {check["name"]: check for check in gate["checks"]}

    assert checks["regime_edge_reliability"]["passed"] is False
    assert gate["regime_reliability_ready"] is False
    assert gate["production_policy_ready"] is False
    assert gate["production_eligible"] is False


def test_release_gate_requires_broad_forward_clv_coverage(tmp_path) -> None:
    now = datetime.now(UTC)
    meta = _meta(now)
    meta["market_intelligence"] = {"multi_book_coverage": 1.0}

    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                "status": "ROBUST",
                "overall": {"roi_ci_95": [-0.10, -0.02]},
                "promotion_sample": {
                    "entry_quote_verified": True,
                    "verified_bets": 1200,
                    "verified_timestamped_provider_bets": 1200,
                    "excluded_unverified_bets": 0,
                    "verified_roi_ci_95": [0.01, 0.05],
                    "avg_verified_clv_proxy": 0.02,
                    "verified_clv_samples": 1080,
                    "verified_clv_coverage": 0.90,
                    "positive_markets": 2,
                    "positive_seasons": 2,
                },
            }
        ),
        encoding="utf-8",
    )

    live_path = tmp_path / "live.json"
    live_path.write_text(
        json.dumps(
            {
                "evidence_source": "portfolio_decisions_v1",
                "portfolio_verified": True,
                "graded_bets": 300,
                "roi": 0.01,
                "avg_clv": 0.02,
                "entry_quote_coverage": 1.0,
                "execution_ready_coverage": 1.0,
                "clv_samples": 267,
                "clv_coverage": 0.89,
            }
        ),
        encoding="utf-8",
    )

    gate = build_release_gate(
        meta,
        {"engineering_readiness_score": 95},
        {"status": "OK"},
        evidence_path=evidence_path,
        live_path=live_path,
        policy_path=_write_policy(tmp_path),
    )
    checks = {check["name"]: check for check in gate["checks"]}

    assert checks["forward_entry_integrity"]["passed"] is True
    assert checks["forward_clv_coverage"]["passed"] is False
    assert checks["live_shadow_evidence"]["passed"] is False
    assert gate["historical_edge_ready"] is True
    assert gate["live_evidence_ready"] is False
    assert gate["release_state"] == "SHADOW"
    assert gate["production_eligible"] is False

    live_path.write_text(
        json.dumps(
            {
                "evidence_source": "portfolio_decisions_v1",
                "portfolio_verified": True,
                "graded_bets": 300,
                "roi": 0.01,
                "avg_clv": 0.02,
                "entry_quote_coverage": 1.0,
                "execution_ready_coverage": 1.0,
                "clv_samples": 270,
                "clv_coverage": 0.90,
            }
        ),
        encoding="utf-8",
    )

    promoted = build_release_gate(
        meta,
        {"engineering_readiness_score": 95},
        {"status": "OK"},
        evidence_path=evidence_path,
        live_path=live_path,
        policy_path=_write_policy(tmp_path),
    )

    assert promoted["live_evidence_ready"] is True
    assert promoted["release_state"] == "PRODUCTION"
    assert promoted["production_eligible"] is True

    
def test_positive_backtest_without_timestamped_entries_cannot_release(tmp_path) -> None:
    """Strong simulated ROI cannot substitute for identifiable entry timestamps."""
    meta = _meta(datetime.now(UTC))
    meta["market_intelligence"] = {"multi_book_coverage": 1.0}
    evidence_path = tmp_path / "unverified.json"
    evidence_path.write_text(
        json.dumps({
            "status": "ROBUST",
            "promotion_sample": {
                "entry_quote_verified": True,
                "verified_bets": 1200,
                "verified_timestamped_provider_bets": 0,
                "verified_roi_ci_95": [0.01, 0.05],
                "avg_verified_clv_proxy": 0.02,
                "verified_clv_coverage": 0.95,
                "positive_markets": 2,
                "positive_seasons": 2,
            },
        }),
        encoding="utf-8",
    )
    gate = build_release_gate(
        meta,
        {"engineering_readiness_score": 95},
        {"status": "OK"},
        evidence_path=evidence_path,
        live_path=tmp_path / "unverified-live.json",
        policy_path=_write_policy(tmp_path),
    )
    checks = {check["name"]: check for check in gate["checks"]}
    assert checks["historical_entry_integrity"]["passed"] is False
    assert checks["historical_entry_integrity"]["value"]["timestamp_verified_bets"] == 0
    assert gate["historical_edge_ready"] is False
    assert gate["production_eligible"] is False
