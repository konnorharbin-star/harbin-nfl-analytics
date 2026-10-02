from __future__ import annotations

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
        },
    }


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
