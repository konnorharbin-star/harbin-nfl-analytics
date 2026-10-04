from __future__ import annotations

import json

import polars as pl

from nfl.publication import (
    build_publication_snapshot,
    validate_publication_files,
    write_publication_bundle,
)


def _current() -> pl.DataFrame:
    base = {
        "season": 2026,
        "week": 4,
        "game_id": "g1",
        "date": "2026-10-04",
        "kickoff": "2026-10-04T17:00:00+00:00",
        "away_team": "NYJ",
        "home_team": "PIT",
        "model_margin_home": 4.0,
        "model_total": 44.0,
        "quant_signal": "BET",
        "quant_book": "Book A",
        "quant_odds": -110,
        "quant_probability": 0.57,
        "quant_edge": 0.03,
        "quant_ev": 0.04,
        "market_book_count": 2,
        "portfolio_candidate_units": 0.2,
        "portfolio_stake_units": 0.0,
        "portfolio_action": "PAPER",
        "execution_ready": True,
    }
    return pl.DataFrame(
        [
            {
                **base,
                "quant_market": "moneyline",
                "quant_side": "home",
                "quant_price": None,
            },
            {
                **base,
                "quant_market": "spread",
                "quant_side": "home",
                "quant_price": -2.5,
            },
            {
                **base,
                "quant_market": "total",
                "quant_side": "over",
                "quant_price": 42.5,
            },
        ]
    )


def _report(html_path: str, png_path: str) -> dict[str, object]:
    return {
        "generated_at": "2026-10-01T15:00:00+00:00",
        "league": "NFL",
        "release_state": "RESEARCH",
        "production_eligible": False,
        "meta": {
            "generated_at": "2026-10-01T15:00:00+00:00",
            "season": 2026,
            "week": 4,
            "market_coverage": {"games": 1, "moneyline": 1, "spread": 1, "total": 1},
            "market_intelligence": {"multi_book_coverage": 1.0},
            "probability": {
                "status": "READY",
                "home_win_brier": 0.23,
                "margin_80_coverage": 0.78,
                "total_80_coverage": 0.80,
            },
            "current_context": {"coverage": 0.95, "qb_coverage": 1.0},
            "data_quality": {"status": "OK", "market_rows": 3},
        },
        "portfolio": {
            "mode": "paper",
            "policy_mode": "paper",
            "approved_units": 0.0,
            "paper_or_shadow_allocated_units": 0.6,
            "proposed_units": 0.6,
        },
        "monitoring": {
            "status": "WARN",
            "live_readiness_score": 82.0,
            "scores": {"distribution_stability": 88.0},
            "alerts": [],
        },
        "release_gate": {
            "release_state": "RESEARCH",
            "production_eligible": False,
            "engineering_ready": False,
            "historical_edge_ready": False,
            "live_evidence_ready": False,
            "checks": [],
            "blockers": ["forward evidence incomplete"],
            "next_requirements": ["accumulate forward evidence"],
        },
        "health": {"status": "WARN"},
        "evidence": {"status": "EARLY SAMPLE", "overall": {"bets": 20, "roi": 0.01}},
        "model_card": {
            "release_state": "RESEARCH",
            "production_eligible": False,
            "core_model": {
                "fair_score": "test",
                "sportsbook_prices_in_score_model": False,
                "prior_season_weight": 0.1,
                "quarterback_layer": "test",
            },
            "probability": {"method": "test"},
            "markets": {
                "historical_primary": "test",
                "current_primary": "test",
                "optional_enrichment": "test",
            },
            "risk": {
                "staking": "capped fractional Kelly",
                "portfolio_caps": True,
                "drawdown_throttle": True,
            },
            "non_negotiables": ["No leakage."],
        },
        "publication": {
            "html": html_path,
            "png_pages": [png_path],
            "release_state": "RESEARCH",
        },
        "games": _current().to_dicts(),
    }


def test_publication_snapshot_rejects_nonproduction_approved_stake(tmp_path) -> None:
    html_path = tmp_path / "week.html"
    png_path = tmp_path / "week.png"
    html_path.write_text("ok")
    png_path.write_bytes(b"png")
    report = _report(str(html_path), str(png_path))
    portfolio = dict(report["portfolio"])
    portfolio["approved_units"] = 0.25
    report["portfolio"] = portfolio

    snapshot = build_publication_snapshot(_current(), report)
    assert snapshot["status"] == "FAIL"
    reconciliation = snapshot["reconciliation"]
    assert reconciliation["status"] == "FAIL"
    assert any(
        item["name"] == "approved_stake_fail_closed"
        for item in reconciliation["errors"]
    )


def test_publication_bundle_matches_canonical_outputs(tmp_path) -> None:
    outputs = tmp_path / "outputs"
    docs = tmp_path / "docs"
    reports = tmp_path / "reports"
    history = tmp_path / "history"
    for directory in (outputs, docs, reports, history):
        directory.mkdir()
    html_path = outputs / "nfl_week_4.html"
    png_path = outputs / "nfl_week_4_page1.png"
    html_path.write_text("<html><body>NFL board</body></html>")
    png_path.write_bytes(b"png")
    report = _report(str(html_path), str(png_path))
    (outputs / "current_model.json").write_text(json.dumps(report, default=str))
    _current().write_csv(outputs / "current_predictions.csv")

    bundle = write_publication_bundle(
        _current(),
        report,
        output_dir=outputs,
        docs_dir=docs,
        reports_dir=reports,
        history_dir=history,
    )
    assert bundle["status"] == "PASS"
    assert (docs / "index.html").exists()
    assert (docs / "audit.html").exists()
    assert (docs / "quant.html").exists()
    assert (outputs / "latest.png").exists()
    assert (docs / "latest.png").exists()
    assert (outputs / "latest.png").read_bytes() == (docs / "latest.png").read_bytes()
    assert (outputs / "RUN_REPORT.md").exists()
    assert (outputs / "MODEL_CARD.md").exists()
    assert (outputs / "portfolio_card.csv").exists()
    assert (history / "audit_snapshots_v1.jsonl").exists()
    assert validate_publication_files(output_dir=outputs, docs_dir=docs)["status"] == "PASS"


def test_publication_validation_detects_stale_public_copy(tmp_path) -> None:
    outputs = tmp_path / "outputs"
    docs = tmp_path / "docs"
    reports = tmp_path / "reports"
    history = tmp_path / "history"
    for directory in (outputs, docs, reports, history):
        directory.mkdir()
    html_path = outputs / "nfl_week_4.html"
    png_path = outputs / "nfl_week_4_page1.png"
    html_path.write_text("<html><body>NFL board</body></html>")
    png_path.write_bytes(b"png")
    report = _report(str(html_path), str(png_path))
    (outputs / "current_model.json").write_text(json.dumps(report, default=str))
    _current().write_csv(outputs / "current_predictions.csv")
    write_publication_bundle(
        _current(),
        report,
        output_dir=outputs,
        docs_dir=docs,
        reports_dir=reports,
        history_dir=history,
    )

    stale = json.loads((docs / "latest.json").read_text())
    stale["release_state"] = "PRODUCTION"
    (docs / "latest.json").write_text(json.dumps(stale))
    validation = validate_publication_files(output_dir=outputs, docs_dir=docs)
    assert validation["status"] == "FAIL"
    assert any(item["name"] == "latest_model_copy_exact" for item in validation["errors"])

def test_publication_validation_detects_stale_png_copy(tmp_path) -> None:
    outputs = tmp_path / "outputs"
    docs = tmp_path / "docs"
    reports = tmp_path / "reports"
    history = tmp_path / "history"
    for directory in (outputs, docs, reports, history):
        directory.mkdir()
    html_path = outputs / "nfl_week_4.html"
    png_path = outputs / "nfl_week_4_page1.png"
    html_path.write_text("<html><body>NFL board</body></html>")
    png_path.write_bytes(b"fresh-png")
    report = _report(str(html_path), str(png_path))
    (outputs / "current_model.json").write_text(json.dumps(report, default=str))
    _current().write_csv(outputs / "current_predictions.csv")
    write_publication_bundle(
        _current(),
        report,
        output_dir=outputs,
        docs_dir=docs,
        reports_dir=reports,
        history_dir=history,
    )

    (docs / "latest.png").write_bytes(b"stale-png")
    validation = validate_publication_files(output_dir=outputs, docs_dir=docs)
    assert validation["status"] == "FAIL"
    assert any(item["name"] == "latest_png_copy_exact" for item in validation["errors"])

