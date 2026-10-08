from __future__ import annotations

import json

import polars as pl

from nfl.publication import (
    _write_cards,
    build_publication_snapshot,
    validate_publication_files,
    write_publication_bundle,
)
from nfl.publication_integrity import validate_publication_manifest
from nfl.reporting import write_canonical_report


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
            "cache_safe_png_pages": [png_path],
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
    readme = (outputs / "README.md").read_text()
    assert "Fresh PNGs for mobile" in readme
    assert "cache-safe" in readme
    assert (outputs / "RUN_REPORT.md").exists()
    assert (outputs / "MODEL_CARD.md").exists()
    assert (outputs / "portfolio_card.csv").exists()
    assert (outputs / "suggested_bets.csv").exists()
    assert "BET" in (outputs / "suggested_bets.csv").read_text()
    assert (outputs / "quant_recommendations.csv").exists()
    assert "PAPER" in (outputs / "quant_recommendations.csv").read_text()
    assert (history / "audit_snapshots_v1.jsonl").exists()
    assert validate_publication_files(output_dir=outputs, docs_dir=docs)["status"] == "PASS"


def test_publication_bundle_keeps_model_suggestions_when_portfolio_passes(
    tmp_path,
) -> None:
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

    current = _current().with_columns(
        pl.lit("PASS").alias("portfolio_action"),
        pl.lit(0.0).alias("portfolio_candidate_units"),
        pl.lit(True).alias("execution_ready"),
    )
    report = _report(str(html_path), str(png_path))
    report["games"] = current.to_dicts()
    (outputs / "current_model.json").write_text(json.dumps(report, default=str))
    current.write_csv(outputs / "current_predictions.csv")

    write_publication_bundle(
        current,
        report,
        output_dir=outputs,
        docs_dir=docs,
        reports_dir=reports,
        history_dir=history,
    )

    suggestions = (outputs / "suggested_bets.csv").read_text()
    recommendations = (outputs / "quant_recommendations.csv").read_text()
    assert "BET" in suggestions
    assert recommendations.count("\n") == 1
    quant_html = (outputs / "quant_card.html").read_text()
    assert "Harbin NFL · Model Suggestions" in quant_html
    assert "PASS" in quant_html


def test_publication_bundle_overwrites_stale_recommendation_csv(tmp_path) -> None:
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
    (outputs / "quant_recommendations.csv").write_text("stale-row\n")

    current = _current().with_columns(
        pl.lit("PASS").alias("portfolio_action"),
        pl.lit(0.0).alias("portfolio_candidate_units"),
        pl.lit(False).alias("execution_ready"),
    )
    report = _report(str(html_path), str(png_path))
    report["games"] = current.to_dicts()
    (outputs / "current_model.json").write_text(json.dumps(report, default=str))
    current.write_csv(outputs / "current_predictions.csv")

    write_publication_bundle(
        current,
        report,
        output_dir=outputs,
        docs_dir=docs,
        reports_dir=reports,
        history_dir=history,
    )

    recommendation_csv = (outputs / "quant_recommendations.csv").read_text()
    assert "stale-row" not in recommendation_csv
    assert recommendation_csv.startswith("season,week,game_id")
    assert "No current executable model suggestions." in (
        outputs / "quant_card.html"
    ).read_text()


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



def test_publication_uses_portfolio_research_signal_when_quant_is_pass(
    tmp_path,
) -> None:
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

    current = _current().with_columns(
        pl.lit("PASS").alias("quant_signal"),
        pl.lit("BET").alias("research_signal"),
        pl.lit("BET").alias("portfolio_signal"),
        pl.lit("PASS").alias("portfolio_action"),
        pl.lit(0.0).alias("portfolio_candidate_units"),
        pl.lit(True).alias("execution_ready"),
        pl.lit(True).alias("context_freshness_veto"),
        pl.lit("injury feed pending/stale").alias("context_freshness_veto_reason"),
    )
    report = _report(str(html_path), str(png_path))
    report["games"] = current.to_dicts()
    (outputs / "current_model.json").write_text(json.dumps(report, default=str))
    current.write_csv(outputs / "current_predictions.csv")

    write_publication_bundle(
        current,
        report,
        output_dir=outputs,
        docs_dir=docs,
        reports_dir=reports,
        history_dir=history,
    )

    suggestions = (outputs / "suggested_bets.csv").read_text()
    recommendations = (outputs / "quant_recommendations.csv").read_text()
    quant_html = (outputs / "quant_card.html").read_text()

    assert "BET" in suggestions
    assert "injury feed pending/stale" in suggestions
    assert recommendations.count("\n") == 1
    assert "<td data-role=\"signal\">BET</td>" in quant_html


def test_final_canonical_report_rewrite_rebuilds_manifest_before_publication(
    tmp_path, monkeypatch
) -> None:
    """The second model JSON write adds bundle metadata and must be rehashed."""
    monkeypatch.chdir(tmp_path)
    for name in ("outputs", "docs", "reports", "history"):
        (tmp_path / name).mkdir()
    prior = _report("unused.html", "unused.png")
    current = _current()
    payload = write_canonical_report(
        current,
        meta=prior["meta"],
        portfolio=prior["portfolio"],
        monitoring=prior["monitoring"],
        release_gate=prior["release_gate"],
        health=prior["health"],
        evidence=prior["evidence"],
        model_card=prior["model_card"],
    )
    saved = json.loads((tmp_path / "outputs/current_model.json").read_text())
    manifest = json.loads((tmp_path / "outputs/publication_manifest.json").read_text())
    assert saved == payload
    assert "bundle" in saved["publication"]
    assert saved["publication"]["bundle"]["manifest"] == (
        "outputs/publication_manifest.json"
    )
    assert manifest["run_tag"] == saved["publication"]["run_tag"]
    assert validate_publication_manifest(
        output_dir=tmp_path / "outputs", docs_dir=tmp_path / "docs"
    )["status"] == "PASS"
    assert validate_publication_files(
        output_dir=tmp_path / "outputs", docs_dir=tmp_path / "docs"
    )["status"] == "PASS"


def test_empty_slate_overwrites_old_suggestions_with_header_only_csv(tmp_path) -> None:
    outputs = tmp_path / "outputs"
    docs = tmp_path / "docs"
    outputs.mkdir()
    docs.mkdir()
    for name in ("portfolio_card.csv", "suggested_bets.csv", "quant_recommendations.csv"):
        (outputs / name).write_text("OLD WEEK BET\n")
    _write_cards(_current().head(0), output_dir=outputs, docs_dir=docs)
    for name in ("portfolio_card.csv", "suggested_bets.csv", "quant_recommendations.csv"):
        value = (outputs / name).read_text()
        assert "OLD WEEK BET" not in value
        assert value.startswith("season,week,game_id")
        assert len(value.strip().splitlines()) == 1
    assert "No current executable model suggestions." in (
        outputs / "quant_card.html"
    ).read_text()


def test_schema_free_empty_slate_still_writes_nonempty_safe_headers(tmp_path) -> None:
    outputs = tmp_path / "outputs"
    docs = tmp_path / "docs"
    outputs.mkdir()
    docs.mkdir()
    _write_cards(pl.DataFrame(), output_dir=outputs, docs_dir=docs)
    for name in ("portfolio_card.csv", "suggested_bets.csv", "quant_recommendations.csv"):
        csv = (outputs / name).read_text()
        assert csv.startswith("season,week,game_id")
        assert csv.endswith("\n")
