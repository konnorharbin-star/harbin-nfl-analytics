"""NFL edge discovery tests: holdout, market-only, quote provenance and stake invariance."""
from __future__ import annotations

import csv
import json
from datetime import UTC, datetime, timedelta

import polars as pl

from nfl.edge_discovery import (
    BLOCKED,
    EVIDENCE_SUPPORTED,
    NO_QUOTE,
    NO_RAW_EDGE,
    RAW_WATCH,
    classify_research_edge,
    enrich_edge_discovery,
    write_edge_discovery,
)
from nfl.policy import DEFAULT_POLICY


NOW = datetime(2026, 10, 7, 18, tzinfo=UTC)


def candidate(**changes: object) -> dict[str, object]:
    row: dict[str, object] = {
        "season": 2026,
        "week": 5,
        "game_id": "2026_05_NYG_WAS",
        "home_team": "WAS",
        "away_team": "NYG",
        "kickoff": (NOW + timedelta(days=4)).isoformat(),
        "quant_market": "spread",
        "quant_side": "home",
        "quant_book": "DraftKings",
        "quant_price": -3.5,
        "quant_odds": -110,
        "quant_quote_at": (NOW - timedelta(minutes=2)).isoformat(),
        "quant_probability": 0.62,
        "quant_market_probability": 0.50,
        "quant_edge": 0.12,
        "quant_ev": 0.62 * (1 + 100 / 110) - 1,
        "market_book_count": 4,
        "market_execution_verified": True,
        "market_quote_timestamp_verified": True,
        "market_quote_sanity_ok": True,
        "market_disagreement_severity": "LOW",
        "market_dispersion_high": False,
        "probability_reliability_ready": True,
        "probability_reliability_veto": False,
        "regime_reliability_ready": True,
        "regime_reliability_status": "RELIABLE",
        "qb_certainty_veto": False,
        "context_freshness_veto": False,
        "context_veto": False,
        "context_injuries_personnel_fresh": True,
        "portfolio_stake_units": 0.0,
        "portfolio_candidate_units": 0.0,
        "portfolio_signal": "PASS",
        "portfolio_action": "PASS",
        "research_signal": "STRONG",
        "quant_signal": "PASS",
    }
    row.update(changes)
    return row


def evidence(alpha=0.75, market_status="RELIABLE", incremental=True):
    regimes = {"summary": {"market_status": {
        "moneyline": market_status,
        "spread": market_status,
        "total": market_status,
    }}}
    shrink = {
        "status": "RESEARCH_ONLY",
        "research_conclusion": (
            "INCREMENTAL_MODEL_VALUE" if incremental else "NO_INCREMENTAL_MODEL_VALUE"
        ),
        "markets": {"spread": {
            "alpha": alpha,
            "status": "VALIDATED_SHRINKAGE" if alpha > 0 else "MARKET_ONLY_PREFERRED",
        }},
    }
    archive = {
        "status": "READY",
        "timestamped_entry_prices": False,
        "by_market": {"spread": {
            "roi_per_unit_staked": -0.05,
            "roi_ci_95_high": 0.02,
            "bets": 420,
        }},
    }
    return regimes, shrink, archive


def tag(row, proofs=None, policy=DEFAULT_POLICY):
    regime, shrink, archive = proofs or evidence()
    return classify_research_edge(
        row, regime_report=regime, shrinkage_report=shrink,
        archive_report=archive, policy=policy, now=NOW,
    )


def test_strictly_favorable_heldout_shrunk_edge_is_still_research_only():
    ranked = tag(candidate())
    assert ranked["edge_discovery_tier"] == EVIDENCE_SUPPORTED
    assert ranked["edge_observed_quote_status"] == "VERIFIED_FRESH"
    assert ranked["edge_shrunk_probability"] > 0.50
    assert ranked["edge_shrunk_ev"] > 0
    assert ranked["edge_stake_authorized"] is False
    assert ranked["edge_archive_market_status"].startswith("ARCHIVED_PROVIDER")


def test_nfl_market_only_selected_by_holdout_overrides_raw_ev():
    proofs = evidence(alpha=0.0, market_status="UNRELIABLE", incremental=False)
    row = tag(candidate(regime_reliability_ready=False,
                        regime_reliability_status="BLOCKED"), proofs)
    assert candidate()["quant_ev"] > 0.10
    assert row["edge_discovery_tier"] == BLOCKED
    assert row["edge_shrunk_probability"] == 0.50
    assert row["edge_shrunk_ev"] < 0
    assert "HOLDOUT_SELECTS_MARKET_ONLY" in row["edge_discovery_reason"]
    assert "NFL_REGIME_MARKET_NOT_RELIABLE" in row["edge_discovery_reason"]


def test_no_vig_probability_and_book_break_even_are_distinct():
    row = tag(candidate(), evidence(alpha=0.0))
    assert row["edge_no_vig_vs_break_even"] < 0
    assert abs(row["edge_shrunk_probability_edge"]) < 1e-12
    assert row["edge_raw_vs_shrunk_ev_gap"] > 0


def test_quote_age_bad_timestamp_and_research_fallback_fail_closed():
    stale = tag(candidate(quant_quote_at=(NOW - timedelta(hours=2)).isoformat()))
    future = tag(candidate(quant_quote_at=(NOW + timedelta(hours=1)).isoformat()))
    research = tag(candidate(market_execution_verified=False))
    bad_sanity = tag(candidate(market_quote_sanity_ok=False))
    assert all(r["edge_discovery_tier"] == NO_QUOTE for r in (
        stale, future, research, bad_sanity
    ))
    assert all(r["edge_stake_authorized"] is False for r in (
        stale, future, research, bad_sanity
    ))
    assert "not a verified sportsbook" in research["edge_quote_reason"]


def test_existing_nfl_vetoes_cannot_become_supported_research():
    for change in (
        {"context_freshness_veto": True},
        {"context_injuries_personnel_fresh": False},
        {"qb_certainty_veto": True},
        {"market_dispersion_high": True},
        {"market_disagreement_severity": "HIGH"},
        {"probability_reliability_veto": True},
        {"regime_reliability_ready": False},
    ):
        ranked = tag(candidate(**change))
        assert ranked["edge_discovery_tier"] == BLOCKED
        assert ranked["edge_discovery_reason"]
        assert ranked["edge_stake_authorized"] is False


def test_historically_negative_archive_market_confidence_blocks():
    regime, shrink, archive = evidence()
    archive["by_market"]["spread"]["roi_ci_95_high"] = -0.01
    ranked = tag(candidate(), (regime, shrink, archive))
    assert ranked["edge_discovery_tier"] == BLOCKED
    assert "NEGATIVE_ARCHIVED_MARKET_CI" in ranked["edge_discovery_reason"]


def test_raw_negative_ev_cannot_be_ranked_as_discovered_value():
    ranked = tag(candidate(quant_ev=-0.02, quant_edge=-0.03))
    assert ranked["edge_discovery_tier"] == NO_RAW_EDGE


def test_disabled_nfl_market_never_creates_validated_edge():
    policy = json.loads(json.dumps(DEFAULT_POLICY))
    policy["markets"]["spread"]["enabled"] = False
    ranked = tag(candidate(), policy=policy)
    assert ranked["edge_discovery_tier"] == BLOCKED
    assert "PRODUCTION_MARKET_DISABLED" in ranked["edge_discovery_reason"]


def test_partial_evidence_fails_closed_without_making_up_shrinkage(tmp_path):
    a = tmp_path / "missing_archive.json"
    regime, shrink, _ = evidence()
    (tmp_path / "regimes.json").write_text(json.dumps(regime))
    (tmp_path / "shrink.json").write_text(json.dumps(shrink))
    frame, summary = enrich_edge_discovery(
        pl.DataFrame([candidate()]), policy=DEFAULT_POLICY, as_of=NOW,
        regime_path=tmp_path / "regimes.json",
        shrinkage_path=tmp_path / "shrink.json", archive_path=a,
    )
    assert summary["status"] == "MISSING_EVIDENCE_FAIL_CLOSED"
    assert frame["edge_discovery_tier"][0] != EVIDENCE_SUPPORTED
    assert summary["validated_support_rows"] == 0


def test_enrichment_retains_the_existing_signals_and_portfolio_stakes(tmp_path):
    regime, shrink, archive = evidence(alpha=0.0, market_status="UNRELIABLE",
                                       incremental=False)
    for name, data in (
        ("r.json", regime), ("s.json", shrink), ("a.json", archive)
    ):
        (tmp_path / name).write_text(json.dumps(data))
    sample = candidate(
        portfolio_stake_units=0.0, portfolio_candidate_units=0.2,
        portfolio_action="PAPER", quant_signal="PASS", research_signal="BET",
    )
    before = pl.DataFrame([sample])
    after, summary = enrich_edge_discovery(
        before, policy=DEFAULT_POLICY, as_of=NOW,
        regime_path=tmp_path / "r.json",
        shrinkage_path=tmp_path / "s.json",
        archive_path=tmp_path / "a.json",
    )
    for column in before.columns:
        assert after[column].to_list() == before[column].to_list()
    assert summary["rows"] == 1
    assert summary["categories"][BLOCKED] == 1
    assert not summary["production_staking_changed"]


def test_exports_are_deterministic_and_empty_supported_file_has_header(tmp_path):
    proofs = evidence(alpha=0.0, market_status="UNRELIABLE", incremental=False)
    for name, data in zip(("r.json", "s.json", "a.json"), proofs):
        (tmp_path / name).write_text(json.dumps(data))
    ranked, report = enrich_edge_discovery(
        pl.DataFrame([candidate(), candidate(game_id="g2", quant_edge=-0.1, quant_ev=-0.1)]),
        policy=DEFAULT_POLICY, as_of=NOW,
        regime_path=tmp_path / "r.json",
        shrinkage_path=tmp_path / "s.json",
        archive_path=tmp_path / "a.json",
    )
    write_edge_discovery(
        ranked, report, output_dir=tmp_path / "outputs",
        docs_dir=tmp_path / "docs"
    )
    for file in ("edge_priority.csv", "edge_supported.csv", "edge_watchlist.csv",
                 "edge_exclusions.csv", "edge_discovery_report.json"):
        assert (tmp_path / "outputs" / file).read_bytes() == (
            tmp_path / "docs" / file
        ).read_bytes()
    with (tmp_path / "outputs" / "edge_supported.csv").open() as handle:
        assert list(csv.DictReader(handle)) == []
    with (tmp_path / "outputs" / "edge_priority.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["edge_discovery_tier"] == BLOCKED
    assert rows[1]["edge_discovery_tier"] == NO_RAW_EDGE
    assert rows[0]["edge_shrunk_ev"]
    assert json.loads((tmp_path / "docs" / "edge_discovery_report.json").read_text())[
        "validated_support_rows"
    ] == 0


def test_empty_frame_writes_all_five_files(tmp_path):
    tagged, report = enrich_edge_discovery(
        pl.DataFrame(), policy=DEFAULT_POLICY, as_of=NOW,
        regime_path=tmp_path / "bad", shrinkage_path=tmp_path / "bad2",
        archive_path=tmp_path / "bad3",
    )
    write_edge_discovery(tagged, report, output_dir=tmp_path / "out", docs_dir=tmp_path / "docs")
    assert tagged.is_empty()
    assert report["categories"][EVIDENCE_SUPPORTED] == 0
    assert (tmp_path / "out" / "edge_priority.csv").read_text().startswith("season,")
