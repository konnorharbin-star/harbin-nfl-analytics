"""NFL-only, research-grade evidence triage of downstream market discrepancies.

This is NOT an alternative betting policy. It does not select additional sports
bets, rewrite the independent football projections, lift existing vetoes, change
portfolio exposures, or assert a profitable edge without chronological evidence.
Every number is labeled as raw, held-out-shrunk, or an observed quoted price.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from .execution_market import validate_execution_row
from .market import american_implied_probability, expected_value_per_unit
from .market_shrinkage import shrink_probability
from .policy import market_allowed

SCHEMA_VERSION = 1
MARKETS = frozenset({"moneyline", "spread", "total"})
EVIDENCE_SUPPORTED = "SUPPORTED_RESEARCH"
RAW_WATCH = "RAW_PRICE_DISAGREEMENT"
BLOCKED = "EVIDENCE_OR_CONTEXT_BLOCKED"
NO_QUOTE = "NO_VERIFIED_PRICE"
NO_RAW_EDGE = "NO_POSITIVE_RAW_EDGE"
CATEGORIES = (EVIDENCE_SUPPORTED, RAW_WATCH, BLOCKED, NO_QUOTE, NO_RAW_EDGE)
# Presentation order only. Never use this tier to determine staking.
_ORDER = {
    EVIDENCE_SUPPORTED: 0,
    RAW_WATCH: 1,
    BLOCKED: 2,
    NO_QUOTE: 3,
    NO_RAW_EDGE: 4,
}
BOARD_COLUMNS = (
    "season", "week", "game_id", "kickoff", "away_team", "home_team",
    "quant_market", "quant_side", "quant_book", "quant_price", "quant_odds",
    "quant_quote_at", "quant_probability", "quant_market_probability",
    "quant_edge", "quant_ev", "market_book_count", "market_execution_verified",
    "market_quote_sanity_ok", "market_disagreement_severity",
    "market_dispersion_high", "regime_reliability_status",
    "context_freshness_veto", "qb_certainty_veto",
    "research_signal", "quant_signal", "portfolio_signal",
    "portfolio_action", "portfolio_stake_units",
    "edge_discovery_tier", "edge_discovery_reason",
    "edge_observed_quote_status", "edge_archive_market_status",
    "edge_archive_roi", "edge_archive_roi_ci_high",
    "edge_regime_market_status", "edge_market_shrinkage_status",
    "edge_validated_shrinkage_alpha", "edge_shrunk_probability",
    "edge_shrunk_probability_edge", "edge_shrunk_ev",
    "edge_raw_vs_shrunk_ev_gap", "edge_no_vig_vs_break_even",
    "edge_discovery_score", "edge_stake_authorized",
)


def _number(value: object) -> float | None:
    try:
        out = float(value)
    except (ValueError, TypeError):
        return None
    return out if math.isfinite(out) else None


def _json(path: str | Path) -> dict[str, object] | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        value = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _bool(value: object) -> bool:
    return value is True


def _market_archive(market: str, report: Mapping[str, object] | None) -> dict[str, object]:
    by_market = report.get("by_market", {}) if isinstance(report, Mapping) else {}
    data = by_market.get(market, {}) if isinstance(by_market, Mapping) else {}
    return data if isinstance(data, dict) else {}


def _shrinkage_market(market: str, report: Mapping[str, object] | None) -> dict[str, object]:
    markets = report.get("markets", {}) if isinstance(report, Mapping) else {}
    data = markets.get(market, {}) if isinstance(markets, Mapping) else {}
    return data if isinstance(data, dict) else {}


def _market_registry(market: str, report: Mapping[str, object] | None) -> str:
    # The fixed 2024 development/2025 holdout registry is the actual model
    # evidence gate; a particular candidate's raw EV is not a holdout.
    summary = report.get("summary", {}) if isinstance(report, Mapping) else {}
    statuses = summary.get("market_status", {}) if isinstance(summary, dict) else {}
    return str(statuses.get(market, "MISSING")) if isinstance(statuses, dict) else "MISSING"


def _market_policy(market: str, policy: Mapping[str, object] | None) -> bool:
    if not isinstance(policy, Mapping):
        return False
    allowed, _ = market_allowed(market, policy=dict(policy))
    return allowed


def _rate(
    row: Mapping[str, object], ref: datetime, limits: Mapping[str, object]
) -> tuple[bool, str]:
    # Delegate timestamp/book/age checks to existing canonical execution code.
    return validate_execution_row(row, limits=limits, now=ref)


def classify_research_edge(
    row: Mapping[str, object],
    *,
    regime_report: Mapping[str, object] | None,
    shrinkage_report: Mapping[str, object] | None,
    archive_report: Mapping[str, object] | None,
    policy: Mapping[str, object] | None,
    now: datetime,
) -> dict[str, object]:
    """Rank an observed market *discrepancy*; do not create a bet."""
    market = str(row.get("quant_market") or "").lower()
    raw_ev = _number(row.get("quant_ev"))
    raw_edge = _number(row.get("quant_edge"))
    probability = _number(row.get("quant_probability"))
    no_vig = _number(row.get("quant_market_probability"))
    odds_value = _number(row.get("quant_odds"))
    archive = _market_archive(market, archive_report)
    shrinkage = _shrinkage_market(market, shrinkage_report)
    archive_status = (
        "ARCHIVED_PROVIDER_OPEN_CLOSE_NOT_TIMESTAMP_VERIFIED"
        if isinstance(archive_report, Mapping)
        and archive_report.get("status") == "READY"
        and archive_report.get("timestamped_entry_prices") is False
        else "MISSING_OR_UNVERIFIED_ARCHIVE_PROOF"
    )
    regime_status = _market_registry(market, regime_report)
    shrinkage_status = str(shrinkage.get("status") or "MISSING")
    alpha = _number(shrinkage.get("alpha"))
    validated_alpha = (
        alpha
        if isinstance(shrinkage_report, Mapping)
        and shrinkage_report.get("status") == "RESEARCH_ONLY"
        and shrinkage_status in {"MARKET_ONLY_PREFERRED", "VALIDATED_SHRINKAGE"}
        and alpha is not None and 0 <= alpha <= 1
        else None
    )
    if validated_alpha is None or probability is None or no_vig is None or not (
        0 < probability < 1 and 0 < no_vig < 1
    ):
        shrunk_prob = None
    else:
        shrunk_prob = shrink_probability(probability, no_vig, validated_alpha)
    if odds_value is None or not (odds_value >= 100 or odds_value <= -100):
        shrunk_ev = None
        break_even = None
    else:
        odds = int(odds_value)
        shrunk_ev = (
            expected_value_per_unit(shrunk_prob, odds) if shrunk_prob is not None else None
        )
        break_even = american_implied_probability(odds)
    shrunk_edge = shrunk_prob - no_vig if shrunk_prob is not None and no_vig is not None else None
    ev_gap = raw_ev - shrunk_ev if raw_ev is not None and shrunk_ev is not None else None
    nv_margin = no_vig - break_even if no_vig is not None and break_even is not None else None

    portfolio = policy.get("portfolio") if isinstance(policy, Mapping) else {}
    limits = portfolio if isinstance(portfolio, dict) else {}
    executable, quote_reason = _rate(row, now, limits)
    if not _bool(row.get("market_execution_verified")):
        executable, quote_reason = False, "not a verified sportsbook offer"
    elif not _bool(row.get("market_quote_timestamp_verified")):
        executable, quote_reason = False, "quote timestamp not verified"
    elif not _bool(row.get("market_quote_sanity_ok")):
        executable, quote_reason = False, "cross-book quote failed sanity checks"
    observed = "VERIFIED_FRESH" if executable else "NOT_EXECUTABLE"
    reasons = []
    if regime_report is None or regime_status != "RELIABLE":
        reasons.append("NFL_REGIME_MARKET_NOT_RELIABLE")
    if str(row.get("regime_reliability_status") or "").upper() != "RELIABLE" or not _bool(
        row.get("regime_reliability_ready")
    ):
        reasons.append("CANDIDATE_REGIME_NOT_SUPPORTED")
    if not _bool(row.get("probability_reliability_ready")) or _bool(
        row.get("probability_reliability_veto")
    ):
        reasons.append("PROBABILITY_RELIABILITY_BLOCKED")
    if not _market_policy(market, policy):
        reasons.append("PRODUCTION_MARKET_DISABLED")
    if archive_status != "ARCHIVED_PROVIDER_OPEN_CLOSE_NOT_TIMESTAMP_VERIFIED":
        reasons.append("MISSING_ARCHIVE_EVIDENCE")
    upper = _number(archive.get("roi_ci_95_high"))
    roi = _number(archive.get("roi_per_unit_staked"))
    if upper is not None and upper < 0:
        reasons.append("NEGATIVE_ARCHIVED_MARKET_CI")
    if isinstance(shrinkage_report, Mapping) and (
        shrinkage_report.get("research_conclusion") == "NO_INCREMENTAL_MODEL_VALUE"
    ):
        reasons.append("NO_VERIFIED_INCREMENTAL_MODEL_VALUE")
    if validated_alpha is None:
        reasons.append("NO_FROZEN_VALIDATED_SHRINKAGE")
    elif validated_alpha <= 0:
        reasons.append("HOLDOUT_SELECTS_MARKET_ONLY")
    if shrunk_ev is None or shrunk_ev <= 0:
        reasons.append("NO_POSITIVE_SHRUNK_EV")
    if _bool(row.get("context_freshness_veto")) or not _bool(
        row.get("context_injuries_personnel_fresh")
    ):
        reasons.append("CURRENT_INJURY_CONTEXT_NOT_FRESH")
    if _bool(row.get("qb_certainty_veto")):
        reasons.append("QB_UNCERTAINTY")
    if _bool(row.get("context_veto")):
        reasons.append("OTHER_CONTEXT_VETO")
    if _bool(row.get("market_dispersion_high")) or str(
        row.get("market_disagreement_severity") or ""
    ).upper() == "HIGH":
        reasons.append("HIGH_MARKET_DISAGREEMENT")
    if not executable:
        reasons.append("NO_VERIFIED_FRESH_QUOTE")
    if not all(
        v is not None for v in (raw_ev, raw_edge, probability, no_vig, break_even)
    ) or not (0 < probability < 1 and 0 < no_vig < 1):
        reasons.append("MISSING_OR_INVALID_PROBABILITY_OR_PRICE")
    elif raw_ev <= 0 or raw_edge <= 0:
        reasons.append("NO_POSITIVE_RAW_MODEL_EDGE")
    if raw_edge is None or raw_ev is None or raw_edge <= 0 or raw_ev <= 0:
        tier = NO_RAW_EDGE
    elif not executable:
        tier = NO_QUOTE
    elif not reasons:
        tier = EVIDENCE_SUPPORTED
    elif (
        regime_status != "RELIABLE"
        or str(row.get("regime_reliability_status") or "") != "RELIABLE"
        or not _bool(row.get("regime_reliability_ready"))
        or not _bool(row.get("probability_reliability_ready"))
        or _bool(row.get("probability_reliability_veto"))
        or not _market_policy(market, policy)
        or archive_status != "ARCHIVED_PROVIDER_OPEN_CLOSE_NOT_TIMESTAMP_VERIFIED"
        or (upper is not None and upper < 0)
        or validated_alpha is None
        or validated_alpha <= 0
        or shrunk_ev is None
        or shrunk_ev <= 0
        or _bool(row.get("context_freshness_veto"))
        or not _bool(row.get("context_injuries_personnel_fresh"))
        or _bool(row.get("qb_certainty_veto"))
        or _bool(row.get("context_veto"))
        or _bool(row.get("market_dispersion_high"))
        or str(row.get("market_disagreement_severity") or "").upper() == "HIGH"
    ):
        tier = BLOCKED
    else:
        tier = RAW_WATCH

    # Score is an ordering aid for *raw market disagreement only*, NOT
    # forecast edge, inferred profitability, validated signal or bet threshold.
    score = (
        min(0.20, max(0.0, raw_edge)) * 100.0
        if raw_edge is not None and math.isfinite(raw_edge) else 0.0
    )
    result = {
        "edge_discovery_tier": tier,
        "edge_discovery_reason": ";".join(dict.fromkeys(reasons)) if reasons else (
            "FULL_RESEARCH_EVIDENCE_GATES_PASSED_NOT_PRODUCTION_APPROVED"
        ),
        "edge_observed_quote_status": observed,
        "edge_quote_reason": quote_reason,
        "edge_archive_market_status": archive_status,
        "edge_archive_roi": roi,
        "edge_archive_roi_ci_high": upper,
        "edge_regime_market_status": regime_status,
        "edge_market_shrinkage_status": shrinkage_status,
        "edge_validated_shrinkage_alpha": validated_alpha,
        "edge_shrunk_probability": shrunk_prob,
        "edge_shrunk_probability_edge": shrunk_edge,
        "edge_shrunk_ev": shrunk_ev,
        "edge_raw_vs_shrunk_ev_gap": ev_gap,
        "edge_no_vig_vs_break_even": nv_margin,
        "edge_discovery_score": round(score, 4),
        "edge_stake_authorized": False,
    }
    return result


def enrich_edge_discovery(
    frame: pl.DataFrame,
    *,
    policy: Mapping[str, object] | None = None,
    as_of: datetime | None = None,
    regime_path: str | Path = "reports/regime_edge_reliability.json",
    shrinkage_path: str | Path = "reports/market_edge_shrinkage.json",
    archive_path: str | Path = "reports/verified_market_backtest.json",
) -> tuple[pl.DataFrame, dict[str, object]]:
    """Attach purely observational rankings AFTER existing portfolio decisions."""
    when = as_of or datetime.now(UTC)
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    when = when.astimezone(UTC)
    regimes = _json(regime_path)
    shrinkage = _json(shrinkage_path)
    archive = _json(archive_path)
    sources = {
        "regime_reliability": "LOADED" if regimes else "MISSING_OR_BAD",
        "market_shrinkage": "LOADED" if shrinkage else "MISSING_OR_BAD",
        "archived_market_backtest": "LOADED" if archive else "MISSING_OR_BAD",
    }
    rows: list[dict[str, object]] = []
    if not frame.is_empty():
        for original in frame.to_dicts():
            tagged = dict(original)
            tagged.update(
                classify_research_edge(
                    original,
                    regime_report=regimes,
                    shrinkage_report=shrinkage,
                    archive_report=archive,
                    policy=policy,
                    now=when,
                )
            )
            rows.append(tagged)
        enriched = pl.DataFrame(rows)
    else:
        enriched = frame.clone()
    counts = dict(Counter(item.get("edge_discovery_tier", "UNKNOWN") for item in rows))
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "RESEARCH_ONLY" if all(value == "LOADED" for value in sources.values())
        else "MISSING_EVIDENCE_FAIL_CLOSED",
        "league": "NFL",
        "as_of": when.isoformat(),
        "source_status": sources,
        "rows": len(rows),
        "categories": {category: counts.get(category, 0) for category in CATEGORIES},
        "positive_raw_model_discrepancies": sum(
            _number(row.get("quant_ev")) is not None
            and _number(row.get("quant_ev")) > 0
            and _number(row.get("quant_edge")) is not None
            and _number(row.get("quant_edge")) > 0
            for row in rows
        ),
        "validated_support_rows": counts.get(EVIDENCE_SUPPORTED, 0),
        "release_gate_unmodified": True,
        "production_staking_changed": False,
        "staking_authorized_by_edge_board": False,
        "independent_nfl_holdout_required": True,
        "source_provenance": {
            "regime": "NFL fixed 2024 validation and untouched 2025 holdout",
            "shrinkage": "NFL chronological model-to-market log-odds shrinkage; research only",
            "backtest": (
                "ESPN archive provider-labeled opening/close; "
                "NOT independently timestamped actionable entry"
            ),
        },
        "notes": [
            "Ranked from the single best observed market candidate per game and market, "
            "not exhaustive same-game alternative sides.",
            "Raw EV/probability edge may be grossly optimistic; an archive "
            "market-only alpha=0 removes modeled edge.",
            "No historic archive price is a verified contemporaneous executable entry.",
            "Research prioritization is not a bet recommendation; "
            "all existing policy and veto decisions are preserved.",
            "Even SUPPORTED_RESEARCH would require independent forward evidence "
            "and release gates before production use.",
        ],
    }
    return enriched, report


def write_edge_discovery(
    frame: pl.DataFrame,
    report: Mapping[str, object],
    *,
    output_dir: str | Path = "outputs",
    docs_dir: str | Path = "docs",
) -> None:
    """Publish research board, evidence-watch/blocked subsets, and audit metadata."""
    for directory in (Path(output_dir), Path(docs_dir)):
        directory.mkdir(parents=True, exist_ok=True)
        rows = frame.to_dicts() if not frame.is_empty() else []
        sorted_rows = sorted(
            rows,
            key=lambda row: (
                _ORDER.get(str(row.get("edge_discovery_tier")), 99),
                -float(_number(row.get("edge_discovery_score")) or 0),
                str(row.get("game_id") or ""),
                str(row.get("quant_market") or ""),
            ),
        )
        exports = {
            "edge_priority.csv": sorted_rows,
            "edge_supported.csv": [
                r for r in sorted_rows if r.get("edge_discovery_tier") == EVIDENCE_SUPPORTED
            ],
            "edge_watchlist.csv": [
                r for r in sorted_rows if r.get("edge_discovery_tier") == RAW_WATCH
            ],
            "edge_exclusions.csv": [
                r for r in sorted_rows
                if r.get("edge_discovery_tier") not in {EVIDENCE_SUPPORTED, RAW_WATCH}
            ],
        }
        import csv

        columns = list(BOARD_COLUMNS)
        columns.insert(columns.index("edge_stake_authorized"), "edge_quote_reason")
        for name, data in exports.items():
            with (directory / name).open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
                writer.writeheader()
                for row in data:
                    writer.writerow({k: row.get(k) for k in columns})
        (directory / "edge_discovery_report.json").write_text(
            json.dumps(dict(report), indent=2, sort_keys=True)
        )
