"""Hard NFL deployment gate mirroring the CFB RESEARCH/PAPER/SHADOW/PRODUCTION states."""

from __future__ import annotations

import json
from pathlib import Path


def _read(path: str | Path) -> dict[str, object]:
    source = Path(path)
    if not source.exists():
        return {}
    try:
        value = json.loads(source.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _check(
    name: str,
    passed: bool,
    value: object,
    requirement: str,
    detail: str = "",
) -> dict[str, object]:
    return {
        "name": name,
        "passed": bool(passed),
        "value": value,
        "requirement": requirement,
        "detail": detail,
    }


def build_release_gate(
    meta: dict[str, object],
    monitor: dict[str, object] | None = None,
    data_quality: dict[str, object] | None = None,
    *,
    evidence_path: str | Path = "reports/evidence_report.json",
    live_path: str | Path = "reports/live_performance.json",
) -> dict[str, object]:
    monitor = monitor or {}
    data_quality = data_quality or {}
    evidence = _read(evidence_path)
    live = _read(live_path)

    market = meta.get("market_coverage")
    if not isinstance(market, dict):
        market = {}
    games = max(1, int(market.get("games", 0) or 0))
    complete_market = min(
        int(market.get("moneyline", 0) or 0),
        int(market.get("spread", 0) or 0),
        int(market.get("total", 0) or 0),
    ) / games

    context = meta.get("current_context")
    if not isinstance(context, dict):
        context = {}
    context_coverage = float(context.get("coverage", 0.0) or 0.0)
    qb_coverage = float(context.get("qb_coverage", context_coverage) or 0.0)

    intel = meta.get("market_intelligence")
    if not isinstance(intel, dict):
        intel = {}
    multi_book = float(intel.get("multi_book_coverage", 0.0) or 0.0)

    probability = meta.get("probability")
    if not isinstance(probability, dict):
        probability = {}
    brier = probability.get("home_win_brier")
    margin_80 = probability.get("margin_80_coverage")
    total_80 = probability.get("total_80_coverage")
    probability_ready = (
        brier is not None
        and float(brier) <= 0.25
        and margin_80 is not None
        and 0.70 <= float(margin_80) <= 0.90
        and total_80 is not None
        and 0.70 <= float(total_80) <= 0.90
    )

    monitor_score = float(
        monitor.get(
            "engineering_readiness_score",
            monitor.get("live_readiness_score", 0.0),
        )
        or 0.0
    )

    overall = evidence.get("overall") if isinstance(evidence.get("overall"), dict) else {}
    promotion = (
        evidence.get("promotion_sample")
        if isinstance(evidence.get("promotion_sample"), dict)
        else {}
    )
    ci = overall.get("roi_ci_95")
    if ci is None:
        ci = [overall.get("roi_ci_95_low"), overall.get("roi_ci_95_high")]
    if not isinstance(ci, list) or len(ci) != 2:
        ci = [None, None]
    historical_ready = (
        str(evidence.get("status") or "").upper() == "ROBUST"
        and bool(promotion.get("entry_quote_verified", False))
        and int(promotion.get("verified_bets", 0) or 0) >= 1000
        and ci[0] is not None
        and float(ci[0]) > 0
        and promotion.get("avg_verified_clv_proxy") is not None
        and float(promotion["avg_verified_clv_proxy"]) > 0
        and int(promotion.get("positive_markets", 0) or 0) >= 2
        and int(promotion.get("positive_seasons", 0) or 0) >= 2
    )

    live_n = int(live.get("graded_bets", live.get("bets", 0)) or 0)
    live_roi = live.get("roi")
    live_clv = live.get("avg_clv")
    portfolio_verified = bool(live.get("portfolio_verified", False))
    live_ready = (
        portfolio_verified
        and live_n >= 300
        and live_roi is not None
        and float(live_roi) >= 0
        and live_clv is not None
        and float(live_clv) > 0
    )

    checks = [
        _check(
            "data_contracts",
            str(data_quality.get("status", "UNKNOWN")).upper() != "FAIL",
            data_quality.get("status", "UNKNOWN"),
            "no ERROR-level data-contract violations",
        ),
        _check(
            "complete_market_coverage",
            complete_market >= 0.95,
            round(complete_market, 4),
            ">=95% current games with ML + spread + total",
        ),
        _check(
            "quarterback_context_coverage",
            qb_coverage >= 0.95,
            round(qb_coverage, 4),
            ">=95% current games with starting-QB state",
        ),
        _check(
            "context_coverage",
            context_coverage >= 0.90,
            round(context_coverage, 4),
            ">=90% current injury/rest/weather/travel context coverage",
        ),
        _check(
            "probability_calibration",
            probability_ready,
            {
                "brier": brier,
                "margin_80_coverage": margin_80,
                "total_80_coverage": total_80,
            },
            "chronological NFL probability holdout meets calibration guardrails",
        ),
        _check(
            "live_monitoring",
            monitor_score >= 90,
            monitor_score,
            "engineering readiness >=90/100; multi-book breadth gated separately",
        ),
        _check(
            "multi_book_consensus",
            multi_book >= 0.75,
            round(multi_book, 4),
            ">=75% current games covered by 2+ verified books",
            (
                "ESPN remains the free primary source; an optional source may "
                "supplement breadth."
            ),
        ),
        _check(
            "historical_entry_integrity",
            bool(promotion.get("entry_quote_verified", False)),
            {
                "verified_bets": promotion.get("verified_bets", 0),
                "excluded_unverified_bets": promotion.get("excluded_unverified_bets", 0),
            },
            (
                "promotion sample uses explicit opening-entry observations, not "
                "archive-final fallbacks"
            ),
        ),
        _check(
            "historical_market_edge",
            historical_ready,
            {
                "status": evidence.get("status"),
                "roi_ci_95": ci,
                "positive_markets": promotion.get("positive_markets", 0),
                "positive_seasons": promotion.get("positive_seasons", 0),
            },
            (
                "ROBUST NFL evidence with positive ROI confidence lower bound and "
                "CLV across markets/seasons"
            ),
        ),
        _check(
            "portfolio_verified_forward_ledger",
            portfolio_verified,
            {"source": live.get("evidence_source"), "graded_bets": live_n},
            "forward evidence comes from cap-constrained persisted portfolio decisions",
        ),
        _check(
            "live_shadow_evidence",
            live_ready,
            {"graded_bets": live_n, "roi": live_roi, "avg_clv": live_clv},
            ">=300 graded portfolio-verified live/shadow bets, non-negative ROI, positive CLV",
        ),
    ]

    engineering_names = {
        "data_contracts",
        "complete_market_coverage",
        "quarterback_context_coverage",
        "context_coverage",
        "probability_calibration",
        "live_monitoring",
    }
    engineering_ready = all(
        bool(check["passed"]) for check in checks if check["name"] in engineering_names
    )
    if not engineering_ready:
        state = "RESEARCH"
    elif historical_ready and multi_book >= 0.75 and live_ready:
        state = "PRODUCTION"
    elif historical_ready:
        state = "SHADOW"
    else:
        state = "PAPER"

    blockers = [
        f"{check['name']}: {check['requirement']}" for check in checks if not check["passed"]
    ]
    next_steps: list[str] = []
    if context_coverage < 0.90:
        next_steps.append(
            "Complete timestamp-safe NFL injuries/rest/weather/travel context coverage."
        )
    if multi_book < 0.75:
        next_steps.append(
            "Accumulate or supplement verified multi-book current pricing coverage."
        )
    if not bool(promotion.get("entry_quote_verified", False)):
        next_steps.append(
            "Accumulate explicit forward/opening entry snapshots; archive-final "
            "fallbacks cannot promote the model."
        )
    if not historical_ready:
        next_steps.append(
            "Keep NFL policy in paper research until verified-entry chronological "
            "ROI/CLV evidence clears the hard gates."
        )
    if not live_ready:
        next_steps.append(
            "Accumulate independently graded portfolio decisions; historical "
            "backtests do not substitute for live evidence."
        )

    return {
        "release_state": state,
        "production_eligible": state == "PRODUCTION",
        "engineering_ready": engineering_ready,
        "historical_edge_ready": historical_ready,
        "live_evidence_ready": live_ready,
        "checks": checks,
        "blockers": blockers,
        "next_requirements": next_steps,
        "meaning": (
            "Hard NFL deployment gate. Structural parity with CFB does not transfer CFB evidence; "
            "PRODUCTION requires independent NFL engineering, historical entry, market breadth, "
            "and portfolio-verified live evidence."
        ),
    }


def write_release_gate(
    meta: dict[str, object],
    monitor: dict[str, object] | None = None,
    data_quality: dict[str, object] | None = None,
    *,
    output_path: str | Path = "outputs/release_gate.json",
    report_path: str | Path = "reports/release_gate.json",
) -> dict[str, object]:
    gate = build_release_gate(meta, monitor, data_quality)
    for path in (output_path, report_path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(gate, indent=2, sort_keys=True, default=str))
    return gate
