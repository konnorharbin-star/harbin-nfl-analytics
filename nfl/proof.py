"""Historical NFL evidence bundle modeled after the CFB proof layer."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl


def _read_json(path: str | Path) -> dict[str, object]:
    source = Path(path)
    if not source.exists():
        return {}
    try:
        value = json.loads(source.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _segment_positive(summary: dict[str, object], minimum_bets: int) -> bool:
    bets = int(summary.get("bets", 0) or 0)
    roi = summary.get("roi", summary.get("roi_per_unit_staked"))
    return bets >= minimum_bets and roi is not None and float(roi) > 0


def build_evidence_report(
    *,
    backtest_path: str | Path = "reports/free_market_backtest.json",
    bets_path: str | Path = "reports/free_market_bets.csv",
) -> dict[str, object]:
    """Summarize historical evidence without upgrading proxy prices to verified entries."""

    backtest = _read_json(backtest_path)
    overall = backtest.get("overall") if isinstance(backtest.get("overall"), dict) else {}
    by_market = backtest.get("by_market") if isinstance(backtest.get("by_market"), dict) else {}
    by_season = backtest.get("by_season") if isinstance(backtest.get("by_season"), dict) else {}

    raw_rows = 0
    verified_bets = 0
    excluded_unverified = 0
    verified_clv_values: list[float] = []
    source = Path(bets_path)
    if source.exists():
        try:
            bets = pl.read_csv(source)
        except Exception:
            bets = pl.DataFrame()
        raw_rows = bets.height
        if not bets.is_empty() and "has_distinct_open" in bets.columns:
            verified = bets.filter(pl.col("has_distinct_open") == True)  # noqa: E712
            verified_bets = verified.height
            excluded_unverified = raw_rows - verified_bets
            if "clv_proxy" in verified.columns:
                verified_clv_values = [
                    float(value) for value in verified.get_column("clv_proxy").drop_nulls().to_list()
                ]

    positive_markets = sum(
        1
        for value in by_market.values()
        if isinstance(value, dict) and _segment_positive(value, 50)
    )
    positive_seasons = sum(
        1
        for value in by_season.values()
        if isinstance(value, dict) and _segment_positive(value, 100)
    )
    ci = overall.get("roi_ci_95")
    if ci is None:
        low = overall.get("roi_ci_95_low")
        high = overall.get("roi_ci_95_high")
        ci = [low, high]
    if not isinstance(ci, list) or len(ci) != 2:
        ci = [None, None]

    avg_verified_clv = (
        sum(verified_clv_values) / len(verified_clv_values) if verified_clv_values else None
    )
    robust = (
        verified_bets >= 1000
        and ci[0] is not None
        and float(ci[0]) > 0
        and avg_verified_clv is not None
        and avg_verified_clv > 0
        and positive_markets >= 2
        and positive_seasons >= 2
    )
    status = "ROBUST" if robust else "ESTABLISHED SAMPLE" if raw_rows >= 500 else "EARLY SAMPLE"
    return {
        "status": status,
        "overall": overall,
        "by_market": by_market,
        "by_season": by_season,
        "promotion_sample": {
            "entry_quote_verified": verified_bets > 0,
            "verified_bets": verified_bets,
            "raw_archive_bets": raw_rows,
            "excluded_unverified_bets": excluded_unverified,
            "avg_verified_clv_proxy": avg_verified_clv,
            "positive_markets": positive_markets,
            "positive_seasons": positive_seasons,
        },
        "source": backtest.get("source", {}),
        "note": (
            "Free nflverse archive-final fallbacks are valid research observations but are not "
            "promotion-quality opening entries. ROBUST status requires explicit distinct opening "
            "entry evidence plus positive uncertainty-adjusted ROI/CLV across markets and seasons."
        ),
    }


def write_evidence_report(
    *,
    output: str | Path = "reports/evidence_report.json",
    backtest_path: str | Path = "reports/free_market_backtest.json",
    bets_path: str | Path = "reports/free_market_bets.csv",
) -> dict[str, object]:
    report = build_evidence_report(backtest_path=backtest_path, bets_path=bets_path)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    return report
