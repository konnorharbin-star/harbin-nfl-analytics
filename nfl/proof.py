"""Historical NFL evidence bundle modeled after the CFB proof layer."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from .free_market_backtest import summarize_archive_bets

HISTORICAL_CLV_COVERAGE_MINIMUM = 0.90


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


def _bool_column(frame: pl.DataFrame, name: str) -> pl.Series:
    return frame.get_column(name).cast(pl.Boolean, strict=False).fill_null(False)


def _verified_segments(
    frame: pl.DataFrame,
    column: str,
) -> dict[str, dict[str, object]]:
    if frame.is_empty() or column not in frame.columns:
        return {}
    values = sorted(str(value) for value in frame.get_column(column).unique().to_list())
    output: dict[str, dict[str, object]] = {}
    for value in values:
        segment = frame.filter(pl.col(column).cast(pl.String) == value)
        output[value] = summarize_archive_bets(segment).to_dict()
    return output


def build_evidence_report(
    *,
    backtest_path: str | Path = "reports/free_market_backtest.json",
    bets_path: str | Path = "reports/free_market_bets.csv",
    verified_bets_path: str | Path = "reports/verified_market_bets.csv",
) -> dict[str, object]:
    """Summarize historical evidence without upgrading proxy prices to verified entries."""

    backtest = _read_json(backtest_path)
    overall = backtest.get("overall") if isinstance(backtest.get("overall"), dict) else {}
    by_market = (
        backtest.get("by_market") if isinstance(backtest.get("by_market"), dict) else {}
    )
    by_season = (
        backtest.get("by_season") if isinstance(backtest.get("by_season"), dict) else {}
    )

    raw_rows = 0
    opening_line_observed_bets = 0
    excluded_unverified = 0
    archive_verified = pl.DataFrame()
    source = Path(bets_path)
    if source.exists():
        try:
            bets = pl.read_csv(source)
        except (OSError, pl.exceptions.PolarsError):
            bets = pl.DataFrame()
        raw_rows = bets.height
        if not bets.is_empty():
            if "entry_line_observed" in bets.columns:
                opening_line_observed_bets = int(
                    _bool_column(bets, "entry_line_observed").sum()
                )
            elif "has_distinct_open" in bets.columns:
                opening_line_observed_bets = int(
                    _bool_column(bets, "has_distinct_open").sum()
                )
            if "entry_price_verified" in bets.columns:
                archive_verified = bets.filter(
                    _bool_column(bets, "entry_price_verified")
                )
        excluded_unverified = raw_rows - archive_verified.height

    provider_verified = pl.DataFrame()
    provider_source = Path(verified_bets_path)
    if provider_source.exists():
        try:
            provider_bets = pl.read_csv(provider_source)
        except (OSError, pl.exceptions.PolarsError):
            provider_bets = pl.DataFrame()
        if (
            not provider_bets.is_empty()
            and "entry_price_verified" in provider_bets.columns
            and "entry_quote_verified" in provider_bets.columns
        ):
            provider_verified = provider_bets.filter(
                _bool_column(provider_bets, "entry_price_verified")
                & _bool_column(provider_bets, "entry_quote_verified")
            )

    verified_frames = [
        frame
        for frame in (archive_verified, provider_verified)
        if not frame.is_empty()
    ]
    if verified_frames:
        verified = pl.concat(verified_frames, how="diagonal_relaxed")
        dedupe = [
            name
            for name in ("season", "week", "game_id", "market_type")
            if name in verified.columns
        ]
        if len(dedupe) == 4:
            verified = verified.unique(subset=dedupe, keep="last")
    else:
        verified = pl.DataFrame()
    verified_bets = verified.height

    verified_summary = (
        summarize_archive_bets(verified).to_dict() if verified_bets else {}
    )
    verified_by_market = _verified_segments(verified, "market_type")
    verified_by_season = _verified_segments(verified, "season")
    positive_markets = sum(
        1
        for value in verified_by_market.values()
        if _segment_positive(value, 50)
    )
    positive_seasons = sum(
        1
        for value in verified_by_season.values()
        if _segment_positive(value, 100)
    )

    verified_ci = [
        verified_summary.get("roi_ci_95_low"),
        verified_summary.get("roi_ci_95_high"),
    ]
    avg_verified_clv = verified_summary.get("average_clv_proxy")
    verified_clv_samples = int(verified_summary.get("clv_samples", 0) or 0)
    verified_clv_coverage = (
        verified_clv_samples / verified_bets if verified_bets else 0.0
    )
    robust = (
        verified_bets >= 1000
        and verified_ci[0] is not None
        and float(verified_ci[0]) > 0
        and avg_verified_clv is not None
        and float(avg_verified_clv) > 0
        and verified_clv_coverage >= HISTORICAL_CLV_COVERAGE_MINIMUM
        and positive_markets >= 2
        and positive_seasons >= 2
    )
    if robust:
        status = "ROBUST"
    elif raw_rows >= 500:
        status = "ESTABLISHED SAMPLE"
    else:
        status = "EARLY SAMPLE"

    return {
        "status": status,
        "overall": overall,
        "by_market": by_market,
        "by_season": by_season,
        "promotion_sample": {
            "entry_quote_verified": verified_bets > 0,
            "entry_price_verified": verified_bets > 0,
            "verified_bets": verified_bets,
            "verified_archive_bets": archive_verified.height,
            "verified_provider_bets": provider_verified.height,
            "raw_archive_bets": raw_rows,
            "opening_line_observed_bets": opening_line_observed_bets,
            "excluded_unverified_bets": excluded_unverified,
            "verified_roi_per_unit_staked": verified_summary.get(
                "roi_per_unit_staked"
            ),
            "verified_roi_ci_95": verified_ci,
            "avg_verified_clv_proxy": avg_verified_clv,
            "verified_clv_samples": verified_clv_samples,
            "verified_clv_coverage": verified_clv_coverage,
            "minimum_verified_clv_coverage": HISTORICAL_CLV_COVERAGE_MINIMUM,
            "positive_markets": positive_markets,
            "positive_seasons": positive_seasons,
            "by_market": verified_by_market,
            "by_season": verified_by_season,
        },
        "source": {
            **(
                backtest.get("source", {})
                if isinstance(backtest.get("source"), dict)
                else {}
            ),
            "verified_provider_bets_path": str(verified_bets_path),
            "verified_provider_rows": provider_verified.height,
        },
        "note": (
            "Free nflverse archive-final fallbacks and opening-line observations without "
            "opening juice remain valid research observations, but they are not promotion-quality "
            "entry-price evidence. ROBUST status requires observed entry prices plus positive "
            "uncertainty-adjusted ROI/CLV across multiple markets and seasons with "
            "broad closing-snapshot coverage."
        ),
    }


def write_evidence_report(
    *,
    output: str | Path = "reports/evidence_report.json",
    backtest_path: str | Path = "reports/free_market_backtest.json",
    bets_path: str | Path = "reports/free_market_bets.csv",
    verified_bets_path: str | Path = "reports/verified_market_bets.csv",
) -> dict[str, object]:
    report = build_evidence_report(
        backtest_path=backtest_path,
        bets_path=bets_path,
        verified_bets_path=verified_bets_path,
    )
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    return report
