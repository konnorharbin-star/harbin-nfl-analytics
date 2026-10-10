"""Historical NFL evidence bundle modeled after the CFB proof layer."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from .entry_integrity import timestamped_promotion_sample
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
    verified_bets_path: str | Path | None = None,
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
            archive_verified = timestamped_promotion_sample(bets)
        excluded_unverified = raw_rows - archive_verified.height

    provider_verified = pl.DataFrame()
    provider_source = (
        None if verified_bets_path is None else Path(verified_bets_path)
    )
    if provider_source is not None and provider_source.exists():
        try:
            provider_bets = pl.read_csv(provider_source)
        except (OSError, pl.exceptions.PolarsError):
            provider_bets = pl.DataFrame()
        provider_verified = timestamped_promotion_sample(provider_bets)

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
    verified_archived_open_close = 0
    verified_timestamped_provider = 0
    if verified_bets and "entry_price_stage" in verified.columns:
        stages = verified.get_column("entry_price_stage").cast(pl.String)
        verified_archived_open_close = int(
            (stages == "espn_archived_open").sum()
        )
        verified_timestamped_provider = int(
            (stages == "timestamped_provider_entry").sum()
        )

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
            "entry_timestamp_verified": verified_bets > 0,
            "verified_bets": verified_bets,
            "verified_archive_bets": archive_verified.height,
            "verified_provider_bets": provider_verified.height,
            "verified_archived_open_close_bets": verified_archived_open_close,
            "verified_timestamped_provider_bets": verified_timestamped_provider,
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
            "verified_provider_bets_path": (
                None if verified_bets_path is None else str(verified_bets_path)
            ),
            "verified_provider_rows": provider_verified.height,
            "historical_entry_provenance": (
                "timestamp-verified pregame provider entry only"
            ),
        },
        "note": (
            "Archive-final and provider-labeled opening/closing stages remain research "
            "diagnostics. Without verified pregame quote timestamps they cannot enter "
            "promotion evidence or policy selection. ROBUST requires timestamped entry "
            "prices and positive uncertainty-adjusted ROI/CLV across independent periods."
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
