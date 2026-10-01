"""Audit free NFL historical betting evidence and entry-price provenance."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from .contracts import DataContractError, require_columns

REQUIRED = {
    "season",
    "week",
    "game_id",
    "market_type",
    "side",
    "american_odds",
    "price_stage",
    "has_distinct_open",
    "result",
    "net_units",
}


def audit_backtest_bets(frame: pl.DataFrame) -> dict[str, object]:
    """Separate broad archive research from promotion-quality opening-entry evidence."""

    if frame.is_empty():
        return {
            "status": "WARN",
            "errors": 0,
            "warnings": 1,
            "bets": 0,
            "issues": [{"severity": "WARNING", "code": "empty_backtest"}],
            "quote_integrity": {
                "verified_opening_entry_bets": 0,
                "unverified_or_final_fallback_bets": 0,
                "verified_opening_entry_rate": 0.0,
            },
        }
    require_columns(frame, REQUIRED, "free_market_backtest_bets")

    issues: list[dict[str, object]] = []
    duplicate_groups = (
        frame.group_by(["game_id", "market_type"]).len().filter(pl.col("len") > 1).height
    )
    if duplicate_groups:
        issues.append(
            {
                "severity": "ERROR",
                "code": "duplicate_game_market_bets",
                "groups": duplicate_groups,
            }
        )

    odds = frame.get_column("american_odds").cast(pl.Float64, strict=False)
    invalid_odds = frame.filter(
        odds.is_null() | (odds == 0) | ((odds > -100) & (odds < 100))
    ).height
    if invalid_odds:
        issues.append(
            {
                "severity": "ERROR",
                "code": "invalid_entry_odds",
                "rows": invalid_odds,
            }
        )

    allowed_stages = {"archive_open_line_final_price", "archive_final_fallback"}
    invalid_stage = frame.filter(~pl.col("price_stage").is_in(sorted(allowed_stages))).height
    if invalid_stage:
        issues.append(
            {
                "severity": "ERROR",
                "code": "invalid_price_stage",
                "rows": invalid_stage,
            }
        )

    distinct = frame.get_column("has_distinct_open").cast(pl.Boolean, strict=False).fill_null(False)
    verified = int(distinct.sum())
    fallback = frame.height - verified
    if fallback:
        issues.append(
            {
                "severity": "WARNING",
                "code": "archive_final_fallback_present",
                "rows": fallback,
                "meaning": "valid research rows; excluded from promotion-quality entry evidence",
            }
        )

    if "clv_proxy" in frame.columns:
        manufactured = frame.filter(
            (~pl.col("has_distinct_open").cast(pl.Boolean, strict=False).fill_null(False))
            & pl.col("clv_proxy").is_not_null()
        ).height
        if manufactured:
            issues.append(
                {
                    "severity": "ERROR",
                    "code": "clv_proxy_on_unverified_entry",
                    "rows": manufactured,
                }
            )

    if "opening_book" in frame.columns:
        verified_missing_book = frame.filter(
            pl.col("has_distinct_open").cast(pl.Boolean, strict=False).fill_null(False)
            & (
                pl.col("opening_book").is_null()
                | (pl.col("opening_book").cast(pl.String).str.strip_chars() == "")
            )
        ).height
        if verified_missing_book:
            issues.append(
                {
                    "severity": "ERROR",
                    "code": "verified_open_missing_source_book",
                    "rows": verified_missing_book,
                }
            )

    errors = sum(issue["severity"] == "ERROR" for issue in issues)
    warnings = sum(issue["severity"] == "WARNING" for issue in issues)
    return {
        "status": "FAIL" if errors else "WARN" if warnings else "PASS",
        "errors": errors,
        "warnings": warnings,
        "bets": frame.height,
        "issues": issues,
        "quote_integrity": {
            "all_archive_bets": frame.height,
            "verified_opening_entry_bets": verified,
            "unverified_or_final_fallback_bets": fallback,
            "verified_opening_entry_rate": verified / frame.height,
            "promotion_rule": (
                "only has_distinct_open=true rows may contribute to verified-entry "
                "promotion evidence"
            ),
        },
    }


def audit_backtest_file(
    path: str | Path = "reports/free_market_bets.csv",
    *,
    output: str | Path = "reports/backtest_audit.json",
    fail_on_error: bool = True,
) -> dict[str, object]:
    source = Path(path)
    if not source.exists():
        report = audit_backtest_bets(pl.DataFrame())
    else:
        try:
            frame = pl.read_csv(source, try_parse_dates=False)
        except Exception as exc:
            raise DataContractError(f"failed to read backtest bets: {exc}") from exc
        report = audit_backtest_bets(frame)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    if fail_on_error and report["errors"]:
        codes = ", ".join(
            str(issue["code"])
            for issue in report["issues"]
            if issue["severity"] == "ERROR"
        )
        raise RuntimeError(f"backtest audit failed: {codes}")
    return report
