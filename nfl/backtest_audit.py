"""Audit free NFL historical betting evidence and entry-price provenance."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from .contracts import DataContractError, require_columns

BASE_REQUIRED = {
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
ENTRY_REQUIRED = {
    "entry_line_observed",
    "entry_price_verified",
    "entry_price_stage",
}


def _bool_expr(name: str) -> pl.Expr:
    return pl.col(name).cast(pl.Boolean, strict=False).fill_null(False)


def _normalize_entry_provenance(frame: pl.DataFrame) -> tuple[pl.DataFrame, bool]:
    """Accept legacy research rows while refusing to infer verified entry prices.

    Older backtest CSVs only carried ``has_distinct_open``. That is enough to preserve
    opening-line coverage for diagnostics, but it is not enough to prove the simulated
    price itself was observed at open. Missing Stage 13 fields therefore default to
    research-only provenance.
    """

    legacy = not ENTRY_REQUIRED.issubset(frame.columns)
    output = frame
    if "entry_line_observed" not in output.columns:
        output = output.with_columns(
            _bool_expr("has_distinct_open").alias("entry_line_observed")
        )
    if "entry_price_verified" not in output.columns:
        output = output.with_columns(pl.lit(False).alias("entry_price_verified"))
    if "entry_price_stage" not in output.columns:
        output = output.with_columns(
            pl.when(_bool_expr("entry_price_verified"))
            .then(pl.lit("archive_open_price"))
            .when(_bool_expr("entry_line_observed"))
            .then(pl.lit("archive_open_line_final_price"))
            .otherwise(pl.lit("archive_final_fallback"))
            .alias("entry_price_stage")
        )
    return output, legacy


def audit_backtest_bets(frame: pl.DataFrame) -> dict[str, object]:
    """Separate broad archive research from promotion-quality opening-price evidence."""

    if frame.is_empty():
        return {
            "status": "WARN",
            "errors": 0,
            "warnings": 1,
            "bets": 0,
            "issues": [{"severity": "WARNING", "code": "empty_backtest"}],
            "quote_integrity": {
                "all_archive_bets": 0,
                "opening_line_observed_bets": 0,
                "opening_price_verified_bets": 0,
                "verified_opening_entry_bets": 0,
                "research_only_entry_price_bets": 0,
                "final_fallback_bets": 0,
                "unverified_or_final_fallback_bets": 0,
                "opening_price_verified_rate": 0.0,
                "verified_opening_entry_rate": 0.0,
            },
        }
    require_columns(frame, BASE_REQUIRED, "free_market_backtest_bets")
    frame, legacy_provenance = _normalize_entry_provenance(frame)
    require_columns(frame, BASE_REQUIRED | ENTRY_REQUIRED, "free_market_backtest_bets")

    issues: list[dict[str, object]] = []
    if legacy_provenance:
        issues.append(
            {
                "severity": "WARNING",
                "code": "legacy_entry_provenance_fail_closed",
                "meaning": (
                    "legacy rows preserve opening-line coverage but count zero verified "
                    "entry prices until regenerated with Stage 13 provenance"
                ),
            }
        )

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

    allowed_legacy_stages = {"archive_open_line_final_price", "archive_final_fallback"}
    invalid_legacy_stage = frame.filter(
        ~pl.col("price_stage").is_in(sorted(allowed_legacy_stages))
    ).height
    if invalid_legacy_stage:
        issues.append(
            {
                "severity": "ERROR",
                "code": "invalid_price_stage",
                "rows": invalid_legacy_stage,
            }
        )

    allowed_entry_stages = {
        "archive_open_price",
        "archive_open_line_final_price",
        "archive_final_fallback",
    }
    invalid_entry_stage = frame.filter(
        ~pl.col("entry_price_stage").is_in(sorted(allowed_entry_stages))
    ).height
    if invalid_entry_stage:
        issues.append(
            {
                "severity": "ERROR",
                "code": "invalid_entry_price_stage",
                "rows": invalid_entry_stage,
            }
        )

    legacy_distinct = _bool_expr("has_distinct_open")
    line_observed = _bool_expr("entry_line_observed")
    price_verified = _bool_expr("entry_price_verified")
    market = pl.col("market_type").cast(pl.String).str.to_lowercase()

    line_flag_mismatch = frame.filter(legacy_distinct != line_observed).height
    if line_flag_mismatch:
        issues.append(
            {
                "severity": "ERROR",
                "code": "opening_line_flag_mismatch",
                "rows": line_flag_mismatch,
            }
        )

    price_without_line = frame.filter(price_verified & ~line_observed).height
    if price_without_line:
        issues.append(
            {
                "severity": "ERROR",
                "code": "verified_price_without_opening_line",
                "rows": price_without_line,
            }
        )

    unsupported_verified_market = frame.filter(
        price_verified & (market != "moneyline")
    ).height
    if unsupported_verified_market:
        issues.append(
            {
                "severity": "ERROR",
                "code": "verified_opening_price_on_unsupported_market",
                "rows": unsupported_verified_market,
                "meaning": (
                    "The current free nflverse source does not provide opening juice "
                    "for spread or total."
                ),
            }
        )

    expected_stage = (
        pl.when(price_verified)
        .then(pl.lit("archive_open_price"))
        .when(line_observed)
        .then(pl.lit("archive_open_line_final_price"))
        .otherwise(pl.lit("archive_final_fallback"))
    )
    stage_mismatch = frame.filter(
        pl.col("entry_price_stage").cast(pl.String) != expected_stage
    ).height
    if stage_mismatch:
        issues.append(
            {
                "severity": "ERROR",
                "code": "entry_price_stage_mismatch",
                "rows": stage_mismatch,
            }
        )

    observed_lines = int(
        frame.get_column("entry_line_observed")
        .cast(pl.Boolean, strict=False)
        .fill_null(False)
        .sum()
    )
    verified_prices = int(
        frame.get_column("entry_price_verified")
        .cast(pl.Boolean, strict=False)
        .fill_null(False)
        .sum()
    )
    final_fallbacks = frame.height - observed_lines
    line_only = observed_lines - verified_prices
    research_only = frame.height - verified_prices

    if final_fallbacks:
        issues.append(
            {
                "severity": "WARNING",
                "code": "archive_final_fallback_present",
                "rows": final_fallbacks,
                "meaning": "valid research rows; no separate opening line was observed",
            }
        )
    if line_only:
        issues.append(
            {
                "severity": "WARNING",
                "code": "opening_line_without_opening_price",
                "rows": line_only,
                "meaning": (
                    "opening line observed, but simulated price uses archive-final juice; "
                    "excluded from promotion-quality entry-price evidence"
                ),
            }
        )

    if "clv_proxy" in frame.columns:
        manufactured = frame.filter(
            (~line_observed) & pl.col("clv_proxy").is_not_null()
        ).height
        if manufactured:
            issues.append(
                {
                    "severity": "ERROR",
                    "code": "clv_proxy_without_opening_line",
                    "rows": manufactured,
                }
            )

    if "opening_book" in frame.columns:
        observed_missing_book = frame.filter(
            line_observed
            & (
                pl.col("opening_book").is_null()
                | (pl.col("opening_book").cast(pl.String).str.strip_chars() == "")
            )
        ).height
        if observed_missing_book:
            issues.append(
                {
                    "severity": "ERROR",
                    "code": "observed_open_missing_source_book",
                    "rows": observed_missing_book,
                }
            )

    errors = sum(issue["severity"] == "ERROR" for issue in issues)
    warnings = sum(issue["severity"] == "WARNING" for issue in issues)
    verified_rate = verified_prices / frame.height
    return {
        "status": "FAIL" if errors else "WARN" if warnings else "PASS",
        "errors": errors,
        "warnings": warnings,
        "bets": frame.height,
        "issues": issues,
        "quote_integrity": {
            "all_archive_bets": frame.height,
            "opening_line_observed_bets": observed_lines,
            "opening_price_verified_bets": verified_prices,
            "verified_opening_entry_bets": verified_prices,
            "research_only_entry_price_bets": research_only,
            "final_fallback_bets": final_fallbacks,
            "unverified_or_final_fallback_bets": research_only,
            "opening_price_verified_rate": verified_rate,
            "verified_opening_entry_rate": verified_rate,
            "promotion_rule": (
                "only entry_price_verified=true rows may contribute to promotion-quality "
                "historical entry evidence"
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
