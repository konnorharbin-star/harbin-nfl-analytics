"""Training/live feature-stack audit aligned with the CFB evidence layer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

MARKET_TOKENS = (
    "odds",
    "moneyline",
    "spread_line",
    "market_",
    "opening_",
    "closing_",
    "clv",
    "quant_",
    "sportsbook",
    "no_vig",
)
TARGET_TOKENS = (
    "actual_",
    "residual",
    "home_score",
    "away_score",
    "final_score",
    "game_result",
    "net_units",
    "profit",
)


def leakage_columns(columns: list[str]) -> list[str]:
    """Return feature names that would expose sportsbook or outcome information."""

    leaking: list[str] = []
    for column in columns:
        lower = column.lower()
        if any(token in lower for token in (*MARKET_TOKENS, *TARGET_TOKENS)):
            leaking.append(column)
    return leaking


def _numeric(frame: pl.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        return np.asarray([], dtype=float)
    series = frame.get_column(column).cast(pl.Float64, strict=False).drop_nulls()
    if series.is_empty():
        return np.asarray([], dtype=float)
    values = np.asarray(series, dtype=float)
    return values[np.isfinite(values)]


def _summary(train: pl.DataFrame, live: pl.DataFrame, column: str) -> dict[str, object]:
    train_values = _numeric(train, column)
    live_values = _numeric(live, column)
    train_rate = train_values.size / max(1, train.height)
    live_rate = live_values.size / max(1, live.height)
    median: float | None = None
    iqr: float | None = None
    live_median: float | None = None
    drift_iqr: float | None = None
    if train_values.size:
        q25, median_value, q75 = np.quantile(train_values, [0.25, 0.5, 0.75])
        median = float(median_value)
        iqr = float(q75 - q25)
    if live_values.size:
        live_median = float(np.median(live_values))
    if (
        median is not None
        and live_median is not None
        and iqr is not None
        and iqr > 1e-12
    ):
        drift_iqr = abs(live_median - median) / iqr
    return {
        "train_nonnull_rate": train_rate,
        "live_nonnull_rate": live_rate,
        "train_median": median,
        "train_iqr": iqr,
        "live_median": live_median,
        "live_median_drift_iqr": drift_iqr,
    }


def audit_feature_stack(
    train: pl.DataFrame,
    live: pl.DataFrame,
    model_columns: list[str],
    *,
    asof_policy: str | None = None,
) -> dict[str, object]:
    """Audit feature parity, leakage, missingness, and live distribution shift."""

    issues: list[dict[str, object]] = []
    missing_train = [column for column in model_columns if column not in train.columns]
    missing_live = [column for column in model_columns if column not in live.columns]
    leaking = leakage_columns(model_columns)
    if len(set(train.columns)) != len(train.columns):
        issues.append({"severity": "ERROR", "code": "duplicate_train_columns"})
    if len(set(live.columns)) != len(live.columns):
        issues.append({"severity": "ERROR", "code": "duplicate_live_columns"})
    if missing_train:
        issues.append(
            {
                "severity": "ERROR",
                "code": "missing_train_features",
                "columns": missing_train,
            }
        )
    if missing_live:
        issues.append(
            {
                "severity": "ERROR",
                "code": "missing_live_features",
                "columns": missing_live,
            }
        )
    if leaking:
        issues.append(
            {
                "severity": "ERROR",
                "code": "leakage_features",
                "columns": leaking,
            }
        )

    manifest: dict[str, object] = {}
    sparse: list[str] = []
    live_empty: list[str] = []
    high_drift: list[str] = []
    for column in model_columns:
        if column not in train.columns or column not in live.columns:
            continue
        summary = _summary(train, live, column)
        manifest[column] = summary
        if float(summary["train_nonnull_rate"]) < 0.20:
            sparse.append(column)
        if float(summary["live_nonnull_rate"]) == 0.0:
            live_empty.append(column)
        drift = summary["live_median_drift_iqr"]
        if drift is not None and float(drift) > 4.0:
            high_drift.append(column)

    if sparse:
        issues.append(
            {
                "severity": "WARNING",
                "code": "sparse_training_features",
                "columns": sparse,
            }
        )
    if live_empty:
        issues.append(
            {
                "severity": "ERROR",
                "code": "empty_live_features",
                "columns": live_empty,
            }
        )
    if high_drift:
        issues.append(
            {
                "severity": "WARNING",
                "code": "large_live_distribution_shift",
                "columns": high_drift,
            }
        )

    errors = sum(issue["severity"] == "ERROR" for issue in issues)
    warnings = sum(issue["severity"] == "WARNING" for issue in issues)
    return {
        "status": "FAIL" if errors else "WARN" if warnings else "PASS",
        "errors": errors,
        "warnings": warnings,
        "model_feature_count": len(model_columns),
        "train_rows": train.height,
        "live_rows": live.height,
        "asof_policy": asof_policy,
        "issues": issues,
        "manifest": manifest,
    }


def write_feature_audit(
    train: pl.DataFrame,
    live: pl.DataFrame,
    model_columns: list[str],
    *,
    asof_policy: str | None = None,
    path: str | Path = "reports/feature_audit.json",
    fail_on_error: bool = True,
) -> dict[str, object]:
    report = audit_feature_stack(
        train,
        live,
        model_columns,
        asof_policy=asof_policy,
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    if fail_on_error and report["errors"]:
        codes = ", ".join(
            str(issue["code"])
            for issue in report["issues"]
            if issue["severity"] == "ERROR"
        )
        raise RuntimeError(f"feature-stack audit failed: {codes}")
    return report
