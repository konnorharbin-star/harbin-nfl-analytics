"""Guarded feedback from independently graded bets into future risk sizing.

This module never changes fair-score projections and never increases stake above the
base model recommendation. It only de-risks sufficiently sampled segments when both
closing-line value and realized returns are adverse.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from math import isfinite
from pathlib import Path
from typing import Any


def _number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _edge_bucket(value: object) -> str:
    edge = _number(value)
    if edge is None:
        return "missing"
    edge = abs(edge)
    # NFL probability edges are commonly stored as decimals while CFB legacy rows
    # may store percentage points. Keep the buckets useful for either convention.
    if edge > 1.0:
        edge /= 100.0
    if edge < 0.02:
        return "<2%"
    if edge < 0.04:
        return "2-4%"
    if edge < 0.06:
        return "4-6%"
    return ">=6%"


def _segment_stats(rows: list[dict[str, object]]) -> dict[str, object]:
    profits = [_number(row.get("_profit")) for row in rows]
    profits = [value for value in profits if value is not None]
    clv = [_number(row.get("_clv")) for row in rows]
    clv = [value for value in clv if value is not None]
    n = len(profits)
    positive_rate = None
    if clv:
        positive_rate = sum(value > 0 for value in clv) / len(clv)
    return {
        "bets": n,
        "roi": None if not profits else sum(profits) / len(profits),
        "avg_clv": None if not clv else sum(clv) / len(clv),
        "positive_clv_rate": positive_rate,
        "clv_samples": len(clv),
        "clv_coverage": 0.0 if n == 0 else len(clv) / n,
    }


def _classify(
    stats: dict[str, object],
    config: dict[str, object],
) -> tuple[str, float]:
    n = int(stats.get("bets") or 0)
    roi = _number(stats.get("roi"))
    avg_clv = _number(stats.get("avg_clv"))
    positive_clv = _number(stats.get("positive_clv_rate"))
    coverage = _number(stats.get("clv_coverage")) or 0.0

    minimum = max(
        1,
        int(float(config.get("feedback_min_segment_bets", 20))),
    )
    min_coverage = max(
        0.0,
        min(1.0, float(config.get("feedback_min_clv_coverage", 0.60))),
    )
    weak_multiplier = max(
        0.0,
        min(1.0, float(config.get("feedback_weak_multiplier", 0.75))),
    )
    severe_multiplier = max(
        0.0,
        min(
            weak_multiplier,
            float(config.get("feedback_severe_multiplier", 0.50)),
        ),
    )
    severe_min = max(
        minimum,
        int(
            float(
                config.get(
                    "feedback_severe_min_bets",
                    max(40, 2 * minimum),
                )
            )
        ),
    )
    severe_roi = float(config.get("feedback_severe_roi", -0.05))
    severe_positive_clv = float(
        config.get("feedback_severe_positive_clv_rate", 0.45)
    )

    if n < minimum or coverage < min_coverage or roi is None or avg_clv is None:
        return "insufficient", 1.0
    if avg_clv >= 0 or roi >= 0:
        return "healthy", 1.0
    if (
        n >= severe_min
        and roi <= severe_roi
        and positive_clv is not None
        and positive_clv < severe_positive_clv
    ):
        return "severe", severe_multiplier
    return "weak", weak_multiplier


def build_performance_feedback(
    live_bets_path: str | Path = "reports/live_graded_bets.csv",
    config: dict[str, object] | None = None,
) -> dict[str, object]:
    """Summarize segment health from timestamp-safe graded bets.

    Multipliers are one-sided: 1.0 is the maximum. Positive historical performance
    never increases risk; only sufficiently sampled adverse CLV plus ROI can reduce it.
    """

    cfg = config or {}
    path = Path(live_bets_path)
    empty = {
        "enabled": bool(cfg.get("enable_performance_feedback", True)),
        "history_available": False,
        "graded_bets": 0,
        "segments": {},
        "reason": "no independent graded betting history",
    }
    if not empty["enabled"]:
        return {
            **empty,
            "reason": "performance feedback disabled by policy",
        }
    if not path.exists() or not path.stat().st_size:
        return empty

    try:
        with path.open(newline="") as handle:
            raw = list(csv.DictReader(handle))
    except OSError as exc:
        return {
            **empty,
            "reason": (
                "graded betting history unreadable: "
                f"{type(exc).__name__}"
            ),
        }

    normalized: list[dict[str, object]] = []
    for row in raw:
        profit = _number(row.get("net_units"))
        if profit is None:
            profit = _number(row.get("profit"))
        if profit is None:
            continue

        clv = _number(row.get("execution_clv"))
        if clv is None:
            clv = _number(row.get("clv_proxy"))

        item = dict(row)
        item["_profit"] = profit
        item["_clv"] = clv
        item["_edge_bucket"] = str(
            row.get("edge_bucket") or _edge_bucket(row.get("quant_edge"))
        )
        normalized.append(item)

    if not normalized:
        return {
            **empty,
            "history_available": True,
            "reason": "graded history has no usable profit rows",
        }

    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in normalized:
        signal = row.get("portfolio_signal") or row.get("quant_signal") or ""
        dimensions = {
            "market": str(row.get("quant_market") or "").strip().lower(),
            "book": str(row.get("quant_book") or "").strip(),
            "signal": str(signal).strip().upper(),
            "edge_bucket": str(row.get("_edge_bucket") or "missing"),
        }
        for kind, value in dimensions.items():
            if value and value.lower() not in {"nan", "none", "unknown"}:
                groups[(kind, value)].append(row)

    segments: dict[str, dict[str, object]] = {}
    for (kind, value), rows in groups.items():
        stats = _segment_stats(rows)
        state, multiplier = _classify(stats, cfg)
        segments[f"{kind}:{value}"] = {
            "dimension": kind,
            "value": value,
            **stats,
            "state": state,
            "multiplier": multiplier,
        }

    overall = _segment_stats(normalized)
    return {
        "enabled": True,
        "history_available": True,
        "graded_bets": len(normalized),
        "overall": overall,
        "segments": segments,
        "reason": "CLV-first segment de-risking; multipliers never exceed 1.0",
    }


def performance_feedback_for_row(
    row: dict[str, Any],
    report: dict[str, object],
) -> dict[str, object]:
    """Return the most conservative eligible segment multiplier for one candidate."""

    if not report.get("enabled") or not report.get("history_available"):
        return {
            "multiplier": 1.0,
            "reason": "neutral: insufficient feedback history",
            "matched_segments": [],
        }

    signal = (
        row.get("portfolio_signal")
        or row.get("research_signal")
        or row.get("quant_signal")
        or ""
    )
    dimensions = {
        "market": str(row.get("quant_market") or "").strip().lower(),
        "book": str(row.get("quant_book") or "").strip(),
        "signal": str(signal).strip().upper(),
        "edge_bucket": str(
            row.get("edge_bucket") or _edge_bucket(row.get("quant_edge"))
        ),
    }
    segments = report.get("segments")
    if not isinstance(segments, dict):
        return {
            "multiplier": 1.0,
            "reason": "neutral: no segment diagnostics",
            "matched_segments": [],
        }

    matched: list[dict[str, object]] = []
    multiplier = 1.0
    for kind, value in dimensions.items():
        if not value:
            continue
        segment = segments.get(f"{kind}:{value}")
        if not isinstance(segment, dict):
            continue

        state = str(segment.get("state") or "insufficient")
        raw_multiplier = _number(segment.get("multiplier"))
        seg_multiplier = 1.0 if raw_multiplier is None else raw_multiplier
        if state in {"weak", "severe"} and seg_multiplier < 1.0:
            multiplier = min(multiplier, seg_multiplier)
            matched.append(
                {
                    "segment": f"{kind}:{value}",
                    "state": state,
                    "bets": int(segment.get("bets") or 0),
                    "roi": segment.get("roi"),
                    "avg_clv": segment.get("avg_clv"),
                    "positive_clv_rate": segment.get("positive_clv_rate"),
                    "multiplier": seg_multiplier,
                }
            )

    if not matched:
        return {
            "multiplier": 1.0,
            "reason": "neutral: no sufficiently sampled adverse segment",
            "matched_segments": [],
        }

    labels = ", ".join(
        f"{item['segment']}={item['state']}" for item in matched
    )
    return {
        "multiplier": max(0.0, min(1.0, multiplier)),
        "reason": f"de-risked by graded CLV/ROI feedback: {labels}",
        "matched_segments": matched,
    }
