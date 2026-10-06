"""Operational/model monitoring aligned with the CFB live-readiness layer."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl


def _clip(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, float(value)))


def _numeric(frame: pl.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        return np.asarray([], dtype=float)
    values = np.asarray(frame.get_column(column).drop_nulls(), dtype=float)
    return values[np.isfinite(values)]


def _distribution_drift(live: np.ndarray, reference: np.ndarray) -> dict[str, object] | None:
    if live.size < 5 or reference.size < 50:
        return None
    ref_std = max(float(reference.std(ddof=1)), 1e-6)
    live_std = float(live.std(ddof=1)) if live.size > 1 else ref_std
    mean_z = abs(float(live.mean()) - float(reference.mean())) / ref_std
    ratio = live_std / ref_std
    score = _clip(100 - min(70.0, 25.0 * mean_z + 25.0 * abs(math.log(max(0.1, ratio)))))
    return {
        "stability_score": round(score, 1),
        "mean_z": round(mean_z, 4),
        "std_ratio": round(ratio, 4),
        "live_mean": float(live.mean()),
        "reference_mean": float(reference.mean()),
        "live_std": live_std,
        "reference_std": ref_std,
        "live_n": int(live.size),
        "reference_n": int(reference.size),
    }


def build_live_monitoring(
    current: pl.DataFrame,
    meta: dict[str, object],
    *,
    reports_dir: str | Path = "reports",
    now: datetime | None = None,
) -> dict[str, object]:
    alerts: list[str] = []
    scores: dict[str, float] = {}

    market = meta.get("market_coverage")
    if not isinstance(market, dict):
        market = {}
    current_games = (
        current.get_column("game_id").n_unique() if "game_id" in current.columns else 0
    )
    games = max(1, int(market.get("games", current_games) or 0))
    complete = min(
        int(market.get("moneyline", 0) or 0),
        int(market.get("spread", 0) or 0),
        int(market.get("total", 0) or 0),
    )
    scores["market_coverage"] = _clip(100.0 * complete / games)

    intelligence = meta.get("market_intelligence")
    if not isinstance(intelligence, dict):
        intelligence = {}
    scores["multi_book"] = _clip(
        100.0 * float(intelligence.get("multi_book_coverage", 0.0) or 0.0)
    )

    probability = meta.get("probability")
    if not isinstance(probability, dict):
        probability = {}
    brier = probability.get("home_win_brier")
    ece = probability.get("home_win_ece")
    margin_80 = probability.get("margin_80_coverage")
    total_80 = probability.get("total_80_coverage")
    mid_games = int(probability.get("mid_confidence_games", 0) or 0)
    mid_gap = probability.get("mid_confidence_gap")
    calibration = 65.0
    if brier is not None:
        calibration = 100.0 - 200.0 * max(0.0, float(brier) - 0.20)
    if ece is not None:
        calibration -= 150.0 * max(0.0, float(ece) - 0.04)
    if margin_80 is not None:
        calibration -= 100.0 * abs(float(margin_80) - 0.80)
    if total_80 is not None:
        calibration -= 100.0 * abs(float(total_80) - 0.80)
    if mid_games >= 30 and mid_gap is not None:
        calibration -= 150.0 * max(0.0, float(mid_gap) - 0.04)
    if (
        "reliability_ready" in probability
        and not bool(probability.get("reliability_ready"))
    ):
        calibration = min(calibration, 55.0)
        alerts.append("probability reliability check is not ready")
    scores["calibration"] = _clip(calibration)

    context = meta.get("current_context")
    if not isinstance(context, dict):
        context = {}
    scores["context"] = _clip(100.0 * float(context.get("coverage", 0.0) or 0.0))

    generated_raw = meta.get("generated_at")
    reference_now = (now or datetime.now(UTC)).astimezone(UTC)
    try:
        generated = datetime.fromisoformat(str(generated_raw).replace("Z", "+00:00"))
        if generated.tzinfo is None:
            generated = generated.replace(tzinfo=UTC)
        age_hours = (reference_now - generated.astimezone(UTC)).total_seconds() / 3600.0
    except (TypeError, ValueError):
        age_hours = 999.0
    scores["freshness"] = _clip(100.0 - 6.0 * max(0.0, age_hours - 1.0))
    if age_hours > 8:
        alerts.append(f"model output is {age_hours:.1f} hours old")

    if current.is_empty():
        missing_rate = 1.0
    else:
        total_cells = max(1, current.height * len(current.columns))
        missing_cells = sum(
            current.get_column(column).null_count() for column in current.columns
        )
        missing_rate = missing_cells / total_cells
    scores["output_completeness"] = _clip(
        100.0 * (1.0 - min(0.5, missing_rate) / 0.5)
    )

    reference_path = Path(reports_dir) / "free_market_predictions.csv"
    drift_details: dict[str, object] = {}
    if reference_path.exists() and not current.is_empty():
        try:
            historical = pl.read_csv(reference_path)
            mappings = (
                ("model_margin_home", "projected_home_margin", "margin"),
                ("model_total", "projected_total", "total"),
                ("calibrated_home_probability", "home_win_probability", "probability"),
            )
            for live_column, historical_column, name in mappings:
                drift = _distribution_drift(
                    _numeric(current, live_column),
                    _numeric(historical, historical_column),
                )
                if drift is not None:
                    drift_details[name] = drift
                    if float(drift["stability_score"]) < 55:
                        alerts.append(f"{name} prediction distribution shifted materially")
        except Exception as exc:  # source health must be visible, not hidden
            alerts.append(f"drift reference unavailable: {type(exc).__name__}")

    drift_scores = [float(value["stability_score"]) for value in drift_details.values()]
    scores["distribution_stability"] = (
        round(float(np.mean(drift_scores)), 1) if drift_scores else 65.0
    )

    if scores["market_coverage"] < 90:
        alerts.append("verified ML/spread/total coverage below 90%")
    if scores["multi_book"] < 50:
        alerts.append("multi-book consensus is limited")
    if scores["context"] < 60:
        alerts.append("NFL injury/weather/travel context coverage is not yet release-ready")

    weights = {
        "market_coverage": 0.18,
        "multi_book": 0.08,
        "context": 0.15,
        "calibration": 0.22,
        "freshness": 0.12,
        "distribution_stability": 0.15,
        "output_completeness": 0.10,
    }
    score = sum(weights[name] * scores.get(name, 0.0) for name in weights)
    engineering_weights = {
        name: weight
        for name, weight in weights.items()
        if name != "multi_book"
    }
    engineering_weight_total = sum(engineering_weights.values())
    engineering_score = (
        sum(
            engineering_weights[name] * scores.get(name, 0.0)
            for name in engineering_weights
        )
        / engineering_weight_total
        if engineering_weight_total
        else 0.0
    )
    severe_drift = any(
        float(value["stability_score"]) < 40 for value in drift_details.values()
    )
    status = "ALERT" if score < 60 or severe_drift else "WARN" if alerts else "OK"
    return {
        "status": status,
        "live_readiness_score": round(score, 1),
        "engineering_readiness_score": round(engineering_score, 1),
        "scores": scores,
        "drift_details": drift_details,
        "drift_reference": str(reference_path) if reference_path.exists() else None,
        "alerts": list(dict.fromkeys(alerts)),
        "meaning": "operational/model-monitoring score; not a profitability guarantee",
    }


def write_live_monitoring(
    current: pl.DataFrame,
    meta: dict[str, object],
    *,
    output: str | Path = "outputs/live_monitoring.json",
    reports_dir: str | Path = "reports",
) -> dict[str, object]:
    report = build_live_monitoring(current, meta, reports_dir=reports_dir)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    return report
