"""Historical market-relative NFL residual challenger: strictly research only.

Fit candidate residual sign and weight on 2024 archive outcomes, report 2025
diagnostic evaluation separately. Both seasons have ALREADY been inspected in
the repo; 2025 is NOT a pristine holdout. No real executable price evidence.
This deliberately does not change the independent football model or bet policy.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

MARKETS = ("moneyline", "spread", "total")
# Declared grid includes possible historically *anti*-informative model signal.
ALPHAS = tuple(round(i / 40, 3) for i in range(-20, 21))
DEVELOPMENT_SEASON = 2024
DIAGNOSTIC_SEASON = 2025
MIN_DEV_OBSERVATIONS = 150
MIN_DIAGNOSTIC_OBSERVATIONS = 150
ALPHA_PENALTY = 0.01


def _probability(value: object) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return x if math.isfinite(x) and 0 < x < 1 else None


def market_logit(p: float) -> float:
    return math.log(p / (1 - p))


def residual_prediction(model: float, no_vig_market: float, alpha: float) -> float:
    """Zero alpha = market baseline. The model is only a tested residual."""
    z = market_logit(no_vig_market) + alpha * (
        market_logit(model) - market_logit(no_vig_market)
    )
    return 1 / (1 + math.exp(-z))


def _rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    parsed = []
    rejected = 0
    with path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            model = _probability(row.get("model_probability"))
            market = _probability(row.get("no_vig_probability"))
            result = str(row.get("result") or "").lower()
            mkt = str(row.get("market_type") or "")
            season = str(row.get("season") or "")
            try:
                dec = float(row.get("decimal_odds") or "nan")
            except ValueError:
                dec = float("nan")
            if (
                model is None or market is None or result not in ("win", "loss")
                or mkt not in MARKETS or season not in ("2024", "2025")
                or not math.isfinite(dec) or dec <= 1
                or not str(row.get("game_id") or "")
            ):
                rejected += 1
                continue
            parsed.append({
                "game_id": str(row["game_id"]),
                "season": int(season), "week": int(row.get("week") or 0),
                "market": mkt, "model": model, "market_p": market,
                "decimal": dec, "win": int(result == "win"),
                "entry_timestamp_verified": (
                    str(row.get("entry_timestamp_verified")).lower() == "true"
                ),
            })
    return parsed, {"excluded_missing_invalid_or_push": rejected,
                    "usable_nonpush_archive_rows": len(parsed)}


def _loss(rows: list[dict[str, Any]], alpha: float) -> tuple[float, float]:
    ll = 0.0
    brier = 0.0
    for row in rows:
        p = residual_prediction(row["model"], row["market_p"], alpha)
        p = max(1e-10, min(1-1e-10,p))
        y = row["win"]
        ll -= y * math.log(p) + (1-y)*math.log1p(-p)
        brier += (p-y)**2
    return (ll/len(rows),brier/len(rows))


def _score(rows: list[dict[str, Any]], alpha: float) -> dict[str, Any]:
    if not rows:
        return {"rows": 0, "log_loss": None, "brier": None,
                "hypothetical_archived_bets_over_3pct": 0,
                "hypothetical_archive_roi": None}
    ll, brier = _loss(rows, alpha)
    payoffs = []
    for row in rows:
        p = residual_prediction(row["model"], row["market_p"], alpha)
        if p * row["decimal"] - 1 > .03:
            payoffs.append(
                row["decimal"] - 1 if row["win"] else -1.0
            )
    return {
        "rows": len(rows), "log_loss": round(ll,8),
        "brier": round(brier,8),
        "hypothetical_archived_bets_over_3pct": len(payoffs),
        "hypothetical_archive_roi": (
            round(sum(payoffs)/len(payoffs),8) if payoffs else None
        ),
    }


def evaluate_residual_challenger(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """2024-selected residual alpha is evaluated once on already-seen 2025."""
    results = {}
    for market in MARKETS:
        development = [x for x in rows
                       if x["season"] == DEVELOPMENT_SEASON and x["market"] == market]
        diagnostic = [x for x in rows
                      if x["season"] == DIAGNOSTIC_SEASON and x["market"] == market]
        if len(development) < MIN_DEV_OBSERVATIONS or len(diagnostic) < MIN_DIAGNOSTIC_OBSERVATIONS:
            results[market] = {
                "status":"INSUFFICIENT_RESEARCH_SAMPLE",
                "development_rows":len(development),
                "diagnostic_rows":len(diagnostic),
            }
            continue
        # Positive/negative alpha is selected on development log loss only,
        # never selected by diagnostic ROI or diagnostic model performance.
        alpha = min(
            ALPHAS,
            key=lambda a: (_loss(development,a)[0] + ALPHA_PENALTY*a*a,
                           abs(a),a),
        )
        baseline = _score(diagnostic, 0)
        candidate = _score(diagnostic, alpha)
        raw = _score(diagnostic, 1)
        results[market] = {
            "status": "HISTORICAL_DIAGNOSTIC_NOT_PRODUCTION_VALIDATED",
            "alpha_selected_on_2024": alpha,
            "2024_development_rows":len(development),
            "2025_diagnostic_rows":len(diagnostic),
            "2025_market_only":baseline,
            "2025_residual_candidate":candidate,
            "2025_raw_model":raw,
            "market_minus_candidate_brier":round(
                baseline["brier"] - candidate["brier"],8
            ),
            "market_minus_candidate_log_loss":round(
                baseline["log_loss"] - candidate["log_loss"],8
            ),
            "positive_archived_model_edge_proven":False,
            "deployment_authorized":False,
        }
    return {
        "schema_version":1,
        "training_period":"2024 archived provider-labeled prices",
        "diagnostic_period":"2025 already-inspected non-pristine archive",
        "candidate_design":"penalized logit residual against no-vig market",
        "alpha_grid":[*ALPHAS],
        "development_only_alpha_selection":True,
        "2025_is_pristine_holdout":False,
        "independently_timestamped_entry_prices_verified":False,
        "betting_policy_change_enabled":False,
        "automatic_betting_enabled":False,
        "paid_sources_used":False,
        "limitations": (
            "Both seasons have been examined in prior model research and the archive "
            "entry timestamps are unverified. Comparisons cannot establish a "
            "tradable edge or justify changed betting rules. Even lower Brier or "
            "log loss than a market-only benchmark does not establish profitable EV."
        ),
        "by_market":results,
    }


def write_residual_challenger(
    source: str | Path = "reports/verified_market_bets.csv",
    destination: str | Path = "docs/market_residual_challenger.json",
) -> dict[str, Any]:
    data,diagnostics = _rows(Path(source))
    report = evaluate_residual_challenger(data)
    report["input_diagnostics"] = diagnostics
    target = Path(destination)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    return report


if __name__ == "__main__":
    print(json.dumps(write_residual_challenger()["by_market"],indent=2))
