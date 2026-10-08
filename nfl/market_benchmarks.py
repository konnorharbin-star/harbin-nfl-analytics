"""Step 5: prespecified NFL market-only, blended and fixed-policy benchmarks.

Research-only, conditional on model-selected first-quote game/market candidates.
This cannot simulate an independent sportsbook selection strategy or fills.
No forward outcome is used to fit weights or choose a betting threshold.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import polars as pl

from .contracts import require_columns
from .forward_edge_validation import (
    GRADED_FIELDS,
    MARKETS,
    SEASON,
    SPEC_VERSION,
    _num,
    _odds,
    _parse,
    _quote_status,
)
from .grading import _grade_value
from .market import american_to_decimal

BENCHMARK_VERSION = 1
# The weights and rules are locked BEFORE the new 2026 cohort settles.
FORECAST_WEIGHTS = {
    "market_only": 0.0,
    "market_plus_25pct_model": 0.25,
    "market_plus_50pct_model": 0.50,
    "raw_model": 1.0,
}
PAPER_RULES = (
    "no_bet_market_only",
    "same_selected_side_flat",
    "model_ev_positive",
    "model_edge_2pp_and_positive_ev",
    "model_edge_5pp_and_positive_ev",
)
MIN_DECIDED = 100
MIN_GAMES = 80
MIN_WEEKS = 8
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20261008
EPS = 1e-6
BENCHMARK_FIELDS = (
    "cohort", "market", "season", "week", "predictor",
    "frozen", "eligible_quotes", "settled", "decided", "pushes",
    "brier", "log_loss", "ece_10", "market_minus_predictor_brier",
    "market_minus_predictor_log_loss", "paired_brier_ci95",
    "paired_log_loss_ci95", "ci_supported",
)
PAPER_FIELDS = (
    "cohort", "market", "rule", "frozen", "eligible_quotes", "settled",
    "eligible_settled", "hypothetical_bets", "wins", "losses", "pushes",
    "hypothetical_units", "hypothetical_roi_on_bets",
    "units_per_eligible_settled_candidate", "net_units_per_candidate_ci95",
    "ci_supported",
)


def _truth(value: object) -> bool:
    return value is True or str(value or "").strip().lower() == "true"


def _p(value: float) -> float:
    return min(1 - EPS, max(EPS, value))


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _ece(bins: list[tuple[float, float]]) -> float | None:
    """ECE is descriptive only; 10 fixed probability bins, no adaptive split."""
    if not bins:
        return None
    counts = [0] * 10
    diffs = [0.0] * 10
    for prediction, actual in bins:
        index = min(9, int(prediction * 10))
        counts[index] += 1
        diffs[index] += prediction - actual
    return sum(abs(diffs[i]) for i in range(10)) / len(bins)


def _week_ci(rows: list[dict[str, object]], key: str) -> list[float] | None:
    included = [row for row in rows if _num(row.get(key)) is not None]
    groups: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in included:
        groups[(int(row["season"]), int(row["week"]))].append(float(row[key]))
    if (
        len(included) < MIN_DECIDED
        or len({row["game_id"] for row in included}) < MIN_GAMES
        or len(groups) < MIN_WEEKS
    ):
        return None
    pairs = [(sum(value), len(value)) for _, value in sorted(groups.items())]
    sums = np.asarray([pair[0] for pair in pairs], dtype=float)
    counts = np.asarray([pair[1] for pair in pairs], dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampled = rng.integers(0, len(pairs), size=(BOOTSTRAP_REPS, len(pairs)))
    estimates = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    return [round(float(v), 8) for v in np.quantile(estimates, [0.025, 0.975])]


def _reconcile(row: Mapping[str, object]) -> str | None:
    """Fail closed on changed or incorrectly settled forward source records."""
    if str(row.get("spec_version")) != SPEC_VERSION:
        return "WRONG_LOCKED_SPEC"
    if str(row.get("season")) != str(SEASON):
        return "WRONG_FORWARD_SEASON"
    if _parse(row.get("captured_at")) is None or _parse(row.get("kickoff")) is None:
        return "UNTRUSTED_CAPTURE_TIME"
    market = str(row.get("market") or "")
    if market not in MARKETS:
        return "UNSUPPORTED_MARKET"
    observation = str(row.get("observation_status") or "")
    if observation not in {
        "PENDING_RESULT", "GRADED_PROBABILITY",
        "GRADED_PUSH_UNSCORED_PROBABILITY", "INVALID_FROZEN_QUOTE",
    }:
        return "UNSUPPORTED_GRADE_STATUS"
    if observation == "INVALID_FROZEN_QUOTE":
        return None
    if row.get("snapshot_status") != "VALID_POINT_IN_TIME_RESEARCH_QUOTE":
        if observation.startswith("GRADED_"):
            return "GRADED_UNVERIFIED_SNAPSHOT"
        return None
    if _quote_status(row) != "VALID_POINT_IN_TIME_RESEARCH_QUOTE":
        return "NONRECONCILED_FROZEN_QUOTE"
    if observation == "PENDING_RESULT":
        # No grade can be attached to a result that is still pending.
        if row.get("result") not in {None, ""}:
            return "PENDING_ROW_HAS_RESULT"
        return None
    home = _num(row.get("home_score"))
    away = _num(row.get("away_score"))
    odds = _odds(row.get("odds"))
    line = _num(row.get("line"))
    model = _num(row.get("model_probability"))
    market_p = _num(row.get("no_vig_probability"))
    units = _num(row.get("simulated_net_units"))
    if (
        home is None or away is None or odds is None or units is None
        or model is None or market_p is None
    ):
        return "INVALID_COMPLETED_SCORE_PRICE_OR_PROBABILITY"
    outcome = _grade_value(
        market, str(row.get("side")), line, home - away, home + away
    )
    expected_result = "win" if outcome > 1e-12 else "loss" if outcome < -1e-12 else "push"
    if str(row.get("result")) != expected_result:
        return "OUTCOME_NOT_RECONCILED_TO_FROZEN_HANDICAP"
    theoretical_units = (
        american_to_decimal(odds) - 1 if expected_result == "win"
        else -1.0 if expected_result == "loss" else 0.0
    )
    if abs(units - theoretical_units) > 1e-6:
        return "PAYOFF_NOT_RECONCILED_TO_FROZEN_ODDS"
    if expected_result == "push":
        if observation != "GRADED_PUSH_UNSCORED_PROBABILITY":
            return "PUSH_MISCLASSIFIED"
        if row.get("model_brier") not in {None, ""}:
            return "PUSH_HAS_BINARY_LOSS"
        return None
    if observation != "GRADED_PROBABILITY":
        return "NONPUSH_MISSING_PROBABILITY_GRADE"
    actual = 1.0 if expected_result == "win" else 0.0
    values = {
        "model_brier": (model - actual) ** 2,
        "market_brier": (market_p - actual) ** 2,
        "model_log_loss": -math.log(_p(model if actual else 1 - model)),
        "market_log_loss": -math.log(_p(market_p if actual else 1 - market_p)),
    }
    for key, calculated in values.items():
        value = _num(row.get(key))
        if value is None or abs(value - calculated) > 1e-6:
            return "PAIRWISE_SCORE_NOT_RECONCILED"
    return None


def _prepare(
    graded: pl.DataFrame,
    source: Mapping[str, object],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    meta: dict[str, object] = {
        "source_report_present": bool(source),
        "source_status": source.get("status"),
        "source_spec": source.get("spec_version"),
        "source_research_only": _truth(source.get("research_only")),
        "source_bets_authorized": source.get("staking_authorized"),
        "source_model_changed": source.get("production_model_change_enabled"),
        "source_frozen_rows": source.get("frozen_candidate_rows"),
        "source_graded_rows": (
            source.get("summary", {}).get("graded")
            if isinstance(source.get("summary"), dict) else None
        ),
        "source_decided_rows": (
            source.get("summary", {}).get("decided")
            if isinstance(source.get("summary"), dict) else None
        ),
        "included_rows": 0,
        "duplicate_keys": 0,
        "integrity_errors": {},
    }
    if not graded.is_empty():
        require_columns(graded, set(GRADED_FIELDS), "market_benchmark_grades")
    rows = graded.to_dicts()
    seen = set()
    errors = Counter()
    decided, settled = 0, 0
    for row in rows:
        key = (str(row.get("game_id")), str(row.get("market")))
        if not key[0] or key in seen:
            errors["DUPLICATE_OR_EMPTY_GAME_MARKET"] += 1
        seen.add(key)
        issue = _reconcile(row)
        if issue:
            errors[issue] += 1
        decided += row.get("observation_status") == "GRADED_PROBABILITY"
        settled += str(row.get("observation_status")).startswith("GRADED_")
    meta["duplicate_keys"] = sum(v for k, v in errors.items() if k == "DUPLICATE_OR_EMPTY_GAME_MARKET")
    meta["integrity_errors"] = dict(sorted(errors.items()))
    meta["included_rows"] = len(rows)
    source_ok = (
        bool(source)
        and source.get("spec_version") == SPEC_VERSION
        and source.get("status") in {
            "PENDING_FORWARD", "RESEARCH_ONLY_SAMPLE_MATURED_NOT_APPROVED",
        }
        and source.get("research_only") is True
        and source.get("staking_authorized") is False
        and source.get("production_model_change_enabled") is False
        and source.get("forward_results_used_to_refit") is False
        and _num(source.get("frozen_candidate_rows")) == len(rows)
        and isinstance(source.get("summary"), dict)
        and _num(source["summary"].get("graded")) == settled
        and _num(source["summary"].get("decided")) == decided
        and not source.get("duplicate_frozen_keys", 0)
    )
    meta["source_reconciled"] = source_ok
    return rows, meta


def _eligible(row: Mapping[str, object]) -> bool:
    return (
        row.get("snapshot_status") == "VALID_POINT_IN_TIME_RESEARCH_QUOTE"
        and row.get("observation_status")
        in {"GRADED_PROBABILITY", "GRADED_PUSH_UNSCORED_PROBABILITY"}
    )


def _forecast_summary(
    rows: list[dict[str, object]],
    predictor: str,
    weight: float,
    *,
    cohort: str,
    market: str | None = None,
    season: int | None = None,
    week: int | None = None,
) -> dict[str, object]:
    selected = [r for r in rows if _eligible(r)]
    decided = [r for r in selected if r["observation_status"] == "GRADED_PROBABILITY"]
    evaluated = []
    for row in decided:
        actual = 1.0 if row["result"] == "win" else 0.0
        p_market = float(row["no_vig_probability"])
        p_model = float(row["model_probability"])
        p = _p(p_market * (1 - weight) + p_model * weight)
        loss = (p - actual) ** 2
        logloss = -math.log(p if actual else 1 - p)
        eval_row = dict(row)
        eval_row["_loss"] = loss
        eval_row["_logloss"] = logloss
        eval_row["_delta_brier"] = (p_market - actual) ** 2 - loss
        eval_row["_delta_log"] = (
            -math.log(_p(p_market if actual else 1 - p_market)) - logloss
        )
        evaluated.append(eval_row)
    brier_ci = _week_ci(evaluated, "_delta_brier")
    log_ci = _week_ci(evaluated, "_delta_log")
    return {
        "cohort": cohort, "market": market, "season": season, "week": week,
        "predictor": predictor,
        "frozen": len(rows),
        "eligible_quotes": sum(
            r["snapshot_status"] == "VALID_POINT_IN_TIME_RESEARCH_QUOTE" for r in rows
        ),
        "settled": len(selected), "decided": len(evaluated),
        "pushes": len(selected) - len(evaluated),
        "brier": _mean([r["_loss"] for r in evaluated]),
        "log_loss": _mean([r["_logloss"] for r in evaluated]),
        "ece_10": _ece([
            (_p(float(r["no_vig_probability"]) * (1 - weight)
                 + float(r["model_probability"]) * weight),
             1.0 if r["result"] == "win" else 0.0) for r in evaluated
        ]),
        "market_minus_predictor_brier": _mean([
            r["_delta_brier"] for r in evaluated
        ]),
        "market_minus_predictor_log_loss": _mean([
            r["_delta_log"] for r in evaluated
        ]),
        "paired_brier_ci95": brier_ci,
        "paired_log_loss_ci95": log_ci,
        "ci_supported": brier_ci is not None and log_ci is not None,
    }


def _paper_selected(row: Mapping[str, object], rule: str) -> bool:
    ev = float(row["raw_ev"])
    edge = float(row["raw_edge"])
    if rule == "no_bet_market_only":
        return False
    if rule == "same_selected_side_flat":
        return True
    if rule == "model_ev_positive":
        return ev > 0
    if rule == "model_edge_2pp_and_positive_ev":
        return ev > 0 and edge >= 0.02
    if rule == "model_edge_5pp_and_positive_ev":
        return ev > 0 and edge >= 0.05
    raise ValueError(f"unregistered paper rule {rule}")


def _paper_summary(
    rows: list[dict[str, object]],
    rule: str,
    *,
    cohort: str, market: str | None = None,
) -> dict[str, object]:
    settled = [row for row in rows if _eligible(row)]
    eligible_quotes = sum(
        row.get("snapshot_status") == "VALID_POINT_IN_TIME_RESEARCH_QUOTE"
        for row in rows
    )
    scores = []
    hypothetically_bet = []
    for row in settled:
        selected = _paper_selected(row, rule)
        observation = dict(row)
        observation["_strategy_units"] = (
            float(row["simulated_net_units"]) if selected else 0.0
        )
        scores.append(observation)
        if selected:
            hypothetically_bet.append(observation)
    ci = _week_ci(scores, "_strategy_units")
    units = sum(float(row["_strategy_units"]) for row in scores)
    return {
        "cohort": cohort, "market": market, "rule": rule,
        "frozen": len(rows), "eligible_quotes": eligible_quotes,
        "settled": sum(
            str(row.get("observation_status")).startswith("GRADED_") for row in rows
        ),
        "eligible_settled": len(settled),
        "hypothetical_bets": len(hypothetically_bet),
        "wins": sum(r["result"] == "win" for r in hypothetically_bet),
        "losses": sum(r["result"] == "loss" for r in hypothetically_bet),
        "pushes": sum(r["result"] == "push" for r in hypothetically_bet),
        "hypothetical_units": units,
        "hypothetical_roi_on_bets": (
            units / len(hypothetically_bet) if hypothetically_bet else None
        ),
        "units_per_eligible_settled_candidate": (
            units / len(settled) if settled else None
        ),
        "net_units_per_candidate_ci95": ci,
        "ci_supported": ci is not None,
    }


def build_market_benchmarks(
    graded: pl.DataFrame,
    *,
    forward_report: Mapping[str, object] | None,
) -> dict[str, object]:
    """Benchmark only identical locked candidate+game results, zero tuning."""
    rows, integrity = _prepare(graded, forward_report or {})
    error = not integrity["source_reconciled"] or bool(integrity["integrity_errors"])
    forecast_rows = []
    paper_rows = []
    weekly_rows = []
    if not error:
        cohorts = [("all", None, rows)]
        cohorts.extend(
            ("market", market, [row for row in rows if row["market"] == market])
            for market in MARKETS
        )
        for cohort, market, selected in cohorts:
            for name, weight in FORECAST_WEIGHTS.items():
                forecast_rows.append(
                    _forecast_summary(
                        selected, name, weight, cohort=cohort, market=market
                    )
                )
            for rule in PAPER_RULES:
                paper_rows.append(
                    _paper_summary(selected, rule, cohort=cohort, market=market)
                )
        observed_weeks = sorted({
            (int(row["season"]), int(row["week"])) for row in rows
        })
        for season, week in observed_weeks:
            for market in MARKETS:
                selected = [
                    row for row in rows if int(row["season"]) == season
                    and int(row["week"]) == week and row["market"] == market
                ]
                for name, weight in FORECAST_WEIGHTS.items():
                    weekly_rows.append(
                        _forecast_summary(
                            selected, name, weight, cohort="weekly_market",
                            season=season, week=week, market=market,
                        )
                    )
    decided = sum(
        row["observation_status"] == "GRADED_PROBABILITY" for row in rows
    )
    report: dict[str, object] = {
        "schema_version": BENCHMARK_VERSION,
        "status": (
            "BLOCKED_SOURCE_INTEGRITY" if error else
            "PENDING_FORWARD" if not decided else
            "RESEARCH_SAMPLE_UNDERPOWERED" if decided < MIN_DECIDED else
            "RESEARCH_ONLY_NOT_APPROVED"
        ),
        "research_only": True,
        "staking_authorized": False,
        "production_model_change_enabled": False,
        "no_calibration_weights_fitted_to_forward_results": True,
        "no_bet_policy_thresholds_tuned_to_forward_results": True,
        "historical_2025_unseen_holdout_claim": False,
        "prospective_season": SEASON,
        "frozen": len(rows),
        "decided": decided,
        "integrity": integrity,
        "predeclared_forecast_weights": FORECAST_WEIGHTS,
        "predeclared_flat_unit_rules": list(PAPER_RULES),
        "predictor_metrics": forecast_rows,
        "paper_policy_metrics": paper_rows,
        "weekly_market_metrics": weekly_rows,
        "market_only_baseline": "same-side frozen no-vig price at initial quote",
        "market_only_no_bet_baseline": "zero hypothetical units and zero exposure",
        "selection_condition": (
            "The cohort side/book/line was chosen by the model's maximum raw EV "
            "before the game. A market-only forecast here is a CONDITIONAL "
            "benchmark, not an independently market-selected betting strategy."
        ),
        "economic_interpretation": (
            "Flat one-unit paper rules reuse model-selected locked first quote; "
            "no fills, liquidity/slippage constraints, serially settled live "
            "bankroll or genuinely independent sportsbook-strategy results."
        ),
        "intervals": {
            "method": "paired full_kickoff_week_cluster_bootstrap",
            "seed": BOOTSTRAP_SEED,
            "repetitions": BOOTSTRAP_REPS,
            "minimum_decided_or_settled_per_cohort": MIN_DECIDED,
            "minimum_distinct_games": MIN_GAMES,
            "minimum_distinct_weeks": MIN_WEEKS,
            "nominal_confidence": 0.95,
            "multiplicity_adjusted": False,
            "selection_aware": False,
            "promotion_test": False,
        },
        "limitations": [
            "First selected game/market side is model-selected; this is not a "
            "genuinely independent market-first betting-strategy comparison.",
            "Only quote-verified completed and nonpush cases enter Brier/log loss.",
            "For fixed paper rules, push stakes are returned and no-bet stakes "
            "stay at zero; returns are hypothetical at captured book prices.",
            "No closing quote, future line or result changes pre-kickoff selection.",
            "Fixed blends do not reprice a new spread/total handicap.",
            "The same game may appear in three markets: bootstrap resamples "
            "whole kickoff weeks to retain shared-game dependence.",
            "Multiple exploratory comparisons are not multiplicity corrected "
            "and may not be used to choose winning policies on this same cohort.",
            "There are no verified fills or proof that an advertised line was "
            "available to the user at wager time.",
            "Full policy/release/probability/regime gates and a later independent "
            "holdout are required before any production strategy change.",
        ],
    }
    return report


def write_market_benchmarks(
    report: Mapping[str, object],
    *,
    reports_dir: str | Path = "reports",
    outputs_dir: str | Path = "outputs",
    docs_dir: str | Path = "docs",
) -> None:
    """Sync research JSON, forecasts, paper policies and weekly decompositions."""
    payload = json.dumps(dict(report), indent=2, sort_keys=True, default=str)
    data = (
        ("forward_market_benchmark_forecasts.csv", BENCHMARK_FIELDS, "predictor_metrics"),
        ("forward_market_benchmark_policies.csv", PAPER_FIELDS, "paper_policy_metrics"),
        ("forward_market_benchmark_weeks.csv", BENCHMARK_FIELDS, "weekly_market_metrics"),
    )
    for root in (Path(reports_dir), Path(outputs_dir), Path(docs_dir)):
        root.mkdir(parents=True, exist_ok=True)
        (root / "forward_market_benchmarks.json").write_text(
            payload, encoding="utf-8"
        )
        for filename, fields, key in data:
            with (root / filename).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=list(fields), extrasaction="ignore"
                )
                writer.writeheader()
                for record in report.get(key, []):
                    writer.writerow({
                        field: (
                            json.dumps(record.get(field))
                            if isinstance(record.get(field), list)
                            else record.get(field)
                        ) for field in fields
                    })
