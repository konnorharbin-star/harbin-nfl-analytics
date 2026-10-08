"""NFL Step 3: preregistered descriptive diagnosis of raw EV versus outcomes.

Strictly observational historical *archive* prices: the ESPN provider-labeled
open is NOT a timestamp-verified sportsbook entry. This module never tunes a
betting policy, trades a holdout segment, reweights the independent football
forecast, authorizes units, or calls the archived closing proxy official CLV.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import polars as pl

from .contracts import DataContractError, require_columns

SCHEMA_VERSION = 1
MARKETS = ("moneyline", "spread", "total")
VALIDATION_SEASON = 2024
DIAGNOSTIC_HOLDOUT_SEASON = 2025
EPS = 1e-6
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20261007
MIN_CI_ROWS = 100
MIN_CI_GAMES = 60
MIN_CI_WEEKS = 8
REQUIRED = {
    "game_id", "season", "week", "market_type", "side", "book",
    "model_probability", "no_vig_probability", "decimal_odds",
    "probability_edge", "expected_value_per_unit", "result", "net_units",
    "entry_timestamp_verified", "entry_quote_verified",
    "entry_price_verified", "entry_price_stage",
}
TABLE_COLUMNS = (
    "cohort", "segment", "season", "market", "rows", "games", "week_blocks",
    "timestamp_verified_rows", "observed_wins", "observed_losses",
    "pushes", "decided_rows", "mean_model_probability",
    "observed_win_rate", "model_calibration_gap",
    "mean_no_vig_probability", "realized_minus_market_probability",
    "mean_raw_probability_edge", "realized_edge_retention_ratio",
    "raw_positive_fraction", "mean_expected_value_per_unit",
    "archive_roi_per_unit", "ev_realization_gap", "mean_market_only_ev",
    "model_brier", "market_brier", "model_minus_market_brier",
    "model_log_loss", "market_log_loss", "model_minus_market_log_loss",
    "ci_95_archive_roi", "ci_95_model_minus_market_brier",
    "ci_95_ev_realization_gap", "evidence_status",
)


def _num(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _yes(value: object) -> bool:
    return value is True or (
        isinstance(value, str) and value.strip().lower() == "true"
    )


def _json(path: str | Path) -> dict[str, object] | None:
    source = Path(path)
    if not source.exists() or not source.stat().st_size:
        return None
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _edge_band(edge: float) -> str:
    """FROZEN categories, not thresholds discovered by optimizing ROI."""
    if edge <= 0:
        return "nonpositive"
    if edge < 0.02:
        return "(0,2pp)"
    if edge < 0.05:
        return "[2,5pp)"
    if edge < 0.10:
        return "[5,10pp)"
    if edge < 0.20:
        return "[10,20pp)"
    return "20pp_plus"


def _probability_band(p: float) -> str:
    if p < 0.50:
        return "below_50"
    if p < 0.55:
        return "[50,55)"
    if p < 0.60:
        return "[55,60)"
    if p < 0.65:
        return "[60,65)"
    if p < 0.70:
        return "[65,70)"
    return "70_plus"


def _graded_records(frame: pl.DataFrame):
    """Fail on duplicate candidate selection; report invalid row coverage.

    A missing/incorrect payout, invalid probability, malformed odds or result
    cannot quietly contribute to a profitable-looking subset.
    """
    require_columns(frame, REQUIRED, "nfl_historical_edge_archive")
    keys: set[tuple[int, str, str]] = set()
    clean: list[dict[str, object]] = []
    excluded: Counter[str] = Counter()
    for original in frame.iter_rows(named=True):
        market = str(original.get("market_type") or "").lower()
        game = str(original.get("game_id") or "")
        season = _num(original.get("season"))
        week = _num(original.get("week"))
        if market not in MARKETS or not game or season is None or week is None:
            excluded["invalid_game_market_week"] += 1
            continue
        if season != int(season) or week != int(week) or week < 1 or week > 22:
            excluded["invalid_game_market_week"] += 1
            continue
        identity = (int(season), game, market)
        if identity in keys:
            raise DataContractError(
                "duplicate historical best-quote candidate: " + str(identity)
            )
        keys.add(identity)
        model_p = _num(original.get("model_probability"))
        market_p = _num(original.get("no_vig_probability"))
        decimal = _num(original.get("decimal_odds"))
        edge = _num(original.get("probability_edge"))
        ev = _num(original.get("expected_value_per_unit"))
        net = _num(original.get("net_units"))
        result = str(original.get("result") or "").lower().strip()
        if (
            model_p is None or market_p is None or decimal is None
            or edge is None or ev is None or net is None
            or not 0 < model_p < 1 or not 0 < market_p < 1
            or decimal <= 1 or result not in {"win", "loss", "push"}
        ):
            excluded["invalid_probability_price_or_grade"] += 1
            continue
        expected_profit = (
            decimal - 1 if result == "win" else -1.0 if result == "loss" else 0.0
        )
        if abs(net - expected_profit) > 1e-5:
            excluded["net_units_not_reconciled_to_price_and_result"] += 1
            continue
        if abs(edge - (model_p - market_p)) > 1e-5:
            excluded["raw_edge_not_reconciled_to_probabilities"] += 1
            continue
        if abs(ev - (model_p * decimal - 1)) > 1e-5:
            excluded["raw_ev_not_reconciled_to_probabilities_and_price"] += 1
            continue
        if not _yes(original.get("entry_price_verified")) or not _yes(
            original.get("entry_quote_verified")
        ):
            excluded["quote_not_verified_even_as_archive"] += 1
            continue
        if int(season) not in (VALIDATION_SEASON, DIAGNOSTIC_HOLDOUT_SEASON):
            excluded["outside_predeclared_2024_2025_windows"] += 1
            continue
        win = 1.0 if result == "win" else 0.0 if result == "loss" else None
        actual_loss = (
            (model_p - win) ** 2 if win is not None else None
        )
        market_loss = (
            (market_p - win) ** 2 if win is not None else None
        )
        model_clip = max(EPS, min(1 - EPS, model_p))
        market_clip = max(EPS, min(1 - EPS, market_p))
        raw_model_log = (
            -win * math.log(model_clip) - (1 - win) * math.log(1 - model_clip)
            if win is not None else None
        )
        market_log = (
            -win * math.log(market_clip) - (1 - win) * math.log(1 - market_clip)
            if win is not None else None
        )
        row = {
            "game_id": game,
            "season": int(season),
            "week": int(week),
            "market": market,
            "side": str(original.get("side") or ""),
            "book": str(original.get("book") or ""),
            "entry_stage": str(original.get("entry_price_stage") or ""),
            "entry_timestamp_verified": _yes(original.get("entry_timestamp_verified")),
            "model_p": model_p,
            "market_p": market_p,
            "decimal": decimal,
            "model_edge": edge,
            "raw_ev": ev,
            "market_only_ev": market_p * decimal - 1,
            "realized_units": net,
            "result": result,
            "win": win,
            "model_brier_loss": actual_loss,
            "market_brier_loss": market_loss,
            "model_log_loss": raw_model_log,
            "market_log_loss": market_log,
            "raw_positive": edge > 0 and ev > 0,
            "edge_band": _edge_band(edge),
            "model_probability_band": _probability_band(model_p),
        }
        clean.append(row)
    return clean, dict(sorted(excluded.items()))


def _average(records: Sequence[Mapping[str, object]], key: str):
    numbers = [
        float(value) for row in records if (value := _num(row.get(key))) is not None
    ]
    return float(np.mean(numbers)) if numbers else None


def _ci(records: list[dict[str, object]], property_name: str, *, seed: int):
    """Clustered bootstrap, resampling entire season/week blocks not tickets."""
    values = [row for row in records if _num(row.get(property_name)) is not None]
    groups: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in values:
        groups[(row["season"], row["week"])].append(float(row[property_name]))
    if (
        len(values) < MIN_CI_ROWS
        or len({row["game_id"] for row in values}) < MIN_CI_GAMES
        or len(groups) < MIN_CI_WEEKS
    ):
        return None
    blocks = [
        (sum(v), len(v)) for _, v in sorted(groups.items())
    ]
    totals = np.array([pair[0] for pair in blocks])
    sizes = np.array([pair[1] for pair in blocks])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(blocks), size=(BOOTSTRAP_REPS, len(blocks)))
    means = totals[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    return [round(float(q), 6) for q in np.quantile(means, (0.025, 0.975))]


def _summary(
    records: list[dict[str, object]], *, cohort: str, segment: str,
    season: int | None = None, market: str | None = None,
):
    decided = [row for row in records if row["win"] is not None]
    rows = len(records)
    model = _average(decided, "model_p")
    actual = _average(decided, "win")
    market_p = _average(decided, "market_p")
    expected = _average(records, "raw_ev")
    realized = _average(records, "realized_units")
    predicted_edge = _average(decided, "model_edge")
    realized_edge = (
        actual - market_p
        if actual is not None and market_p is not None else None
    )
    retention = (
        realized_edge / predicted_edge
        if realized_edge is not None and predicted_edge is not None
        and predicted_edge > 0 else None
    )
    model_brier = _average(decided, "model_brier_loss")
    market_brier = _average(decided, "market_brier_loss")
    model_log = _average(decided, "model_log_loss")
    market_log = _average(decided, "market_log_loss")
    ci_rows = []
    for row in records:
        copy = dict(row)
        copy["ev_realization_gap"] = row["raw_ev"] - row["realized_units"]
        if row["win"] is not None:
            copy["brier_delta"] = row["model_brier_loss"] - row["market_brier_loss"]
        ci_rows.append(copy)
    results = {
        "cohort": cohort, "segment": segment, "season": season, "market": market,
        "rows": rows,
        "games": len({row["game_id"] for row in records}),
        "week_blocks": len({(row["season"], row["week"]) for row in records}),
        "timestamp_verified_rows": sum(row["entry_timestamp_verified"] for row in records),
        "observed_wins": sum(row["result"] == "win" for row in records),
        "observed_losses": sum(row["result"] == "loss" for row in records),
        "pushes": sum(row["result"] == "push" for row in records),
        "decided_rows": len(decided),
        "mean_model_probability": model,
        "observed_win_rate": actual,
        "model_calibration_gap": (
            model - actual if model is not None and actual is not None else None
        ),
        "mean_no_vig_probability": market_p,
        "realized_minus_market_probability": realized_edge,
        "mean_raw_probability_edge": predicted_edge,
        "realized_edge_retention_ratio": retention,
        "raw_positive_fraction": (
            sum(row["raw_positive"] for row in records) / rows if rows else None
        ),
        "mean_expected_value_per_unit": expected,
        "archive_roi_per_unit": realized,
        "ev_realization_gap": (
            expected - realized if expected is not None and realized is not None else None
        ),
        "mean_market_only_ev": _average(records, "market_only_ev"),
        "model_brier": model_brier,
        "market_brier": market_brier,
        "model_minus_market_brier": (
            model_brier - market_brier
            if model_brier is not None and market_brier is not None else None
        ),
        "model_log_loss": model_log,
        "market_log_loss": market_log,
        "model_minus_market_log_loss": (
            model_log - market_log if model_log is not None and market_log is not None
            else None
        ),
        "ci_95_archive_roi": _ci(
            ci_rows, "realized_units", seed=BOOTSTRAP_SEED
        ),
        "ci_95_model_minus_market_brier": _ci(
            ci_rows, "brier_delta", seed=BOOTSTRAP_SEED + 1
        ),
        "ci_95_ev_realization_gap": _ci(
            ci_rows, "ev_realization_gap", seed=BOOTSTRAP_SEED + 2
        ),
        "evidence_status": (
            "ARCHIVE_DIAGNOSTIC_NOT_EXECUTION_VERIFIED" if rows
            else "NO_VALID_ROWS"
        ),
    }
    return results


def _segments(records: list[dict[str, object]]):
    rows: list[dict[str, object]] = []
    rows.append(_summary(records, cohort="all", segment="all"))
    for season in (VALIDATION_SEASON, DIAGNOSTIC_HOLDOUT_SEASON):
        seasonal = [r for r in records if r["season"] == season]
        rows.append(_summary(seasonal, cohort="season", segment=str(season), season=season))
        for market in MARKETS:
            group = [r for r in seasonal if r["market"] == market]
            rows.append(_summary(
                group, cohort="season_market", segment=f"{season}:{market}",
                season=season, market=market,
            ))
            selected = [r for r in group if r["raw_positive"]]
            rows.append(_summary(
                selected, cohort="raw_positive_season_market",
                segment=f"{season}:{market}", season=season, market=market,
            ))
    for market in MARKETS:
        group = [r for r in records if r["market"] == market]
        rows.append(_summary(group, cohort="market", segment=market, market=market))
        selected = [r for r in group if r["raw_positive"]]
        rows.append(_summary(
            selected, cohort="raw_positive_market", segment=market, market=market,
        ))
        for band in ("nonpositive", "(0,2pp)", "[2,5pp)", "[5,10pp)",
                     "[10,20pp)", "20pp_plus"):
            subset = [r for r in group if r["edge_band"] == band]
            rows.append(_summary(
                subset, cohort="market_raw_edge_band",
                segment=f"{market}:{band}", market=market,
            ))
        for band in ("below_50", "[50,55)", "[55,60)", "[60,65)",
                     "[65,70)", "70_plus"):
            subset = [r for r in group if r["model_probability_band"] == band]
            rows.append(_summary(
                subset, cohort="market_probability_band",
                segment=f"{market}:{band}", market=market,
            ))
    return rows


def build_historical_edge_failure_report(
    archive_bets: pl.DataFrame,
    *,
    archive_backtest: Mapping[str, object] | None = None,
    market_shrinkage: Mapping[str, object] | None = None,
    regime_reliability: Mapping[str, object] | None = None,
):
    """Diagnose failure without fitting/tuning any strategy on the 2025 outcomes."""
    records, excluded = _graded_records(archive_bets)
    segments = _segments(records)
    all_rows = segments[0]
    archive = archive_backtest if isinstance(archive_backtest, Mapping) else {}
    shrinkage = market_shrinkage if isinstance(market_shrinkage, Mapping) else {}
    reliability = regime_reliability if isinstance(regime_reliability, Mapping) else {}
    groups = shrinkage.get("markets")
    statuses = reliability.get("summary")
    market_statuses = statuses.get("market_status") if isinstance(statuses, dict) else {}
    flags = {}
    for market in MARKETS:
        row = next(
            item for item in segments
            if item["cohort"] == "raw_positive_market" and item["market"] == market
        )
        shrink = groups.get(market) if isinstance(groups, dict) else {}
        market_alpha = shrink.get("alpha") if isinstance(shrink, dict) else None
        market_reliability = (
            market_statuses.get(market, "MISSING")
            if isinstance(market_statuses, dict) else "MISSING"
        )
        failures = []
        if row["rows"] == 0:
            failures.append("NO_POSITIVE_EDGE_ARCHIVE_COHORT")
        if row["model_calibration_gap"] is not None and row["model_calibration_gap"] > 0:
            failures.append("MODEL_PROBABILITY_OVERCONFIDENCE")
        if (row["model_minus_market_brier"] is not None
                and row["model_minus_market_brier"] > 0):
            failures.append("MODEL_BRIER_WORSE_THAN_NO_VIG")
        if row["archive_roi_per_unit"] is not None and row["archive_roi_per_unit"] < 0:
            failures.append("POSITIVE_RAW_EV_BUT_NEGATIVE_ARCHIVE_RETURN")
        if (row["ev_realization_gap"] is not None
                and row["ev_realization_gap"] > 0):
            failures.append("MODEL_EV_OVERSTATES_ARCHIVE_REALIZATION")
        if market_alpha is None or float(market_alpha) <= 0:
            failures.append("NO_POSITIVE_HOLDOUT_SELECTED_MODEL_WEIGHT")
        if market_reliability != "RELIABLE":
            failures.append("NFL_REGIME_NOT_RELIABLE")
        if row["timestamp_verified_rows"] != row["rows"]:
            failures.append("NO_COMPLETE_TIMESTAMP_VERIFIED_ENTRY_COHORT")
        flags[market] = {
            "raw_positive_rows": row["rows"],
            "failure_flags": failures,
            "holdout_selected_alpha": market_alpha,
            "regime_status": market_reliability,
        }
    source_check = {
        "archive_report": bool(archive),
        "shrinkage_report": bool(shrinkage),
        "regime_reliability_report": bool(reliability),
        "archive_timestamp_verified": bool(archive.get("timestamped_entry_prices", False)),
        "archive_entry_provenance": archive.get("entry_provenance"),
        "shrinkage_holdout_season": shrinkage.get("holdout_season"),
        "regime_holdout_season": reliability.get("holdout_season"),
    }
    source_consistent = (
        all(source_check[k] for k in (
            "archive_report", "shrinkage_report", "regime_reliability_report"
        ))
        and source_check["archive_entry_provenance"] == "provider_labeled_open"
        and source_check["shrinkage_holdout_season"] == DIAGNOSTIC_HOLDOUT_SEASON
        and source_check["regime_holdout_season"] == DIAGNOSTIC_HOLDOUT_SEASON
    )
    all_verified = bool(records) and all(
        record["entry_timestamp_verified"] for record in records
    )
    if not records:
        state = "NO_VALID_HISTORICAL_EVIDENCE"
    elif not source_consistent or excluded:
        state = "EVIDENCE_INCOMPLETE_OR_INVALID_FAIL_CLOSED"
    elif not all_verified or not source_check["archive_timestamp_verified"]:
        state = "ARCHIVE_DIAGNOSTICS_NO_EXECUTABLE_ENTRY_PROOF"
    else:
        state = "TIMESTAMPED_RESEARCH_NOT_APPROVED"
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": state,
        "research_only": True,
        "betting_authorized": False,
        "approved_units": 0,
        "policy_thresholds_changed": False,
        "fair_score_parameters_changed": False,
        "historical_subgroup_promotion_allowed": False,
        "previously_accessed_2025_outcomes_not_untouched_holdout": True,
        "fixed_splits": {
            "validation_season": VALIDATION_SEASON,
            "diagnostic_holdout_season": DIAGNOSTIC_HOLDOUT_SEASON,
            "sample_policy": (
                "fixed_archived_best_quote_game_market_selection; no after-outcome "
                "filter tuning; 2025 outcomes already inspected in prior research"
            ),
        },
        "data_integrity": {
            "source_rows": archive_bets.height,
            "included_rows": len(records),
            "excluded_rows": sum(excluded.values()),
            "excluded_reasons": excluded,
            "distinct_games": all_rows["games"],
            "distinct_season_week_blocks": all_rows["week_blocks"],
            "entry_timestamp_verified_rows": all_rows["timestamp_verified_rows"],
            "source_check": source_check,
            "source_consistent": source_consistent,
            "provenance_warning": (
                "Provider-labeled ESPN opening and closing fields are archived "
                "stages; they are not independent timestamped book quotes or "
                "evidence that a bet could have been executed at that price."
            ),
        },
        "overall": all_rows,
        "by_market_failure": flags,
        "by_segment": segments,
        "confidence": {
            "method": "paired_season_week_cluster_bootstrap",
            "repetitions": BOOTSTRAP_REPS,
            "seed": BOOTSTRAP_SEED,
            "minimum_rows": MIN_CI_ROWS,
            "minimum_distinct_games": MIN_CI_GAMES,
            "minimum_distinct_season_week_blocks": MIN_CI_WEEKS,
            "ci_missing_when_underpowered": True,
            "interpretation": (
                "Descriptive sampling intervals on already-selected historical "
                "archive rows; not predictive or multiplicity-corrected bounds."
            ),
        },
        "comparison_interpretation": {
            "raw_predicted_ev": "model_probability * archived_decimal_odds - 1",
            "archive_realization": "observed hypothetical unit payoff at archived price",
            "ev_realization_gap": "mean_raw_predicted_ev - archive_realization",
            "market_probability": "same-quote two-way no-vig implied probability",
            "brier_delta": "model Brier minus no-vig-market Brier on same decided rows",
            "brier_delta_positive": "model forecast worse than no-vig market",
            "pushes": "included in simulated ROI, excluded from binary Brier/log loss",
            "raw_positive_filter": "fixed probability_edge>0 and expected_value>0",
            "realized_edge_retention": (
                "(observed non-push win rate - no-vig market probability)"
                " / mean raw positive modeled edge; descriptive only"
            ),
        },
        "limitations": [
            "Not a point-in-time executable-entry backtest; not certified closing-line value.",
            "Prior candidate choice selected highest raw EV in a game/market; "
            "selection bias exists.",
            "All subgroup views are descriptive; no search for best-looking winning regime.",
            "2025 was already inspected in previous research; require new prospective validation.",
            "Cannot determine whether bad odds, misspecified fair scores, injury context or "
            "unmodeled pushes specifically caused a given loss from these archive rows alone.",
            "Multiple groups share games; comparisons are correlated and lack "
            "multiplicity correction.",
            "Positive subgroup returns never bypass reliability, quote provenance "
            "or release gates.",
        ],
    }
    return report


def write_historical_edge_failure(
    report: Mapping[str, object],
    *,
    report_dir: str | Path = "reports",
    outputs_dir: str | Path = "outputs",
    docs_dir: str | Path = "docs",
):
    payload = json.dumps(dict(report), indent=2, sort_keys=True, default=str)
    segments = report.get("by_segment", [])
    for folder in (Path(report_dir), Path(outputs_dir), Path(docs_dir)):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "historical_edge_failure.json").write_text(payload, encoding="utf-8")
        with (folder / "historical_edge_segments.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(TABLE_COLUMNS), extrasaction="ignore")
            writer.writeheader()
            for row in segments:
                writer.writerow({
                    key: json.dumps(row.get(key)) if isinstance(row.get(key), list)
                    else row.get(key)
                    for key in TABLE_COLUMNS
                })
