"""Chronological shrinkage of NFL model edges toward the no-vig market.

The football projection stays independent. This module operates only after a model
probability and a two-way no-vig market probability already exist. It estimates how
much of the model-vs-market log-odds difference survives out of sample.

Alpha meanings:
- 0.0: market-only probability
- 1.0: raw model probability
- between 0 and 1: shrink the model edge toward the market

Alpha is selected on development outcomes only. Validation and holdout seasons are
never allowed to choose alpha. Archive-final fallback prices remain research-only, so
this module cannot promote a betting policy or alter the canonical model.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import polars as pl

from .contracts import require_columns

REQUIRED_COLUMNS = {
    "season",
    "market_type",
    "model_probability",
    "no_vig_probability",
    "decimal_odds",
    "result",
    "net_units",
}
MARKETS = ("moneyline", "spread", "total")
DEFAULT_ALPHA_GRID = tuple(index / 10 for index in range(11))
_EPS = 1e-6


@dataclass(frozen=True)
class ProbabilityMetrics:
    rows: int
    brier: float | None
    log_loss: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BettingMetrics:
    bets: int
    wins: int
    losses: int
    pushes: int
    net_units: float
    roi: float | None
    average_edge: float | None
    average_ev: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _clip_probability(value: float) -> float:
    return min(1.0 - _EPS, max(_EPS, float(value)))


def _logit(value: float) -> float:
    probability = _clip_probability(value)
    return math.log(probability / (1.0 - probability))


def _sigmoid(value: float) -> float:
    if value >= 0:
        exp_negative = math.exp(-value)
        return 1.0 / (1.0 + exp_negative)
    exp_positive = math.exp(value)
    return exp_positive / (1.0 + exp_positive)


def shrink_probability(
    model_probability: float,
    no_vig_probability: float,
    alpha: float,
) -> float:
    """Shrink a downstream model probability toward the no-vig market in log-odds."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    market_logit = _logit(no_vig_probability)
    model_logit = _logit(model_probability)
    return _sigmoid(market_logit + alpha * (model_logit - market_logit))



SIGNED_ALPHA_GRID = (-0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 0.75, 1.0)


def signed_residual_probability(
    model_probability: float,
    no_vig_probability: float,
    alpha: float,
) -> float:
    """Research-only signed model-minus-market log-odds adjustment.

    Negative alpha tests whether the raw NFL model disagreement is directionally
    reversed. The result is NEVER an execution probability or a released bet.
    """
    if not -1.0 <= alpha <= 1.0:
        raise ValueError("signed research alpha must be in [-1, 1]")
    return _sigmoid(
        _logit(no_vig_probability)
        + alpha * (_logit(model_probability) - _logit(no_vig_probability))
    )


def signed_residual_research(
    development: pl.DataFrame,
    validation: pl.DataFrame,
    holdout: pl.DataFrame,
    *,
    minimum_rows: int = 100,
) -> dict[str, object]:
    """Fit the signed adjustment only before 2024; report both later years untouched."""
    grid: dict[str, object] = {}
    options: list[tuple[float, float, float]] = []
    for alpha in SIGNED_ALPHA_GRID:
        actual, raw, market, _ = _probability_arrays(development, alpha=0)
        corrected = np.asarray([
            signed_residual_probability(model, price, alpha)
            for model, price in zip(raw, market, strict=True)
        ])
        metrics = _loss_metrics(actual, corrected)
        grid[f"{alpha:+.2f}"] = metrics.to_dict()
        if metrics.rows >= minimum_rows and metrics.log_loss is not None:
            options.append((metrics.log_loss, metrics.brier, alpha))
    if not options:
        return {"status": "INSUFFICIENT_DEVELOPMENT", "development_grid": grid,
                "betting_policy_change_enabled": False}
    selected = min(options)[2]

    def score(frame: pl.DataFrame) -> dict[str, object]:
        actual, raw, market, _ = _probability_arrays(frame, alpha=0)
        corrected = np.asarray([
            signed_residual_probability(model, price, selected)
            for model, price in zip(raw, market, strict=True)
        ])
        return {
            "research_adjustment": _loss_metrics(actual, corrected).to_dict(),
            "sportsbook_no_vig": _loss_metrics(actual, market).to_dict(),
            "raw_model": _loss_metrics(actual, raw).to_dict(),
        }

    validation_metrics = score(validation)
    holdout_metrics = score(holdout)
    return {
        "status": "RESEARCH_ONLY",
        "selected_alpha": selected,
        "selection": "pre-2024 development log loss only",
        "negative_alpha_means": "raw model disagreement direction reversed",
        "development_grid": grid,
        "validation_2024": validation_metrics,
        "holdout_2025": holdout_metrics,
        "improves_both_years_vs_market": (
            _strictly_better(validation_metrics["research_adjustment"],
                             validation_metrics["sportsbook_no_vig"])
            and _strictly_better(holdout_metrics["research_adjustment"],
                                 holdout_metrics["sportsbook_no_vig"])
        ),
        "betting_policy_change_enabled": False,
        "canonical_market_probability_change_enabled": False,
        "archive_entry_prices_verified": False,
    }


def _binary_outcome(value: object) -> float | None:
    result = str(value or "").strip().lower()
    if result == "win":
        return 1.0
    if result == "loss":
        return 0.0
    return None


def _probability_arrays(
    frame: pl.DataFrame,
    *,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    actual: list[float] = []
    raw: list[float] = []
    market: list[float] = []
    shrunk: list[float] = []
    for row in frame.iter_rows(named=True):
        outcome = _binary_outcome(row.get("result"))
        if outcome is None:
            continue
        try:
            model_probability = float(row["model_probability"])
            no_vig_probability = float(row["no_vig_probability"])
        except (TypeError, ValueError):
            continue
        if not (
            math.isfinite(model_probability)
            and math.isfinite(no_vig_probability)
            and 0.0 <= model_probability <= 1.0
            and 0.0 <= no_vig_probability <= 1.0
        ):
            continue
        actual.append(outcome)
        raw.append(_clip_probability(model_probability))
        market.append(_clip_probability(no_vig_probability))
        shrunk.append(
            shrink_probability(model_probability, no_vig_probability, alpha)
        )
    return (
        np.asarray(actual, dtype=float),
        np.asarray(raw, dtype=float),
        np.asarray(market, dtype=float),
        np.asarray(shrunk, dtype=float),
    )


def _loss_metrics(actual: np.ndarray, probability: np.ndarray) -> ProbabilityMetrics:
    if actual.size == 0:
        return ProbabilityMetrics(0, None, None)
    clipped = np.clip(probability, _EPS, 1.0 - _EPS)
    brier = float(np.mean(np.square(clipped - actual)))
    log_loss = float(
        -np.mean(
            actual * np.log(clipped)
            + (1.0 - actual) * np.log(1.0 - clipped)
        )
    )
    return ProbabilityMetrics(int(actual.size), brier, log_loss)


def probability_comparison(
    frame: pl.DataFrame,
    *,
    alpha: float,
) -> dict[str, object]:
    actual, raw, market, shrunk = _probability_arrays(frame, alpha=alpha)
    return {
        "raw_model": _loss_metrics(actual, raw).to_dict(),
        "market_only": _loss_metrics(actual, market).to_dict(),
        "shrunk_model": _loss_metrics(actual, shrunk).to_dict(),
    }



def calibration_buckets(frame: pl.DataFrame) -> dict[str, object]:
    """Predeclared no-vig probability buckets; describe error, never fit on holdout.

    Each bin reports raw/model and sportsbook forecast calibration on the
    *same decided outcomes*. Pushes and invalid probabilities are excluded.
    Bucketing on market probability, not model prediction/error, avoids
    chasing extreme raw-model disagreements in the held-out year.
    """
    bounds = ((0.0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.0))
    bins: list[list[tuple[float, float, float]]] = [[] for _ in bounds]
    for row in frame.iter_rows(named=True):
        outcome = _binary_outcome(row.get("result"))
        if outcome is None:
            continue
        try:
            model = float(row["model_probability"])
            market = float(row["no_vig_probability"])
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(model) and math.isfinite(market)):
            continue
        if not (0 <= model <= 1 and 0 <= market <= 1):
            continue
        for index, (lower, upper) in enumerate(bounds):
            if lower <= market < upper or (index == len(bounds) - 1 and market == 1):
                bins[index].append((outcome, model, market))
                break

    output: dict[str, object] = {}
    for (lower, upper), rows in zip(bounds, bins, strict=True):
        key = f"{lower:.1f}-{upper:.1f}"
        if not rows:
            output[key] = {"rows": 0, "status": "EMPTY"}
            continue
        actual = np.array([value[0] for value in rows])
        raw = np.array([value[1] for value in rows])
        market = np.array([value[2] for value in rows])
        raw_brier = float(np.mean(np.square(actual - raw)))
        market_brier = float(np.mean(np.square(actual - market)))
        output[key] = {
            "rows": len(rows),
            "status": "DESCRIPTIVE_ONLY",
            "observed_win_rate": float(np.mean(actual)),
            "raw_mean_probability": float(np.mean(raw)),
            "market_mean_probability": float(np.mean(market)),
            "raw_calibration_error": float(np.mean(raw - actual)),
            "market_calibration_error": float(np.mean(market - actual)),
            "raw_brier": raw_brier,
            "market_brier": market_brier,
            "raw_brier_minus_market": raw_brier - market_brier,
        }
    return output


def fit_alpha(
    development: pl.DataFrame,
    *,
    alpha_grid: tuple[float, ...] = DEFAULT_ALPHA_GRID,
    minimum_rows: int = 100,
) -> tuple[float | None, dict[str, object]]:
    """Choose alpha on development log loss only, with Brier as a tie-breaker."""

    candidates: list[tuple[float, float, float, int]] = []
    diagnostics: dict[str, object] = {}
    for alpha in alpha_grid:
        actual, _, _, shrunk = _probability_arrays(development, alpha=float(alpha))
        metrics = _loss_metrics(actual, shrunk)
        diagnostics[f"{float(alpha):.2f}"] = metrics.to_dict()
        if (
            metrics.rows >= minimum_rows
            and metrics.log_loss is not None
            and metrics.brier is not None
        ):
            candidates.append(
                (
                    float(metrics.log_loss),
                    float(metrics.brier),
                    float(alpha),
                    int(metrics.rows),
                )
            )
    if not candidates:
        return None, diagnostics
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return candidates[0][2], diagnostics


def _betting_metrics(
    frame: pl.DataFrame,
    *,
    alpha: float,
    min_edge: float = 0.0,
    min_ev: float = 0.0,
    min_probability: float = 0.0,
) -> BettingMetrics:
    selected: list[tuple[str, float, float, float]] = []
    for row in frame.iter_rows(named=True):
        try:
            model_probability = float(row["model_probability"])
            no_vig_probability = float(row["no_vig_probability"])
            decimal_odds = float(row["decimal_odds"])
            net_units = float(row["net_units"])
        except (TypeError, ValueError):
            continue
        if not all(
            math.isfinite(value)
            for value in (
                model_probability,
                no_vig_probability,
                decimal_odds,
                net_units,
            )
        ):
            continue
        probability = shrink_probability(
            model_probability,
            no_vig_probability,
            alpha,
        )
        edge = probability - no_vig_probability
        expected_value = probability * decimal_odds - 1.0
        if (
            edge < min_edge
            or expected_value < min_ev
            or probability < min_probability
        ):
            continue
        selected.append(
            (
                str(row.get("result") or "").strip().lower(),
                net_units,
                edge,
                expected_value,
            )
        )
    if not selected:
        return BettingMetrics(0, 0, 0, 0, 0.0, None, None, None)

    bets = len(selected)
    net_units = float(sum(row[1] for row in selected))
    return BettingMetrics(
        bets=bets,
        wins=sum(row[0] == "win" for row in selected),
        losses=sum(row[0] == "loss" for row in selected),
        pushes=sum(row[0] == "push" for row in selected),
        net_units=net_units,
        roi=net_units / bets,
        average_edge=float(np.mean([row[2] for row in selected])),
        average_ev=float(np.mean([row[3] for row in selected])),
    )


def _not_worse(
    candidate: dict[str, object],
    baseline: dict[str, object],
) -> bool:
    candidate_brier = candidate.get("brier")
    baseline_brier = baseline.get("brier")
    candidate_log = candidate.get("log_loss")
    baseline_log = baseline.get("log_loss")
    return (
        candidate_brier is not None
        and baseline_brier is not None
        and candidate_log is not None
        and baseline_log is not None
        and float(candidate_brier) <= float(baseline_brier) + 1e-12
        and float(candidate_log) <= float(baseline_log) + 1e-12
    )


def _strictly_better(
    candidate: dict[str, object],
    baseline: dict[str, object],
) -> bool:
    if not _not_worse(candidate, baseline):
        return False
    return (
        float(candidate["brier"]) < float(baseline["brier"]) - 1e-12
        or float(candidate["log_loss"]) < float(baseline["log_loss"]) - 1e-12
    )




def archived_cohort_economics(frame: pl.DataFrame) -> dict[str, object]:
    """Descriptive per-bet economic outcomes with explicit quote provenance.

    Archives have no verified entry timestamps. A positive point estimate
    or approximate interval is never proof of executable betting value.
    """
    outcomes: list[float] = []
    clv: list[float] = []
    verified = 0
    for row in frame.iter_rows(named=True):
        if str(row.get("result") or "").lower() not in {"win", "loss", "push"}:
            continue
        try:
            net = float(row["net_units"])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(net):
            continue
        outcomes.append(net)
        if row.get("entry_price_verified") is True:
            verified += 1
        value = row.get("clv_proxy")
        try:
            clv_value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(clv_value):
            clv.append(clv_value)
    n = len(outcomes)
    if not n:
        return {"status": "NO_GRADED_OUTCOMES", "bets": 0}
    values = np.asarray(outcomes, dtype=float)
    average = float(values.mean())
    standard_error = (
        float(values.std(ddof=1) / math.sqrt(n)) if n > 1 else None
    )
    return {
        "status": "ARCHIVED_PRICE_RESEARCH_ONLY",
        "bets": n,
        "net_units": float(values.sum()),
        "roi": average,
        "approximate_roi_interval_95": (
            [average - 1.96 * standard_error, average + 1.96 * standard_error]
            if standard_error is not None else None
        ),
        "interval_method": "normal approximation; exploratory, not multiple-testing adjusted",
        "positive_roi_lower_bound": (
            average - 1.96 * standard_error > 0
            if standard_error is not None else False
        ),
        "verified_entry_prices": verified,
        "verified_entry_fraction": verified / n,
        "clv_proxy_samples": len(clv),
        "mean_clv_proxy": float(np.mean(clv)) if clv else None,
        "execution_clv_verified": False,
        "staking_authorized": False,
    }


def fixed_cohort_calibration(
    frame: pl.DataFrame,
    *,
    validation_season: int,
    holdout_season: int,
    min_development: int = 100,
    min_evaluation: int = 30,
) -> dict[str, object]:
    """Predeclared one-dimensional NFL cohorts, never optimized on holdout.

    This audit requires no new features beyond recorded pregame quote data.
    Its cohorts overlap, are not independent bets, and cannot authorize staking.
    Missing context/margin projections are explicitly not imputed.
    """
    required = {"side", "no_vig_probability", "line", "week"}
    if not required.issubset(frame.columns):
        return {"status": "MISSING_PREDECLARED_FEATURES",
                "missing": sorted(required - set(frame.columns))}
    candidates: dict[str, pl.Expr] = {
        "side:home": pl.col("side") == "home",
        "side:away": pl.col("side") == "away",
        "side:over": pl.col("side") == "over",
        "side:under": pl.col("side") == "under",
        "moneyline:favorite": (
            (pl.col("market_type") == "moneyline")
            & (pl.col("no_vig_probability") > 0.5)
        ),
        "moneyline:underdog": (
            (pl.col("market_type") == "moneyline")
            & (pl.col("no_vig_probability") < 0.5)
        ),
        "spread:favorite": (
            (pl.col("market_type") == "spread") & (pl.col("line") < 0)
        ),
        "spread:underdog": (
            (pl.col("market_type") == "spread") & (pl.col("line") > 0)
        ),
        "week:early": pl.col("week") <= 4,
        "week:middle": (pl.col("week") >= 5) & (pl.col("week") <= 12),
        "week:late": pl.col("week") >= 13,
    }
    has_pregame_margin = "projected_home_margin" in frame.columns
    if has_pregame_margin:
        # NFL-native margin environments, based ONLY on frozen pregame scores.
        magnitude = pl.col("projected_home_margin").abs()
        candidates.update({
            "projected_margin:close_lt3": magnitude < 3,
            "projected_margin:moderate_3_to7": (magnitude >= 3) & (magnitude < 7),
            "projected_margin:large_ge7": magnitude >= 7,
            "moneyline:favorite_large_margin": (
                (pl.col("market_type") == "moneyline")
                & (pl.col("no_vig_probability") > 0.5)
                & (magnitude >= 7)
            ),
        })
    output: dict[str, object] = {}
    for label, condition in candidates.items():
        cohort = frame.filter(condition.fill_null(False))
        development = cohort.filter(pl.col("season") < validation_season)
        validation = cohort.filter(pl.col("season") == validation_season)
        holdout = cohort.filter(pl.col("season") == holdout_season)
        if development.height < min_development or (
            validation.height < min_evaluation
            or holdout.height < min_evaluation
        ):
            output[label] = {
                "status": "INSUFFICIENT_PREDECLARED_SAMPLE",
                "development_rows": development.height,
                "validation_rows": validation.height,
                "holdout_rows": holdout.height,
            }
            continue
        research = signed_residual_research(
            development, validation, holdout, minimum_rows=min_development
        )
        # Screening threshold is descriptive, not a statistical discovery or
        # an operational promotion rule. Both untouched years must clear it.
        effect_by_year: dict[str, object] = {}
        for year_key in ("validation_2024", "holdout_2025"):
            year_data = research.get(year_key, {})
            adjusted = year_data.get("research_adjustment", {})
            baseline = year_data.get("sportsbook_no_vig", {})
            brier_gain = (
                float(baseline["brier"]) - float(adjusted["brier"])
                if baseline.get("brier") is not None
                and adjusted.get("brier") is not None else None
            )
            log_loss_gain = (
                float(baseline["log_loss"]) - float(adjusted["log_loss"])
                if baseline.get("log_loss") is not None
                and adjusted.get("log_loss") is not None else None
            )
            effect_by_year[year_key] = {
                "brier_gain": brier_gain,
                "log_loss_gain": log_loss_gain,
                "passes_minimum_effect": (
                    brier_gain is not None and brier_gain >= 0.002
                    and log_loss_gain is not None and log_loss_gain >= 0.004
                ),
            }
        material = all(
            bool(year["passes_minimum_effect"])
            for year in effect_by_year.values()
        )
        output[label] = {
            "status": "DESCRIPTIVE_RESEARCH_ONLY",
            "archived_economic_evidence": {
                "validation_2024": archived_cohort_economics(validation),
                "holdout_2025": archived_cohort_economics(holdout),
            },
            "economic_evidence_eligible_for_staking": False,
            "minimum_effect_in_both_years": material,
            "effect_size_thresholds": {"brier": 0.002, "log_loss": 0.004},
            "effect_size_by_year": effect_by_year,
            "development_rows": development.height,
            "validation_rows": validation.height,
            "holdout_rows": holdout.height,
            "signed_residual": research,
            "not_independent_of_other_cohorts": True,
            "betting_policy_change_enabled": False,
        }
    return {
        "status": "RESEARCH_ONLY",
        "dimensions": "fixed side / quoted role / season phase",
        "projected_margin_cohorts": (
            "PREGAME_PROJECTIONS_JOINED" if has_pregame_margin
            else "UNAVAILABLE_IN_FREE_MARKET_BETS"
        ),
        "multiple_testing": "overlapping descriptive cohorts; no promotion",
        "cohorts": output,
        "betting_policy_change_enabled": False,
    }


def evaluate_market_edge_shrinkage(
    bets: pl.DataFrame,
    *,
    validation_season: int = 2024,
    holdout_season: int = 2025,
    alpha_grid: tuple[float, ...] = DEFAULT_ALPHA_GRID,
    minimum_development_rows: int = 100,
) -> dict[str, object]:
    """Evaluate fixed downstream edge shrinkage with untouched chronological holdout."""

    require_columns(bets, REQUIRED_COLUMNS, "market_edge_shrinkage_bets")
    if validation_season >= holdout_season:
        raise ValueError("validation_season must be earlier than holdout_season")

    markets: dict[str, object] = {}
    validated_markets = 0
    incremental_markets = 0

    for market_type in MARKETS:
        market = bets.filter(pl.col("market_type") == market_type)
        development = market.filter(pl.col("season") < validation_season)
        validation = market.filter(pl.col("season") == validation_season)
        holdout = market.filter(pl.col("season") == holdout_season)

        alpha, development_grid = fit_alpha(
            development,
            alpha_grid=alpha_grid,
            minimum_rows=minimum_development_rows,
        )
        if alpha is None or validation.is_empty() or holdout.is_empty():
            markets[market_type] = {
                "status": "INSUFFICIENT_DATA",
                "alpha": alpha,
                "development_rows": development.height,
                "validation_rows": validation.height,
                "holdout_rows": holdout.height,
                "development_grid": development_grid,
                "canonical_change_enabled": False,
            }
            continue

        validation_probability = probability_comparison(validation, alpha=alpha)
        holdout_probability = probability_comparison(holdout, alpha=alpha)

        validation_pass = _not_worse(
            validation_probability["shrunk_model"],
            validation_probability["raw_model"],
        )
        holdout_pass = _strictly_better(
            holdout_probability["shrunk_model"],
            holdout_probability["raw_model"],
        )
        probability_validated = validation_pass and holdout_pass

        market_incremental_value = (
            alpha > 0.0
            and _not_worse(
                validation_probability["shrunk_model"],
                validation_probability["market_only"],
            )
            and _strictly_better(
                holdout_probability["shrunk_model"],
                holdout_probability["market_only"],
            )
        )

        if probability_validated:
            validated_markets += 1
        if market_incremental_value:
            incremental_markets += 1

        if probability_validated and alpha == 0.0:
            market_status = "MARKET_ONLY_PREFERRED"
        elif probability_validated:
            market_status = "VALIDATED_SHRINKAGE"
        else:
            market_status = "NO_VALIDATED_SHRINKAGE"

        markets[market_type] = {
            "status": market_status,
            "alpha": alpha,
            "development_rows": development.height,
            "validation_rows": validation.height,
            "holdout_rows": holdout.height,
            "development_grid": development_grid,
            "validation_probability": validation_probability,
            "holdout_probability": holdout_probability,
            "signed_residual_research": signed_residual_research(
                development, validation, holdout,
                minimum_rows=minimum_development_rows,
            ),
            "calibration_diagnostics": {
                "method": "fixed no-vig probability bins; descriptive only",
                "validation_2024": calibration_buckets(validation),
                "holdout_2025": calibration_buckets(holdout),
            },
            "validation_probability_pass": validation_pass,
            "holdout_probability_pass": holdout_pass,
            "probability_validated": probability_validated,
            "incremental_model_value_vs_market": market_incremental_value,
            "validation_betting_raw_positive": _betting_metrics(
                validation,
                alpha=1.0,
            ).to_dict(),
            "validation_betting_shrunk_positive": _betting_metrics(
                validation,
                alpha=alpha,
            ).to_dict(),
            "holdout_betting_raw_positive": _betting_metrics(
                holdout,
                alpha=1.0,
            ).to_dict(),
            "holdout_betting_shrunk_positive": _betting_metrics(
                holdout,
                alpha=alpha,
            ).to_dict(),
            "canonical_change_enabled": False,
        }

    if incremental_markets == 0:
        research_conclusion = "NO_INCREMENTAL_MODEL_VALUE"
    elif incremental_markets == len(MARKETS):
        research_conclusion = "BROAD_INCREMENTAL_MODEL_VALUE"
    else:
        research_conclusion = "PARTIAL_INCREMENTAL_MODEL_VALUE"

    return {
        "version": 1,
        "status": "RESEARCH_ONLY",
        "research_conclusion": research_conclusion,
        "validation_season": validation_season,
        "holdout_season": holdout_season,
        "alpha_grid": list(alpha_grid),
        "selection_objective": "development log loss; Brier tie-breaker",
        "validated_shrinkage_markets": validated_markets,
        "incremental_model_value_markets": incremental_markets,
        "markets": markets,
        "fixed_cohort_research": fixed_cohort_calibration(
            bets,
            validation_season=validation_season,
            holdout_season=holdout_season,
            min_development=minimum_development_rows,
        ),
        "canonical_market_probability_change_enabled": False,
        "betting_policy_change_enabled": False,
        "forward_shrinkage_shadow_recommended": False,
        "archive_entry_prices_verified": False,
        "meaning": (
            "Downstream research only. Alpha is selected before validation/holdout. "
            "Sportsbook probabilities never enter the fair-score model, and unverified "
            "archive prices cannot promote a betting policy."
        ),
    }
