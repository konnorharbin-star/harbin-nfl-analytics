"""Operational NFL probability selection with chronological validation.

This module is the only live bridge between historical calibration evidence and the
current market-comparison layer. Candidate probability changes are selected using a
validation season, checked on a later untouched season, and only then refit on all
information available before the current slate.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Protocol

import polars as pl

from .contracts import DataContractError, require_columns
from .probability import (
    NORMAL_DF,
    ConditionalStudentTScoreDistribution,
    GaussianScoreDistribution,
    evaluate_conditional_probability_holdout,
)
from .win_probability import (
    LogisticWinModel,
    evaluate_win_probability_holdout,
    score_gaussian,
)


class ScoreDistribution(Protocol):
    def home_win_probability(
        self,
        projected_home_margin: float,
        projected_total: float | None = None,
    ) -> float: ...

    def home_cover_probability(
        self,
        projected_home_margin: float,
        home_spread: float,
        projected_total: float | None = None,
    ) -> float: ...

    def over_probability(
        self,
        projected_total: float,
        total_line: float,
        projected_home_margin: float | None = None,
    ) -> float: ...

    def margin_sigma_for(
        self,
        projected_home_margin: float,
        projected_total: float | None = None,
    ) -> float: ...

    def total_sigma_for(
        self,
        projected_total: float,
        projected_home_margin: float | None = None,
    ) -> float: ...


class OperationalProbabilityDistribution:
    """Score distribution plus an independently validated moneyline calibrator."""

    def __init__(
        self,
        score: GaussianScoreDistribution,
        *,
        win_model: LogisticWinModel | None = None,
    ) -> None:
        self.score = score
        self.win_model = win_model

    def home_win_probability(
        self,
        projected_home_margin: float,
        projected_total: float | None = None,
    ) -> float:
        if self.win_model is not None:
            return self.win_model.predict_probability(projected_home_margin)
        return self.score.home_win_probability(
            projected_home_margin,
            projected_total,
        )

    def home_cover_probability(
        self,
        projected_home_margin: float,
        home_spread: float,
        projected_total: float | None = None,
    ) -> float:
        return self.score.home_cover_probability(
            projected_home_margin,
            home_spread,
            projected_total,
        )

    def over_probability(
        self,
        projected_total: float,
        total_line: float,
        projected_home_margin: float | None = None,
    ) -> float:
        return self.score.over_probability(
            projected_total,
            total_line,
            projected_home_margin,
        )

    def margin_sigma_for(
        self,
        projected_home_margin: float,
        projected_total: float | None = None,
    ) -> float:
        return self.score.margin_sigma_for(
            projected_home_margin,
            projected_total,
        )

    def total_sigma_for(
        self,
        projected_total: float,
        projected_home_margin: float | None = None,
    ) -> float:
        return self.score.total_sigma_for(
            projected_total,
            projected_home_margin,
        )


def _fallback(
    historical: pl.DataFrame,
    *,
    reason: str,
) -> tuple[OperationalProbabilityDistribution, dict[str, object]]:
    score = GaussianScoreDistribution().fit(historical)
    distribution = OperationalProbabilityDistribution(score)
    return distribution, {
        "status": "FALLBACK",
        "model_family": "gaussian_unconditional",
        "conditional_uncertainty_enabled": False,
        "moneyline_logistic_enabled": False,
        "reliability_ready": False,
        "reason": reason,
        "training_games": historical.height,
        "home_win_brier": None,
        "home_win_log_loss": None,
        "home_win_ece": None,
        "mid_confidence_games": 0,
        "mid_confidence_mean_probability": None,
        "mid_confidence_actual_rate": None,
        "mid_confidence_gap": None,
        "margin_50_coverage": None,
        "margin_80_coverage": None,
        "total_50_coverage": None,
        "total_80_coverage": None,
    }


def build_operational_probability_distribution(
    historical: pl.DataFrame,
    *,
    current_season: int,
) -> tuple[OperationalProbabilityDistribution, dict[str, object]]:
    """Fit the current probability model only after untouched holdout validation."""

    require_columns(
        historical,
        {
            "projected_home_margin",
            "projected_total",
            "actual_home_margin",
            "actual_total",
        },
        "operational_probability_history",
    )
    if historical.height < 64:
        raise DataContractError(
            "operational probability model requires at least 64 prior games"
        )
    if "season" not in historical.columns:
        return _fallback(
            historical,
            reason=(
                "historical season labels are unavailable; chronological "
                "probability promotion is disabled"
            ),
        )

    validation_season = current_season - 2
    holdout_season = current_season - 1
    full_past = historical.filter(pl.col("season") < current_season)

    try:
        conditional = evaluate_conditional_probability_holdout(
            full_past,
            validation_season=validation_season,
            holdout_season=holdout_season,
        )
        win = evaluate_win_probability_holdout(
            full_past,
            validation_season=validation_season,
            holdout_season=holdout_season,
        )
    except Exception as exc:
        return _fallback(
            historical,
            reason=f"{type(exc).__name__}: {exc}",
        )

    margin_use_candidate = conditional.margin_candidate_pass
    total_use_candidate = conditional.total_candidate_pass

    score_kwargs = {
        "margin_scale": (
            conditional.margin_scale
            if margin_use_candidate
            else conditional.baseline_margin_scale
        ),
        "total_scale": (
            conditional.total_scale
            if total_use_candidate
            else conditional.baseline_total_scale
        ),
        "margin_df": (
            conditional.margin_df if margin_use_candidate else NORMAL_DF
        ),
        "total_df": (
            conditional.total_df if total_use_candidate else NORMAL_DF
        ),
        "margin_strength": (
            conditional.margin_strength if margin_use_candidate else 0.0
        ),
        "total_strength": (
            conditional.total_strength if total_use_candidate else 0.0
        ),
    }
    score = ConditionalStudentTScoreDistribution(**score_kwargs).fit(
        historical
    )

    holdout_training = full_past.filter(
        pl.col("season") < holdout_season
    )
    holdout = full_past.filter(pl.col("season") == holdout_season)
    holdout_score = ConditionalStudentTScoreDistribution(
        **score_kwargs
    ).fit(holdout_training)
    selected_score_win_metrics = score_gaussian(
        holdout,
        holdout_score,
    )

    win_model: LogisticWinModel | None = None
    if win.candidate_pass:
        win_model = LogisticWinModel(alpha=win.alpha).fit(historical)

    distribution = OperationalProbabilityDistribution(
        score,
        win_model=win_model,
    )

    selected_score_metrics = conditional.holdout_candidate
    baseline_score_metrics = conditional.holdout_baseline
    margin_metrics = (
        selected_score_metrics if margin_use_candidate else baseline_score_metrics
    )
    total_metrics = (
        selected_score_metrics if total_use_candidate else baseline_score_metrics
    )
    selected_win_metrics = (
        win.logistic if win.candidate_pass else selected_score_win_metrics
    )

    mid_gap = selected_win_metrics.mid_confidence_gap
    mid_reliable = (
        selected_win_metrics.mid_confidence_games < 30
        or mid_gap is None
        or mid_gap <= 0.08
    )
    reliability_ready = (
        selected_win_metrics.brier <= 0.25
        and selected_win_metrics.ece <= 0.08
        and mid_reliable
        and 0.70 <= margin_metrics.margin_80_coverage <= 0.90
        and 0.70 <= total_metrics.total_80_coverage <= 0.90
    )

    families: list[str] = []
    if margin_use_candidate or total_use_candidate:
        families.append("conditional_student_t")
    else:
        families.append("gaussian")
    if win.candidate_pass:
        families.append("logistic_moneyline")

    return distribution, {
        "status": "READY",
        "model_family": "+".join(families),
        "validation_season": validation_season,
        "holdout_season": holdout_season,
        "training_games": historical.height,
        "holdout_games": conditional.holdout_games,
        "conditional_uncertainty_enabled": (
            margin_use_candidate or total_use_candidate
        ),
        "margin_conditional_enabled": margin_use_candidate,
        "total_conditional_enabled": total_use_candidate,
        "moneyline_logistic_enabled": win.candidate_pass,
        "margin_scale": (
            conditional.margin_scale
            if margin_use_candidate
            else conditional.baseline_margin_scale
        ),
        "total_scale": (
            conditional.total_scale
            if total_use_candidate
            else conditional.baseline_total_scale
        ),
        "margin_df": conditional.margin_df if margin_use_candidate else NORMAL_DF,
        "total_df": conditional.total_df if total_use_candidate else NORMAL_DF,
        "margin_conditional_strength": (
            conditional.margin_strength if margin_use_candidate else 0.0
        ),
        "total_conditional_strength": (
            conditional.total_strength if total_use_candidate else 0.0
        ),
        "margin_nll": margin_metrics.margin_nll,
        "total_nll": total_metrics.total_nll,
        "margin_nll_improvement": conditional.margin_nll_improvement,
        "total_nll_improvement": conditional.total_nll_improvement,
        "margin_50_coverage": margin_metrics.margin_50_coverage,
        "margin_80_coverage": margin_metrics.margin_80_coverage,
        "total_50_coverage": total_metrics.total_50_coverage,
        "total_80_coverage": total_metrics.total_80_coverage,
        "home_win_brier": selected_win_metrics.brier,
        "home_win_log_loss": selected_win_metrics.log_loss,
        "home_win_ece": selected_win_metrics.ece,
        "mid_confidence_games": selected_win_metrics.mid_confidence_games,
        "mid_confidence_mean_probability": (
            selected_win_metrics.mid_confidence_mean_probability
        ),
        "mid_confidence_actual_rate": (
            selected_win_metrics.mid_confidence_actual_rate
        ),
        "mid_confidence_gap": selected_win_metrics.mid_confidence_gap,
        "reliability_ready": reliability_ready,
        "conditional_holdout": conditional.to_dict(),
        "win_holdout": asdict(win),
    }


def apply_probability_reliability_veto(
    candidates: pl.DataFrame,
    probability_meta: dict[str, object],
) -> pl.DataFrame:
    """Suppress all betting when probability reliability has not cleared validation."""

    if candidates.is_empty():
        return candidates
    ready = bool(probability_meta.get("reliability_ready", False))
    reason = ""
    if not ready:
        reason = (
            "probability calibration/reliability gate is not ready: "
            + str(
                probability_meta.get("reason")
                or probability_meta.get("model_family")
                or "unknown"
            )
        )

    rows: list[dict[str, object]] = []
    for row in candidates.iter_rows(named=True):
        item = dict(row)
        item["probability_reliability_veto"] = not ready
        item["probability_reliability_veto_reason"] = reason
        item["probability_model_family"] = probability_meta.get("model_family")
        item["probability_reliability_ready"] = ready
        if not ready:
            for field in (
                "quant_signal",
                "production_signal",
                "research_signal",
                "portfolio_signal",
            ):
                if field in item:
                    item[field] = "PASS"
            for field in ("stake_units", "research_stake_units"):
                if field in item:
                    item[field] = 0.0
        rows.append(item)
    return pl.DataFrame(rows)
