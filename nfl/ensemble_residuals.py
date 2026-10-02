"""NCAA-style linear + nonlinear residual research for the NFL fair-score model.

The architecture mirrors the CFB model's residual-learning shape while keeping every
weight and selection decision NFL-native. Sportsbook information is never accepted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .contracts import DataContractError, require_columns
from .residuals import RESIDUAL_FEATURES

ENSEMBLE_FEATURES = (
    *RESIDUAL_FEATURES,
    "baseline_home_margin",
    "baseline_total",
    "home_off_plays",
    "away_off_plays",
    "home_def_plays",
    "away_def_plays",
)

RIDGE_ALPHA_GRID = (1.0, 10.0, 100.0)
RIDGE_SHARE_GRID = (0.0, 0.25, 0.50, 0.75, 1.0)
RESIDUAL_WEIGHT_GRID = (0.0, 0.25, 0.50, 0.75, 1.0)


@dataclass(frozen=True)
class EnsembleSpec:
    ridge_alpha: float
    ridge_share: float
    residual_weight: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class ErrorMetrics:
    mae: float
    rmse: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class EnsembleFold:
    season: int
    training_games: int
    core_games: int
    tune_games: int
    evaluation_games: int
    selected_spec: EnsembleSpec | None
    tune_baseline: ErrorMetrics
    tune_adjusted: ErrorMetrics
    evaluation_baseline: ErrorMetrics
    evaluation_adjusted: ErrorMetrics
    passed: bool

    def to_dict(self) -> dict[str, object]:
        out = asdict(self)
        out["selected_spec"] = (
            None if self.selected_spec is None else self.selected_spec.to_dict()
        )
        out["tune_baseline"] = self.tune_baseline.to_dict()
        out["tune_adjusted"] = self.tune_adjusted.to_dict()
        out["evaluation_baseline"] = self.evaluation_baseline.to_dict()
        out["evaluation_adjusted"] = self.evaluation_adjusted.to_dict()
        return out


@dataclass(frozen=True)
class RollingEnsembleResult:
    target: str
    folds: tuple[EnsembleFold, ...]
    games: int
    baseline_mae: float
    adjusted_mae: float
    baseline_rmse: float
    adjusted_rmse: float
    mae_improvement: float
    rmse_improvement: float
    positive_folds: int
    eligible: bool
    canonical_score_adjustment_enabled: bool = False
    promotion_eligible: bool = False

    def to_dict(self) -> dict[str, object]:
        out = asdict(self)
        out["folds"] = [fold.to_dict() for fold in self.folds]
        return out


def _target_columns(target: str) -> tuple[str, str, str]:
    if target == "margin":
        return "actual_home_margin", "baseline_home_margin", "margin_residual"
    if target == "total":
        return "actual_total", "baseline_total", "total_residual"
    raise ValueError("target must be 'margin' or 'total'")


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> ErrorMetrics:
    error = predicted - actual
    return ErrorMetrics(
        mae=float(np.mean(np.abs(error))),
        rmse=float(sqrt(float(np.mean(np.square(error))))),
    )


def _matrix(frame: pl.DataFrame) -> np.ndarray:
    require_columns(frame, set(ENSEMBLE_FEATURES), "ensemble_features")
    matrix = frame.select(list(ENSEMBLE_FEATURES)).to_numpy().astype(float)
    if not np.isfinite(matrix).all():
        raise DataContractError("ensemble features contain null/non-finite values")
    return matrix


def _whole_week_split(
    training: pl.DataFrame,
    *,
    tune_fraction: float = 0.25,
    min_core_games: int = 100,
    min_tune_games: int = 30,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split training chronologically without splitting a season/week block."""

    require_columns(training, {"season", "week"}, "ensemble_training")
    if not 0.10 <= tune_fraction <= 0.40:
        raise ValueError("tune_fraction must be between 0.10 and 0.40")

    groups = (
        training.select(["season", "week"])
        .unique()
        .sort(["season", "week"])
        .iter_rows()
    )
    keys = [(int(season), int(week)) for season, week in groups]
    if len(keys) < 4:
        raise DataContractError("ensemble training requires at least four week blocks")

    tune_groups = max(2, int(round(len(keys) * tune_fraction)))
    tune_groups = min(tune_groups, len(keys) - 2)
    tune_keys = set(keys[-tune_groups:])
    key_expr = pl.struct(["season", "week"]).map_elements(
        lambda row: (int(row["season"]), int(row["week"])) in tune_keys,
        return_dtype=pl.Boolean,
    )
    tune = training.filter(key_expr)
    core = training.filter(~key_expr)

    if core.height < min_core_games or tune.height < min_tune_games:
        raise DataContractError(
            "chronological ensemble split is too small "
            f"(core={core.height}, tune={tune.height})"
        )
    return core, tune


def _fit_components(
    frame: pl.DataFrame,
    *,
    residual_column: str,
    ridge_alpha: float,
) -> tuple[Pipeline, HistGradientBoostingRegressor]:
    require_columns(frame, {residual_column}, "ensemble_training")
    x = _matrix(frame)
    y = np.asarray(frame.get_column(residual_column), dtype=float)
    if not np.isfinite(y).all():
        raise DataContractError("ensemble residual target contains null/non-finite values")

    ridge = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=float(ridge_alpha))),
        ]
    )
    boost = HistGradientBoostingRegressor(
        max_depth=3,
        learning_rate=0.04,
        max_iter=280,
        l2_regularization=9.0,
        min_samples_leaf=28,
        random_state=26,
    )
    ridge.fit(x, y)
    boost.fit(x, y)
    return ridge, boost


def _component_predictions(
    pair: tuple[Pipeline, HistGradientBoostingRegressor],
    frame: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    x = _matrix(frame)
    ridge, boost = pair
    return np.asarray(ridge.predict(x), dtype=float), np.asarray(boost.predict(x), dtype=float)


def _blend_residual(
    ridge_pred: np.ndarray,
    boost_pred: np.ndarray,
    *,
    ridge_share: float,
) -> np.ndarray:
    return float(ridge_share) * ridge_pred + (1.0 - float(ridge_share)) * boost_pred


def _strictly_better(candidate: ErrorMetrics, baseline: ErrorMetrics) -> bool:
    return candidate.mae < baseline.mae and candidate.rmse < baseline.rmse


def _selection_score(candidate: ErrorMetrics, baseline: ErrorMetrics) -> float:
    return candidate.mae / baseline.mae + candidate.rmse / baseline.rmse


def select_ensemble_spec(
    core: pl.DataFrame,
    tune: pl.DataFrame,
    *,
    target: str,
    ridge_alpha_grid: tuple[float, ...] = RIDGE_ALPHA_GRID,
    ridge_share_grid: tuple[float, ...] = RIDGE_SHARE_GRID,
    residual_weight_grid: tuple[float, ...] = RESIDUAL_WEIGHT_GRID,
) -> tuple[EnsembleSpec | None, ErrorMetrics, ErrorMetrics]:
    """Choose an NFL-only ensemble specification on a strictly earlier tune block."""

    actual_col, baseline_col, residual_col = _target_columns(target)
    require_columns(tune, {actual_col, baseline_col}, "ensemble_tune")
    actual = np.asarray(tune.get_column(actual_col), dtype=float)
    baseline = np.asarray(tune.get_column(baseline_col), dtype=float)
    baseline_metrics = _metrics(actual, baseline)

    eligible: list[tuple[float, float, float, float, EnsembleSpec, ErrorMetrics]] = []
    for alpha in ridge_alpha_grid:
        if alpha <= 0:
            raise ValueError("ridge alpha grid must contain positive values")
        pair = _fit_components(core, residual_column=residual_col, ridge_alpha=alpha)
        ridge_pred, boost_pred = _component_predictions(pair, tune)
        for ridge_share in ridge_share_grid:
            if not 0.0 <= ridge_share <= 1.0:
                raise ValueError("ridge shares must be in [0, 1]")
            raw = _blend_residual(ridge_pred, boost_pred, ridge_share=ridge_share)
            for residual_weight in residual_weight_grid:
                if not 0.0 <= residual_weight <= 1.0:
                    raise ValueError("residual weights must be in [0, 1]")
                if residual_weight <= 1e-12:
                    continue
                adjusted = baseline + float(residual_weight) * raw
                metrics = _metrics(actual, adjusted)
                if _strictly_better(metrics, baseline_metrics):
                    spec = EnsembleSpec(
                        ridge_alpha=float(alpha),
                        ridge_share=float(ridge_share),
                        residual_weight=float(residual_weight),
                    )
                    eligible.append(
                        (
                            _selection_score(metrics, baseline_metrics),
                            float(alpha),
                            float(ridge_share),
                            float(residual_weight),
                            spec,
                            metrics,
                        )
                    )

    if not eligible:
        return None, baseline_metrics, baseline_metrics

    eligible.sort(key=lambda item: item[:4])
    _, _, _, _, spec, metrics = eligible[0]
    return spec, baseline_metrics, metrics


def evaluate_ensemble_fold(
    dataset: pl.DataFrame,
    *,
    season: int,
    target: str,
) -> EnsembleFold:
    """Tune using only seasons before season and evaluate that season once."""

    actual_col, baseline_col, residual_col = _target_columns(target)
    required = {
        "season",
        "week",
        actual_col,
        baseline_col,
        residual_col,
        *ENSEMBLE_FEATURES,
    }
    require_columns(dataset, required, "ensemble_dataset")

    training = dataset.filter(pl.col("season") < season)
    evaluation = dataset.filter(pl.col("season") == season)
    if training.is_empty() or evaluation.is_empty():
        raise DataContractError(f"missing training/evaluation rows for season {season}")

    core, tune = _whole_week_split(training)
    spec, tune_baseline, tune_adjusted = select_ensemble_spec(core, tune, target=target)

    actual = np.asarray(evaluation.get_column(actual_col), dtype=float)
    baseline = np.asarray(evaluation.get_column(baseline_col), dtype=float)
    baseline_metrics = _metrics(actual, baseline)

    if spec is None:
        adjusted_metrics = baseline_metrics
        passed = False
    else:
        pair = _fit_components(
            training,
            residual_column=residual_col,
            ridge_alpha=spec.ridge_alpha,
        )
        ridge_pred, boost_pred = _component_predictions(pair, evaluation)
        raw = _blend_residual(ridge_pred, boost_pred, ridge_share=spec.ridge_share)
        adjusted = baseline + spec.residual_weight * raw
        adjusted_metrics = _metrics(actual, adjusted)
        passed = _strictly_better(adjusted_metrics, baseline_metrics)

    return EnsembleFold(
        season=int(season),
        training_games=training.height,
        core_games=core.height,
        tune_games=tune.height,
        evaluation_games=evaluation.height,
        selected_spec=spec,
        tune_baseline=tune_baseline,
        tune_adjusted=tune_adjusted,
        evaluation_baseline=baseline_metrics,
        evaluation_adjusted=adjusted_metrics,
        passed=passed,
    )


def evaluate_rolling_ensemble(
    dataset: pl.DataFrame,
    *,
    target: str,
    test_seasons: tuple[int, ...] = (2023, 2024, 2025),
) -> RollingEnsembleResult:
    """Evaluate the NCAA-style ensemble architecture on rolling NFL seasons."""

    if not test_seasons:
        raise ValueError("test_seasons must not be empty")
    folds = tuple(
        evaluate_ensemble_fold(dataset, season=season, target=target)
        for season in test_seasons
    )

    baseline_abs = 0.0
    adjusted_abs = 0.0
    baseline_sq = 0.0
    adjusted_sq = 0.0
    games = 0
    for fold in folds:
        n = fold.evaluation_games
        games += n
        baseline_abs += n * fold.evaluation_baseline.mae
        adjusted_abs += n * fold.evaluation_adjusted.mae
        baseline_sq += n * (fold.evaluation_baseline.rmse**2)
        adjusted_sq += n * (fold.evaluation_adjusted.rmse**2)

    baseline_mae = baseline_abs / games
    adjusted_mae = adjusted_abs / games
    baseline_rmse = sqrt(baseline_sq / games)
    adjusted_rmse = sqrt(adjusted_sq / games)
    positive_folds = sum(fold.passed for fold in folds)
    eligible = (
        positive_folds == len(folds)
        and adjusted_mae < baseline_mae
        and adjusted_rmse < baseline_rmse
    )

    return RollingEnsembleResult(
        target=target,
        folds=folds,
        games=games,
        baseline_mae=float(baseline_mae),
        adjusted_mae=float(adjusted_mae),
        baseline_rmse=float(baseline_rmse),
        adjusted_rmse=float(adjusted_rmse),
        mae_improvement=float(baseline_mae - adjusted_mae),
        rmse_improvement=float(baseline_rmse - adjusted_rmse),
        positive_folds=int(positive_folds),
        eligible=bool(eligible),
    )
