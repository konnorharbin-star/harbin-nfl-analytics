"""Chronological nonlinear residual ensemble for NFL fair-score research.

Architecture mirrors the NCAA model at a high level: a standardized ridge residual
model is blended with a shallow histogram gradient-boosting residual model, and a
separate residual weight is tuned chronologically with weight 0 always admissible.

NFL model hyperparameters are selected independently. Sportsbook information is not
accepted by the dataset or this learner.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from math import sqrt

import numpy as np
import polars as pl
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .contracts import DataContractError, require_columns
from .ensemble_dataset import assert_no_feature_leakage, ensemble_feature_columns

DEFAULT_RIDGE_ALPHAS = (10.0, 30.0)
DEFAULT_RIDGE_FRACTIONS = (0.50, 0.70)
DEFAULT_BOOST_DEPTHS = (2, 3)
DEFAULT_RESIDUAL_WEIGHTS = (0.0, 0.25, 0.50, 0.75, 1.0)


@dataclass(frozen=True)
class EnsembleSpec:
    ridge_alpha: float
    ridge_fraction: float
    boost_depth: int

    @property
    def label(self) -> str:
        return (
            f"ridge={self.ridge_alpha:g}|mix={self.ridge_fraction:.2f}|"
            f"depth={self.boost_depth}"
        )


@dataclass(frozen=True)
class TuneSelection:
    spec: EnsembleSpec | None
    residual_weight: float
    baseline_mae: float
    selected_mae: float
    baseline_rmse: float
    selected_rmse: float


@dataclass(frozen=True)
class EnsembleFoldMetrics:
    tune_season: int
    test_season: int
    training_games: int
    tune_games: int
    test_games: int
    selected_spec: EnsembleSpec | None
    residual_weight: float
    tune_baseline_mae: float
    tune_selected_mae: float
    tune_baseline_rmse: float
    tune_selected_rmse: float
    test_baseline_mae: float
    test_adjusted_mae: float
    test_baseline_rmse: float
    test_adjusted_rmse: float

    @property
    def test_improves(self) -> bool:
        return (
            self.residual_weight > 0
            and self.test_adjusted_mae < self.test_baseline_mae
            and self.test_adjusted_rmse < self.test_baseline_rmse
        )


@dataclass(frozen=True)
class EnsembleSideEvaluation:
    side: str
    folds: tuple[EnsembleFoldMetrics, ...]
    aggregate_baseline_mae: float
    aggregate_adjusted_mae: float
    aggregate_baseline_rmse: float
    aggregate_adjusted_rmse: float
    positive_folds: int
    architecture_candidate: bool


@dataclass(frozen=True)
class NonlinearEnsembleEvaluation:
    seasons: tuple[int, ...]
    feature_count: int
    feature_columns: tuple[str, ...]
    margin: EnsembleSideEvaluation
    total: EnsembleSideEvaluation
    canonical_score_adjustment_enabled: bool
    promotion_eligible: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class FootballResidualEnsemble:
    """Ridge + shallow HGB residual ensemble fit on pregame football features."""

    def __init__(self, spec: EnsembleSpec, feature_columns: tuple[str, ...]) -> None:
        if spec.ridge_alpha <= 0:
            raise ValueError("ridge_alpha must be > 0")
        if not 0 <= spec.ridge_fraction <= 1:
            raise ValueError("ridge_fraction must be in [0, 1]")
        if spec.boost_depth < 1:
            raise ValueError("boost_depth must be >= 1")
        if not feature_columns:
            raise ValueError("feature_columns must not be empty")
        assert_no_feature_leakage(feature_columns)
        self.spec = spec
        self.feature_columns = tuple(feature_columns)
        self.ridge_model: Pipeline | None = None
        self.boost_model: Pipeline | None = None

    def _matrix(self, frame: pl.DataFrame) -> np.ndarray:
        require_columns(frame, set(self.feature_columns), "ensemble_features")
        return frame.select(list(self.feature_columns)).to_numpy().astype(float)

    def fit(self, frame: pl.DataFrame, target: str) -> FootballResidualEnsemble:
        require_columns(frame, {target}, "ensemble_training")
        if frame.height < 100:
            raise DataContractError("nonlinear ensemble requires at least 100 training games")
        x = self._matrix(frame)
        y = np.asarray(frame.get_column(target), dtype=float)
        if not np.isfinite(y).all():
            raise DataContractError(f"target {target} contains null/non-finite values")

        ridge = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=self.spec.ridge_alpha)),
            ]
        )
        boost = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_depth=self.spec.boost_depth,
                        learning_rate=0.05,
                        max_iter=200,
                        l2_regularization=10.0,
                        min_samples_leaf=20,
                        random_state=26,
                    ),
                ),
            ]
        )
        ridge.fit(x, y)
        boost.fit(x, y)
        self.ridge_model = ridge
        self.boost_model = boost
        return self

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        if self.ridge_model is None or self.boost_model is None:
            raise RuntimeError("nonlinear ensemble has not been fitted")
        x = self._matrix(frame)
        ridge_pred = np.asarray(self.ridge_model.predict(x), dtype=float)
        boost_pred = np.asarray(self.boost_model.predict(x), dtype=float)
        mix = self.spec.ridge_fraction
        return mix * ridge_pred + (1.0 - mix) * boost_pred


def _spec_grid(
    *,
    ridge_alphas: tuple[float, ...] = DEFAULT_RIDGE_ALPHAS,
    ridge_fractions: tuple[float, ...] = DEFAULT_RIDGE_FRACTIONS,
    boost_depths: tuple[int, ...] = DEFAULT_BOOST_DEPTHS,
) -> tuple[EnsembleSpec, ...]:
    if any(alpha <= 0 for alpha in ridge_alphas):
        raise ValueError("ridge_alphas must be positive")
    if any(not 0 <= fraction <= 1 for fraction in ridge_fractions):
        raise ValueError("ridge_fractions must be in [0, 1]")
    if any(depth < 1 for depth in boost_depths):
        raise ValueError("boost_depths must be >= 1")
    return tuple(
        EnsembleSpec(float(alpha), float(fraction), int(depth))
        for alpha, fraction, depth in product(
            ridge_alphas,
            ridge_fractions,
            boost_depths,
        )
    )


def _target_spec(side: str) -> tuple[str, str, str]:
    if side == "margin":
        return "margin_residual", "actual_home_margin", "baseline_home_margin"
    if side == "total":
        return "total_residual", "actual_total", "baseline_total"
    raise ValueError("side must be margin or total")


def _metrics(actual: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    error = prediction - actual
    return (
        float(np.mean(np.abs(error))),
        float(sqrt(float(np.mean(np.square(error))))),
    )


def _relative_objective(
    mae: float,
    rmse: float,
    baseline_mae: float,
    baseline_rmse: float,
) -> float:
    return mae / baseline_mae + rmse / baseline_rmse


def _select_on_tune(
    train: pl.DataFrame,
    tune: pl.DataFrame,
    *,
    side: str,
    specs: tuple[EnsembleSpec, ...],
    weight_grid: tuple[float, ...],
    features: tuple[str, ...],
) -> TuneSelection:
    target, actual_col, baseline_col = _target_spec(side)
    actual = np.asarray(tune.get_column(actual_col), dtype=float)
    baseline = np.asarray(tune.get_column(baseline_col), dtype=float)
    baseline_mae, baseline_rmse = _metrics(actual, baseline)

    best: tuple[float, float, str, EnsembleSpec, float, float] | None = None
    for spec in specs:
        model = FootballResidualEnsemble(spec, features).fit(train, target)
        correction = model.predict(tune)
        for weight in weight_grid:
            if weight <= 0:
                continue
            adjusted = baseline + float(weight) * correction
            mae, rmse = _metrics(actual, adjusted)
            if mae >= baseline_mae or rmse >= baseline_rmse:
                continue
            objective = _relative_objective(
                mae,
                rmse,
                baseline_mae,
                baseline_rmse,
            )
            candidate = (
                objective,
                float(weight),
                spec.label,
                spec,
                mae,
                rmse,
            )
            if best is None or candidate[:3] < best[:3]:
                best = candidate

    if best is None:
        return TuneSelection(
            spec=None,
            residual_weight=0.0,
            baseline_mae=baseline_mae,
            selected_mae=baseline_mae,
            baseline_rmse=baseline_rmse,
            selected_rmse=baseline_rmse,
        )

    _, weight, _, spec, mae, rmse = best
    return TuneSelection(
        spec=spec,
        residual_weight=weight,
        baseline_mae=baseline_mae,
        selected_mae=mae,
        baseline_rmse=baseline_rmse,
        selected_rmse=rmse,
    )


def _evaluate_fold(
    dataset: pl.DataFrame,
    *,
    side: str,
    tune_season: int,
    test_season: int,
    specs: tuple[EnsembleSpec, ...],
    weight_grid: tuple[float, ...],
    features: tuple[str, ...],
) -> tuple[EnsembleFoldMetrics, np.ndarray, np.ndarray, np.ndarray]:
    if tune_season >= test_season:
        raise ValueError("tune_season must be earlier than test_season")
    initial_train = dataset.filter(pl.col("season") < tune_season)
    tune = dataset.filter(pl.col("season") == tune_season)
    final_train = dataset.filter(pl.col("season") <= tune_season)
    test = dataset.filter(pl.col("season") == test_season)
    if initial_train.height < 100 or tune.is_empty() or test.is_empty():
        raise DataContractError("ensemble fold is missing sufficient train/tune/test games")

    selection = _select_on_tune(
        initial_train,
        tune,
        side=side,
        specs=specs,
        weight_grid=weight_grid,
        features=features,
    )
    target, actual_col, baseline_col = _target_spec(side)
    actual = np.asarray(test.get_column(actual_col), dtype=float)
    baseline = np.asarray(test.get_column(baseline_col), dtype=float)
    adjusted = baseline.copy()

    if selection.spec is not None and selection.residual_weight > 0:
        model = FootballResidualEnsemble(selection.spec, features).fit(
            final_train,
            target,
        )
        adjusted = (
            baseline
            + selection.residual_weight * model.predict(test)
        )

    baseline_mae, baseline_rmse = _metrics(actual, baseline)
    adjusted_mae, adjusted_rmse = _metrics(actual, adjusted)
    fold = EnsembleFoldMetrics(
        tune_season=tune_season,
        test_season=test_season,
        training_games=initial_train.height,
        tune_games=tune.height,
        test_games=test.height,
        selected_spec=selection.spec,
        residual_weight=selection.residual_weight,
        tune_baseline_mae=selection.baseline_mae,
        tune_selected_mae=selection.selected_mae,
        tune_baseline_rmse=selection.baseline_rmse,
        tune_selected_rmse=selection.selected_rmse,
        test_baseline_mae=baseline_mae,
        test_adjusted_mae=adjusted_mae,
        test_baseline_rmse=baseline_rmse,
        test_adjusted_rmse=adjusted_rmse,
    )
    return fold, actual, baseline, adjusted


def _evaluate_side(
    dataset: pl.DataFrame,
    *,
    side: str,
    fold_pairs: tuple[tuple[int, int], ...],
    specs: tuple[EnsembleSpec, ...],
    weight_grid: tuple[float, ...],
    features: tuple[str, ...],
) -> EnsembleSideEvaluation:
    folds: list[EnsembleFoldMetrics] = []
    actual_parts: list[np.ndarray] = []
    baseline_parts: list[np.ndarray] = []
    adjusted_parts: list[np.ndarray] = []
    for tune_season, test_season in fold_pairs:
        fold, actual, baseline, adjusted = _evaluate_fold(
            dataset,
            side=side,
            tune_season=tune_season,
            test_season=test_season,
            specs=specs,
            weight_grid=weight_grid,
            features=features,
        )
        folds.append(fold)
        actual_parts.append(actual)
        baseline_parts.append(baseline)
        adjusted_parts.append(adjusted)

    actual_all = np.concatenate(actual_parts)
    baseline_all = np.concatenate(baseline_parts)
    adjusted_all = np.concatenate(adjusted_parts)
    baseline_mae, baseline_rmse = _metrics(actual_all, baseline_all)
    adjusted_mae, adjusted_rmse = _metrics(actual_all, adjusted_all)
    positive_folds = sum(fold.test_improves for fold in folds)
    architecture_candidate = (
        positive_folds == len(folds)
        and adjusted_mae < baseline_mae
        and adjusted_rmse < baseline_rmse
    )
    return EnsembleSideEvaluation(
        side=side,
        folds=tuple(folds),
        aggregate_baseline_mae=baseline_mae,
        aggregate_adjusted_mae=adjusted_mae,
        aggregate_baseline_rmse=baseline_rmse,
        aggregate_adjusted_rmse=adjusted_rmse,
        positive_folds=positive_folds,
        architecture_candidate=architecture_candidate,
    )


def evaluate_nonlinear_ensemble(
    dataset: pl.DataFrame,
    *,
    fold_pairs: tuple[tuple[int, int], ...] = ((2023, 2024), (2024, 2025)),
    ridge_alphas: tuple[float, ...] = DEFAULT_RIDGE_ALPHAS,
    ridge_fractions: tuple[float, ...] = DEFAULT_RIDGE_FRACTIONS,
    boost_depths: tuple[int, ...] = DEFAULT_BOOST_DEPTHS,
    weight_grid: tuple[float, ...] = DEFAULT_RESIDUAL_WEIGHTS,
) -> NonlinearEnsembleEvaluation:
    """Run nested chronological ensemble selection with zero-weight fallback."""

    require_columns(
        dataset,
        {
            "season",
            "margin_residual",
            "total_residual",
            "actual_home_margin",
            "baseline_home_margin",
            "actual_total",
            "baseline_total",
        },
        "ensemble_dataset",
    )
    if not fold_pairs:
        raise ValueError("fold_pairs must not be empty")
    if 0.0 not in weight_grid:
        raise ValueError("weight_grid must include 0.0")
    if any(weight < 0 or weight > 1 for weight in weight_grid):
        raise ValueError("weight_grid values must be in [0, 1]")

    features = ensemble_feature_columns(dataset)
    assert_no_feature_leakage(features)
    specs = _spec_grid(
        ridge_alphas=ridge_alphas,
        ridge_fractions=ridge_fractions,
        boost_depths=boost_depths,
    )
    margin = _evaluate_side(
        dataset,
        side="margin",
        fold_pairs=fold_pairs,
        specs=specs,
        weight_grid=weight_grid,
        features=features,
    )
    total = _evaluate_side(
        dataset,
        side="total",
        fold_pairs=fold_pairs,
        specs=specs,
        weight_grid=weight_grid,
        features=features,
    )
    seasons = tuple(
        sorted(int(value) for value in dataset.get_column("season").unique().to_list())
    )
    return NonlinearEnsembleEvaluation(
        seasons=seasons,
        feature_count=len(features),
        feature_columns=features,
        margin=margin,
        total=total,
        canonical_score_adjustment_enabled=False,
        promotion_eligible=False,
    )
