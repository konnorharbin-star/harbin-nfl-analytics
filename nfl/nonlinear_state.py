"""Nonlinear residual models over the leak-safe NFL online-state feature family."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .contracts import DataContractError, require_columns
from .online_state import ONLINE_STATE_FEATURES


@dataclass(frozen=True)
class NonlinearStateSpec:
    name: str
    ridge_share: float
    max_depth: int
    learning_rate: float
    max_iter: int
    l2_regularization: float
    min_samples_leaf: int


NONLINEAR_STATE_SPECS = (
    NonlinearStateSpec(
        name="boost_d2",
        ridge_share=0.0,
        max_depth=2,
        learning_rate=0.05,
        max_iter=180,
        l2_regularization=12.0,
        min_samples_leaf=30,
    ),
    NonlinearStateSpec(
        name="hybrid_d2",
        ridge_share=0.50,
        max_depth=2,
        learning_rate=0.05,
        max_iter=180,
        l2_regularization=12.0,
        min_samples_leaf=30,
    ),
    NonlinearStateSpec(
        name="hybrid_d3",
        ridge_share=0.50,
        max_depth=3,
        learning_rate=0.04,
        max_iter=220,
        l2_regularization=18.0,
        min_samples_leaf=28,
    ),
)


def nonlinear_spec(name: str) -> NonlinearStateSpec:
    for spec in NONLINEAR_STATE_SPECS:
        if spec.name == name:
            return spec
    raise ValueError(f"unknown nonlinear state spec: {name}")


class NonlinearStateResidualModel:
    """Deterministic shallow boosting, optionally blended with standardized ridge."""

    def __init__(
        self,
        spec: NonlinearStateSpec,
        *,
        features: tuple[str, ...] = ONLINE_STATE_FEATURES,
        ridge_alpha: float = 10.0,
    ) -> None:
        if not features:
            raise ValueError("features must not be empty")
        if ridge_alpha <= 0:
            raise ValueError("ridge_alpha must be > 0")
        if not 0.0 <= spec.ridge_share <= 1.0:
            raise ValueError("ridge_share must be in [0, 1]")
        self.spec = spec
        self.features = tuple(features)
        self.ridge_alpha = float(ridge_alpha)
        self.scaler: StandardScaler | None = None
        self.ridge: Ridge | None = None
        self.boost: HistGradientBoostingRegressor | None = None

    def _matrix(self, frame: pl.DataFrame) -> np.ndarray:
        require_columns(frame, set(self.features), "nonlinear_state_features")
        matrix = frame.select(list(self.features)).to_numpy().astype(float)
        if not np.isfinite(matrix).all():
            raise DataContractError("nonlinear state features contain null/non-finite values")
        return matrix

    def fit(self, frame: pl.DataFrame, target: str) -> NonlinearStateResidualModel:
        require_columns(frame, {target}, "nonlinear_state_training")
        if frame.height < max(100, len(self.features) * 6):
            raise DataContractError("insufficient rows for nonlinear state residual model")
        x = self._matrix(frame)
        y = np.asarray(frame.get_column(target), dtype=float)
        if not np.isfinite(y).all():
            raise DataContractError(f"target {target} contains null/non-finite values")

        spec = self.spec
        self.boost = HistGradientBoostingRegressor(
            max_depth=spec.max_depth,
            learning_rate=spec.learning_rate,
            max_iter=spec.max_iter,
            l2_regularization=spec.l2_regularization,
            min_samples_leaf=spec.min_samples_leaf,
            random_state=26,
        ).fit(x, y)

        if spec.ridge_share > 0:
            self.scaler = StandardScaler().fit(x)
            z = self.scaler.transform(x)
            self.ridge = Ridge(alpha=self.ridge_alpha).fit(z, y)
        return self

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        if self.boost is None:
            raise RuntimeError("nonlinear state residual model is not fitted")
        x = self._matrix(frame)
        boost_prediction = np.asarray(self.boost.predict(x), dtype=float)
        share = self.spec.ridge_share
        if share <= 0:
            return boost_prediction
        if self.scaler is None or self.ridge is None:
            raise RuntimeError("hybrid nonlinear state residual model is incomplete")
        ridge_prediction = np.asarray(self.ridge.predict(self.scaler.transform(x)), dtype=float)
        return share * ridge_prediction + (1.0 - share) * boost_prediction
