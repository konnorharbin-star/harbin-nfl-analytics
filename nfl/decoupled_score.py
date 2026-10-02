"""Research-only decoupled NFL margin and total score architecture.

The canonical fair-score model estimates team points and derives margin/total from those
two point forecasts. This module tests a different football-only architecture:

* margin is estimated directly from ridge-shrunk team strength plus home field;
* total is estimated directly from ridge-shrunk team scoring environment.

Both models use only completed regular-season schedule results available before the
target week. Sportsbook information is not accepted anywhere in this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt

import numpy as np
import polars as pl

from .contracts import DataContractError, require_columns
from .data import completed_games
from .ratings import VALIDATED_PRIOR_SEASON_WEIGHT, fit_pregame_fair_score

DEFAULT_RIDGE_GRID = (2.0, 8.0, 32.0)


@dataclass(frozen=True)
class DecoupledProjection:
    home_team: str
    away_team: str
    home_margin: float
    total: float

    @property
    def home_points(self) -> float:
        return (self.total + self.home_margin) / 2.0

    @property
    def away_points(self) -> float:
        return (self.total - self.home_margin) / 2.0


class DecoupledScoreModel:
    """Fit separate game-level margin and total ridge models."""

    def __init__(self, *, margin_ridge: float = 8.0, total_ridge: float = 8.0) -> None:
        if margin_ridge <= 0 or total_ridge <= 0:
            raise ValueError("ridge penalties must be > 0")
        self.margin_ridge = float(margin_ridge)
        self.total_ridge = float(total_ridge)
        self.teams: tuple[str, ...] = ()
        self.home_field: float | None = None
        self.margin_rating: dict[str, float] = {}
        self.league_total: float | None = None
        self.total_environment: dict[str, float] = {}

    @property
    def fitted(self) -> bool:
        return self.home_field is not None and self.league_total is not None

    @staticmethod
    def _solve(
        x: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        ridge: float,
    ) -> np.ndarray:
        sqrt_weight = np.sqrt(weights)
        weighted_x = x * sqrt_weight[:, None]
        weighted_y = y * sqrt_weight
        penalty = np.eye(x.shape[1], dtype=float) * float(ridge)
        penalty[0, 0] = 0.0
        return np.linalg.solve(
            weighted_x.T @ weighted_x + penalty,
            weighted_x.T @ weighted_y,
        )

    def fit(
        self,
        games: pl.DataFrame,
        *,
        sample_weight: np.ndarray | None = None,
    ) -> DecoupledScoreModel:
        required = {
            "home_team",
            "away_team",
            "home_score",
            "away_score",
        }
        require_columns(games, required, "decoupled_score_games")
        if games.height < 2:
            raise DataContractError("decoupled score model requires at least two games")

        teams = sorted(
            set(games.get_column("home_team").drop_nulls().to_list())
            | set(games.get_column("away_team").drop_nulls().to_list())
        )
        if len(teams) < 2:
            raise DataContractError("decoupled score model requires at least two teams")

        n_games = games.height
        if sample_weight is None:
            weights = np.ones(n_games, dtype=float)
        else:
            weights = np.asarray(sample_weight, dtype=float)
            if weights.shape != (n_games,):
                raise DataContractError(
                    f"sample_weight must have shape ({n_games},), got {weights.shape}"
                )
            if not np.isfinite(weights).all() or np.any(weights <= 0):
                raise DataContractError(
                    "sample_weight must contain finite positive values"
                )

        team_index = {team: index for index, team in enumerate(teams)}
        home = games.get_column("home_team").to_list()
        away = games.get_column("away_team").to_list()
        home_score = np.asarray(
            games.get_column("home_score").cast(pl.Float64),
            dtype=float,
        )
        away_score = np.asarray(
            games.get_column("away_score").cast(pl.Float64),
            dtype=float,
        )

        margin_x = np.zeros((n_games, 1 + len(teams)), dtype=float)
        total_x = np.zeros_like(margin_x)
        margin_x[:, 0] = 1.0
        total_x[:, 0] = 1.0
        for row, (home_team, away_team) in enumerate(zip(home, away, strict=True)):
            margin_x[row, 1 + team_index[home_team]] = 1.0
            margin_x[row, 1 + team_index[away_team]] = -1.0
            total_x[row, 1 + team_index[home_team]] = 1.0
            total_x[row, 1 + team_index[away_team]] = 1.0

        margin_y = home_score - away_score
        total_y = home_score + away_score
        margin_beta = self._solve(
            margin_x,
            margin_y,
            weights,
            self.margin_ridge,
        )
        total_beta = self._solve(
            total_x,
            total_y,
            weights,
            self.total_ridge,
        )

        raw_margin = margin_beta[1:]
        margin_mean = float(np.mean(raw_margin))
        raw_total = total_beta[1:]
        total_mean = float(np.mean(raw_total))

        self.teams = tuple(teams)
        self.home_field = float(margin_beta[0])
        self.margin_rating = {
            team: float(raw_margin[index] - margin_mean)
            for team, index in team_index.items()
        }
        self.league_total = float(total_beta[0] + (2.0 * total_mean))
        self.total_environment = {
            team: float(raw_total[index] - total_mean)
            for team, index in team_index.items()
        }
        return self

    def project(self, home_team: str, away_team: str) -> DecoupledProjection:
        if not self.fitted:
            raise RuntimeError("decoupled score model has not been fitted")
        if home_team not in self.margin_rating or away_team not in self.margin_rating:
            raise DataContractError("target team was not present in training history")
        assert self.home_field is not None
        assert self.league_total is not None
        margin = (
            self.home_field
            + self.margin_rating[home_team]
            - self.margin_rating[away_team]
        )
        total = (
            self.league_total
            + self.total_environment[home_team]
            + self.total_environment[away_team]
        )
        return DecoupledProjection(
            home_team=home_team,
            away_team=away_team,
            home_margin=float(margin),
            total=max(0.0, float(total)),
        )


def decoupled_history(
    schedules: pl.DataFrame,
    season: int,
    week: int,
) -> pl.DataFrame:
    """Return the exact canonical regular-season pregame score history."""

    require_columns(
        schedules,
        {"season", "week", "game_type", "home_score", "away_score"},
        "schedules",
    )
    if week < 1:
        raise ValueError("week must be >= 1")
    history = completed_games(schedules).filter(pl.col("game_type") == "REG")
    return history.filter(
        (pl.col("season") == season - 1)
        | ((pl.col("season") == season) & (pl.col("week") < week))
    )


def decoupled_sample_weights(
    history: pl.DataFrame,
    *,
    season: int,
    prior_season_weight: float = VALIDATED_PRIOR_SEASON_WEIGHT,
) -> np.ndarray:
    """Weight current-season games fully and the prior season at the validated weight."""

    require_columns(history, {"season"}, "decoupled_history")
    if not 0 < prior_season_weight <= 1:
        raise ValueError("prior_season_weight must be in (0, 1]")
    seasons = np.asarray(history.get_column("season"), dtype=int)
    valid = (seasons == season) | (seasons == season - 1)
    if not valid.all():
        raise DataContractError("decoupled history contains an unsupported season")
    return np.where(seasons == season, 1.0, float(prior_season_weight))


def _target_games(schedules: pl.DataFrame, season: int, week: int) -> pl.DataFrame:
    return completed_games(schedules).filter(
        (pl.col("season") == season)
        & (pl.col("game_type") == "REG")
        & (pl.col("week") == week)
    )


def build_week_decoupled_predictions(
    schedules: pl.DataFrame,
    season: int,
    week: int,
    *,
    margin_ridge: float,
    total_ridge: float,
    baseline_ridge: float = 8.0,
    prior_season_weight: float = VALIDATED_PRIOR_SEASON_WEIGHT,
) -> pl.DataFrame:
    """Project one completed week from the strict pregame boundary."""

    targets = _target_games(schedules, season, week)
    if targets.is_empty():
        return pl.DataFrame()
    history = decoupled_history(schedules, season, week)
    if history.is_empty():
        raise DataContractError("no pregame score history is available")

    weights = decoupled_sample_weights(
        history,
        season=season,
        prior_season_weight=prior_season_weight,
    )
    candidate = DecoupledScoreModel(
        margin_ridge=margin_ridge,
        total_ridge=total_ridge,
    ).fit(history, sample_weight=weights)
    baseline = fit_pregame_fair_score(
        schedules,
        season,
        week,
        ridge=baseline_ridge,
        prior_season_weight=prior_season_weight,
    )

    rows: list[dict[str, object]] = []
    for game in targets.iter_rows(named=True):
        home_team = str(game["home_team"])
        away_team = str(game["away_team"])
        candidate_projection = candidate.project(home_team, away_team)
        baseline_projection = baseline.project(home_team, away_team)
        home_score = float(game["home_score"])
        away_score = float(game["away_score"])
        rows.append(
            {
                "season": season,
                "week": week,
                "game_id": str(game["game_id"]),
                "home_team": home_team,
                "away_team": away_team,
                "actual_home_margin": home_score - away_score,
                "actual_total": home_score + away_score,
                "baseline_home_margin": baseline_projection.home_margin,
                "baseline_total": baseline_projection.total,
                "decoupled_home_margin": candidate_projection.home_margin,
                "decoupled_total": candidate_projection.total,
            }
        )
    return pl.DataFrame(rows).sort(["week", "game_id"])


def build_decoupled_walkforward(
    schedules: pl.DataFrame,
    season: int,
    *,
    margin_ridge: float,
    total_ridge: float,
    start_week: int = 5,
    end_week: int = 18,
    baseline_ridge: float = 8.0,
) -> pl.DataFrame:
    """Reconstruct one season week by week with no target-week leakage."""

    if start_week < 2:
        raise ValueError("start_week must be >= 2")
    regular = completed_games(schedules).filter(
        (pl.col("season") == season) & (pl.col("game_type") == "REG")
    )
    if regular.is_empty():
        raise DataContractError(f"no completed regular-season games found for {season}")
    final_week = min(end_week, int(regular.get_column("week").max()))
    frames = [
        build_week_decoupled_predictions(
            schedules,
            season,
            week,
            margin_ridge=margin_ridge,
            total_ridge=total_ridge,
            baseline_ridge=baseline_ridge,
        )
        for week in range(start_week, final_week + 1)
    ]
    frames = [frame for frame in frames if not frame.is_empty()]
    if not frames:
        raise DataContractError("decoupled walk-forward produced no games")
    return pl.concat(frames, how="vertical_relaxed").sort(
        ["season", "week", "game_id"]
    )


@dataclass(frozen=True)
class SideFoldMetrics:
    season: int
    games: int
    baseline_mae: float
    candidate_mae: float
    baseline_rmse: float
    candidate_rmse: float

    @property
    def improves(self) -> bool:
        return (
            self.candidate_mae < self.baseline_mae
            and self.candidate_rmse < self.baseline_rmse
        )


@dataclass(frozen=True)
class SideCandidateMetrics:
    ridge: float
    folds: tuple[SideFoldMetrics, ...]
    baseline_mae: float
    candidate_mae: float
    baseline_rmse: float
    candidate_rmse: float
    positive_folds: int

    @property
    def passes(self) -> bool:
        return (
            self.positive_folds == len(self.folds)
            and self.candidate_mae < self.baseline_mae
            and self.candidate_rmse < self.baseline_rmse
        )

    @property
    def relative_objective(self) -> float:
        return (
            self.candidate_mae / self.baseline_mae
            + self.candidate_rmse / self.baseline_rmse
        )


@dataclass(frozen=True)
class DecoupledArchitectureEvaluation:
    test_seasons: tuple[int, ...]
    games: int
    selected_margin_ridge: float | None
    selected_total_ridge: float | None
    margin_shadow_candidate: bool
    total_shadow_candidate: bool
    margin_baseline_mae: float
    margin_selected_mae: float
    margin_baseline_rmse: float
    margin_selected_rmse: float
    total_baseline_mae: float
    total_selected_mae: float
    total_baseline_rmse: float
    total_selected_rmse: float
    margin_candidates: tuple[SideCandidateMetrics, ...]
    total_candidates: tuple[SideCandidateMetrics, ...]
    canonical_score_change_enabled: bool
    promotion_eligible: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _errors(frame: pl.DataFrame, prediction: str, actual: str) -> tuple[float, float]:
    error = np.asarray(frame.get_column(prediction) - frame.get_column(actual), dtype=float)
    return (
        float(np.mean(np.abs(error))),
        float(sqrt(float(np.mean(np.square(error))))),
    )


def _evaluate_side(
    schedules: pl.DataFrame,
    *,
    side: str,
    ridge: float,
    test_seasons: tuple[int, ...],
    start_week: int,
    end_week: int,
) -> SideCandidateMetrics:
    if side not in {"margin", "total"}:
        raise ValueError("side must be margin or total")

    frames: list[pl.DataFrame] = []
    folds: list[SideFoldMetrics] = []
    for season in test_seasons:
        frame = build_decoupled_walkforward(
            schedules,
            season,
            margin_ridge=ridge if side == "margin" else 8.0,
            total_ridge=ridge if side == "total" else 8.0,
            start_week=start_week,
            end_week=end_week,
        )
        frames.append(frame)
        actual = "actual_home_margin" if side == "margin" else "actual_total"
        baseline = "baseline_home_margin" if side == "margin" else "baseline_total"
        candidate = (
            "decoupled_home_margin" if side == "margin" else "decoupled_total"
        )
        baseline_mae, baseline_rmse = _errors(frame, baseline, actual)
        candidate_mae, candidate_rmse = _errors(frame, candidate, actual)
        folds.append(
            SideFoldMetrics(
                season=season,
                games=frame.height,
                baseline_mae=baseline_mae,
                candidate_mae=candidate_mae,
                baseline_rmse=baseline_rmse,
                candidate_rmse=candidate_rmse,
            )
        )

    aggregate = pl.concat(frames, how="vertical_relaxed")
    actual = "actual_home_margin" if side == "margin" else "actual_total"
    baseline = "baseline_home_margin" if side == "margin" else "baseline_total"
    candidate = "decoupled_home_margin" if side == "margin" else "decoupled_total"
    baseline_mae, baseline_rmse = _errors(aggregate, baseline, actual)
    candidate_mae, candidate_rmse = _errors(aggregate, candidate, actual)
    return SideCandidateMetrics(
        ridge=float(ridge),
        folds=tuple(folds),
        baseline_mae=baseline_mae,
        candidate_mae=candidate_mae,
        baseline_rmse=baseline_rmse,
        candidate_rmse=candidate_rmse,
        positive_folds=sum(fold.improves for fold in folds),
    )


def _select_side(
    candidates: tuple[SideCandidateMetrics, ...],
) -> SideCandidateMetrics | None:
    eligible = [candidate for candidate in candidates if candidate.passes]
    if not eligible:
        return None
    return min(eligible, key=lambda item: (item.relative_objective, item.ridge))


def evaluate_decoupled_architecture(
    schedules: pl.DataFrame,
    *,
    test_seasons: tuple[int, ...] = (2023, 2024, 2025),
    ridge_grid: tuple[float, ...] = DEFAULT_RIDGE_GRID,
    start_week: int = 5,
    end_week: int = 18,
) -> DecoupledArchitectureEvaluation:
    """Evaluate fixed margin/total architectures with an explicit zero fallback."""

    if not test_seasons:
        raise ValueError("test_seasons must not be empty")
    if not ridge_grid or any(ridge <= 0 for ridge in ridge_grid):
        raise ValueError("ridge_grid must contain positive values")

    margin_candidates = tuple(
        _evaluate_side(
            schedules,
            side="margin",
            ridge=ridge,
            test_seasons=test_seasons,
            start_week=start_week,
            end_week=end_week,
        )
        for ridge in ridge_grid
    )
    total_candidates = tuple(
        _evaluate_side(
            schedules,
            side="total",
            ridge=ridge,
            test_seasons=test_seasons,
            start_week=start_week,
            end_week=end_week,
        )
        for ridge in ridge_grid
    )
    selected_margin = _select_side(margin_candidates)
    selected_total = _select_side(total_candidates)

    reference = margin_candidates[0]
    total_reference = total_candidates[0]
    games = sum(fold.games for fold in reference.folds)
    return DecoupledArchitectureEvaluation(
        test_seasons=test_seasons,
        games=games,
        selected_margin_ridge=None if selected_margin is None else selected_margin.ridge,
        selected_total_ridge=None if selected_total is None else selected_total.ridge,
        margin_shadow_candidate=selected_margin is not None,
        total_shadow_candidate=selected_total is not None,
        margin_baseline_mae=reference.baseline_mae,
        margin_selected_mae=(
            reference.baseline_mae
            if selected_margin is None
            else selected_margin.candidate_mae
        ),
        margin_baseline_rmse=reference.baseline_rmse,
        margin_selected_rmse=(
            reference.baseline_rmse
            if selected_margin is None
            else selected_margin.candidate_rmse
        ),
        total_baseline_mae=total_reference.baseline_mae,
        total_selected_mae=(
            total_reference.baseline_mae
            if selected_total is None
            else selected_total.candidate_mae
        ),
        total_baseline_rmse=total_reference.baseline_rmse,
        total_selected_rmse=(
            total_reference.baseline_rmse
            if selected_total is None
            else selected_total.candidate_rmse
        ),
        margin_candidates=margin_candidates,
        total_candidates=total_candidates,
        canonical_score_change_enabled=False,
        promotion_eligible=False,
    )
