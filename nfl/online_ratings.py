"""NCAA-style online opponent-adjusted ratings, validated independently for NFL.

This research module mirrors the *architecture* of the CFB baseline: persistent team
offense/defense state, offseason regression, a league scoring level, and residual
updates after games. NFL coefficients are selected independently and sportsbook data
is never accepted.

All games in a target week are predicted before that week's results update state.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from itertools import product
from math import sqrt

import numpy as np
import polars as pl

from .contracts import DataContractError, require_columns
from .data import completed_games
from .recency import build_score_walkforward

DEFAULT_UPDATE_GRID = (0.04, 0.08, 0.12)
DEFAULT_CARRY_GRID = (0.40, 0.60, 0.80)
DEFAULT_HOME_FIELD_GRID = (1.0, 1.75, 2.5)
NFL_LEAGUE_PPG_PRIOR = 22.5
LEAGUE_WEEK_ALPHA = 0.08
LEAGUE_OFFSEASON_CARRY = 0.50


@dataclass
class OnlineTeamState:
    offense: float = 0.0
    defense: float = 0.0

    def offseason_regress(self, carry: float) -> None:
        self.offense *= float(carry)
        self.defense *= float(carry)


@dataclass(frozen=True)
class OnlineConfig:
    update_alpha: float
    offseason_carry: float
    home_field: float

    @property
    def label(self) -> str:
        return (
            f"alpha={self.update_alpha:.3f}|carry={self.offseason_carry:.2f}|"
            f"hfa={self.home_field:.2f}"
        )


class OnlineOpponentAdjustedRatings:
    """Persistent offense/defense state updated only after a full week is predicted."""

    def __init__(
        self,
        config: OnlineConfig,
        *,
        league_ppg_prior: float = NFL_LEAGUE_PPG_PRIOR,
        league_week_alpha: float = LEAGUE_WEEK_ALPHA,
    ) -> None:
        if not 0 < config.update_alpha <= 1:
            raise ValueError("update_alpha must be in (0, 1]")
        if not 0 <= config.offseason_carry <= 1:
            raise ValueError("offseason_carry must be in [0, 1]")
        if config.home_field < 0:
            raise ValueError("home_field must be >= 0")
        if not 0 < league_week_alpha <= 1:
            raise ValueError("league_week_alpha must be in (0, 1]")
        self.config = config
        self.league_ppg_prior = float(league_ppg_prior)
        self.league_ppg = float(league_ppg_prior)
        self.league_week_alpha = float(league_week_alpha)
        self.state: defaultdict[str, OnlineTeamState] = defaultdict(OnlineTeamState)
        self.season: int | None = None

    def enter_season(self, season: int) -> None:
        if self.season is None:
            self.season = int(season)
            return
        if int(season) == self.season:
            return
        for team_state in self.state.values():
            team_state.offseason_regress(self.config.offseason_carry)
        self.league_ppg = self.league_ppg_prior + LEAGUE_OFFSEASON_CARRY * (
            self.league_ppg - self.league_ppg_prior
        )
        self.season = int(season)

    def project(self, home_team: str, away_team: str) -> tuple[float, float]:
        home = self.state[home_team]
        away = self.state[away_team]
        home_points = (
            self.league_ppg
            + home.offense
            - away.defense
            + self.config.home_field
        )
        away_points = self.league_ppg + away.offense - home.defense
        return max(0.0, float(home_points)), max(0.0, float(away_points))

    def update_week(self, games: pl.DataFrame) -> None:
        """Update team states only after every game in the week has been predicted."""

        if games.is_empty():
            return
        require_columns(
            games,
            {"home_team", "away_team", "home_score", "away_score"},
            "online_rating_week",
        )
        alpha = self.config.update_alpha
        updates: list[tuple[str, str, float, float]] = []
        ppg_values: list[float] = []
        for game in games.iter_rows(named=True):
            home_team = str(game["home_team"])
            away_team = str(game["away_team"])
            predicted_home, predicted_away = self.project(home_team, away_team)
            home_score = float(game["home_score"])
            away_score = float(game["away_score"])
            updates.append(
                (
                    home_team,
                    away_team,
                    home_score - predicted_home,
                    away_score - predicted_away,
                )
            )
            ppg_values.append((home_score + away_score) / 2.0)

        for home_team, away_team, home_residual, away_residual in updates:
            home = self.state[home_team]
            away = self.state[away_team]
            home.offense += alpha * home_residual
            away.defense -= alpha * home_residual
            away.offense += alpha * away_residual
            home.defense -= alpha * away_residual

        week_ppg = float(np.mean(ppg_values))
        self.league_ppg = (
            (1.0 - self.league_week_alpha) * self.league_ppg
            + self.league_week_alpha * week_ppg
        )


def _regular_games(
    schedules: pl.DataFrame,
    *,
    warmup_start: int,
    final_season: int,
) -> pl.DataFrame:
    require_columns(
        schedules,
        {
            "season",
            "week",
            "game_id",
            "game_type",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
        },
        "schedules",
    )
    return (
        completed_games(schedules)
        .filter(
            (pl.col("game_type") == "REG")
            & (pl.col("season") >= warmup_start)
            & (pl.col("season") <= final_season)
        )
        .sort(["season", "week", "game_id"])
    )


def build_online_predictions(
    schedules: pl.DataFrame,
    config: OnlineConfig,
    *,
    test_seasons: tuple[int, ...] = (2023, 2024, 2025),
    warmup_start: int = 2021,
    start_week: int = 5,
    end_week: int = 18,
) -> pl.DataFrame:
    """Build strict pregame predictions from a persistent weekly state machine."""

    if not test_seasons:
        raise ValueError("test_seasons must not be empty")
    final_season = max(test_seasons)
    games = _regular_games(
        schedules,
        warmup_start=warmup_start,
        final_season=final_season,
    )
    if games.is_empty():
        raise DataContractError("online ratings received no completed regular-season games")

    model = OnlineOpponentAdjustedRatings(config)
    rows: list[dict[str, object]] = []
    for season in sorted(games.get_column("season").unique().to_list()):
        model.enter_season(int(season))
        season_games = games.filter(pl.col("season") == season)
        for week in sorted(season_games.get_column("week").unique().to_list()):
            week_games = season_games.filter(pl.col("week") == week)
            should_score = (
                int(season) in test_seasons
                and start_week <= int(week) <= end_week
            )
            if should_score:
                for game in week_games.iter_rows(named=True):
                    home_team = str(game["home_team"])
                    away_team = str(game["away_team"])
                    home_points, away_points = model.project(home_team, away_team)
                    rows.append(
                        {
                            "season": int(season),
                            "week": int(week),
                            "game_id": str(game["game_id"]),
                            "online_home_margin": home_points - away_points,
                            "online_total": home_points + away_points,
                        }
                    )
            model.update_week(week_games)

    if not rows:
        raise DataContractError("online ratings produced no evaluation predictions")
    return pl.DataFrame(rows).sort(["season", "week", "game_id"])


@dataclass(frozen=True)
class OnlineFoldMetrics:
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
class OnlineCandidateMetrics:
    config: OnlineConfig
    folds: tuple[OnlineFoldMetrics, ...]
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
class OnlineRatingsEvaluation:
    test_seasons: tuple[int, ...]
    games: int
    selected_margin_config: OnlineConfig | None
    selected_total_config: OnlineConfig | None
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
    margin_candidates: tuple[OnlineCandidateMetrics, ...]
    total_candidates: tuple[OnlineCandidateMetrics, ...]
    canonical_score_change_enabled: bool
    promotion_eligible: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _error_metrics(
    frame: pl.DataFrame,
    prediction: str,
    actual: str,
) -> tuple[float, float]:
    error = np.asarray(frame.get_column(prediction) - frame.get_column(actual), dtype=float)
    return (
        float(np.mean(np.abs(error))),
        float(sqrt(float(np.mean(np.square(error))))),
    )


def _baseline_frame(
    schedules: pl.DataFrame,
    test_seasons: tuple[int, ...],
    *,
    start_week: int,
    end_week: int,
) -> pl.DataFrame:
    frames = [
        build_score_walkforward(
            schedules,
            season,
            start_week=start_week,
            end_week=end_week,
            ridge=8.0,
        )
        for season in test_seasons
    ]
    return pl.concat(frames, how="vertical_relaxed").select(
        "season",
        "week",
        "game_id",
        pl.col("projected_home_margin").alias("baseline_home_margin"),
        pl.col("projected_total").alias("baseline_total"),
        "actual_home_margin",
        "actual_total",
    )


def _evaluate_config(
    baseline: pl.DataFrame,
    candidate: pl.DataFrame,
    *,
    config: OnlineConfig,
    side: str,
    test_seasons: tuple[int, ...],
) -> OnlineCandidateMetrics:
    joined = baseline.join(
        candidate,
        on=["season", "week", "game_id"],
        how="inner",
        validate="1:1",
    )
    if joined.height != baseline.height:
        raise DataContractError(
            "online prediction coverage does not match canonical baseline coverage"
        )
    actual = "actual_home_margin" if side == "margin" else "actual_total"
    baseline_col = "baseline_home_margin" if side == "margin" else "baseline_total"
    candidate_col = "online_home_margin" if side == "margin" else "online_total"

    folds: list[OnlineFoldMetrics] = []
    for season in test_seasons:
        fold = joined.filter(pl.col("season") == season)
        baseline_mae, baseline_rmse = _error_metrics(fold, baseline_col, actual)
        candidate_mae, candidate_rmse = _error_metrics(fold, candidate_col, actual)
        folds.append(
            OnlineFoldMetrics(
                season=season,
                games=fold.height,
                baseline_mae=baseline_mae,
                candidate_mae=candidate_mae,
                baseline_rmse=baseline_rmse,
                candidate_rmse=candidate_rmse,
            )
        )

    baseline_mae, baseline_rmse = _error_metrics(joined, baseline_col, actual)
    candidate_mae, candidate_rmse = _error_metrics(joined, candidate_col, actual)
    return OnlineCandidateMetrics(
        config=config,
        folds=tuple(folds),
        baseline_mae=baseline_mae,
        candidate_mae=candidate_mae,
        baseline_rmse=baseline_rmse,
        candidate_rmse=candidate_rmse,
        positive_folds=sum(fold.improves for fold in folds),
    )


def _select_candidate(
    candidates: tuple[OnlineCandidateMetrics, ...],
) -> OnlineCandidateMetrics | None:
    eligible = [candidate for candidate in candidates if candidate.passes]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda item: (
            item.relative_objective,
            item.config.update_alpha,
            item.config.offseason_carry,
            item.config.home_field,
        ),
    )


def evaluate_online_ratings(
    schedules: pl.DataFrame,
    *,
    test_seasons: tuple[int, ...] = (2023, 2024, 2025),
    update_grid: tuple[float, ...] = DEFAULT_UPDATE_GRID,
    carry_grid: tuple[float, ...] = DEFAULT_CARRY_GRID,
    home_field_grid: tuple[float, ...] = DEFAULT_HOME_FIELD_GRID,
    warmup_start: int = 2021,
    start_week: int = 5,
    end_week: int = 18,
) -> OnlineRatingsEvaluation:
    """Evaluate NFL-tuned NCAA-style online ratings with an explicit zero fallback."""

    if not update_grid or any(not 0 < value <= 1 for value in update_grid):
        raise ValueError("update_grid must contain values in (0, 1]")
    if not carry_grid or any(not 0 <= value <= 1 for value in carry_grid):
        raise ValueError("carry_grid must contain values in [0, 1]")
    if not home_field_grid or any(value < 0 for value in home_field_grid):
        raise ValueError("home_field_grid must contain non-negative values")

    baseline = _baseline_frame(
        schedules,
        test_seasons,
        start_week=start_week,
        end_week=end_week,
    )
    configs = tuple(
        OnlineConfig(float(alpha), float(carry), float(home_field))
        for alpha, carry, home_field in product(
            update_grid,
            carry_grid,
            home_field_grid,
        )
    )

    margin_candidates: list[OnlineCandidateMetrics] = []
    total_candidates: list[OnlineCandidateMetrics] = []
    for config in configs:
        candidate = build_online_predictions(
            schedules,
            config,
            test_seasons=test_seasons,
            warmup_start=warmup_start,
            start_week=start_week,
            end_week=end_week,
        )
        margin_candidates.append(
            _evaluate_config(
                baseline,
                candidate,
                config=config,
                side="margin",
                test_seasons=test_seasons,
            )
        )
        total_candidates.append(
            _evaluate_config(
                baseline,
                candidate,
                config=config,
                side="total",
                test_seasons=test_seasons,
            )
        )

    margin_tuple = tuple(margin_candidates)
    total_tuple = tuple(total_candidates)
    selected_margin = _select_candidate(margin_tuple)
    selected_total = _select_candidate(total_tuple)
    margin_reference = margin_tuple[0]
    total_reference = total_tuple[0]

    return OnlineRatingsEvaluation(
        test_seasons=test_seasons,
        games=baseline.height,
        selected_margin_config=None if selected_margin is None else selected_margin.config,
        selected_total_config=None if selected_total is None else selected_total.config,
        margin_shadow_candidate=selected_margin is not None,
        total_shadow_candidate=selected_total is not None,
        margin_baseline_mae=margin_reference.baseline_mae,
        margin_selected_mae=(
            margin_reference.baseline_mae
            if selected_margin is None
            else selected_margin.candidate_mae
        ),
        margin_baseline_rmse=margin_reference.baseline_rmse,
        margin_selected_rmse=(
            margin_reference.baseline_rmse
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
        margin_candidates=margin_tuple,
        total_candidates=total_tuple,
        canonical_score_change_enabled=False,
        promotion_eligible=False,
    )
