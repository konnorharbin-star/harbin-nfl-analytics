"""NCAA-style sequential pregame team state for NFL research.

The state is updated only after an entire NFL week has been snapshotted, so every
Week W feature is based on weeks < W. Sportsbook information is not accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log1p

import polars as pl

from .contracts import DataContractError, require_columns
from .data import completed_games

ONLINE_STATE_FEATURES = (
    "online_home_points",
    "online_away_points",
    "online_margin",
    "online_total",
    "elo_diff_home",
    "offense_diff_home",
    "defense_diff_home",
    "net_eff_diff_home",
    "margin_form_diff_home",
    "total_form_avg",
    "win_rate_diff_home",
    "volatility_avg",
    "sos_diff_home",
    "home_games",
    "away_games",
    "games_diff_home",
    "early_season",
)


@dataclass(frozen=True)
class OnlineStateConfig:
    name: str
    score_alpha: float
    form_alpha: float
    elo_k: float
    home_field_points: float
    offseason_skill_carry: float
    offseason_form_carry: float
    offseason_elo_carry: float


ONLINE_STATE_CONFIGS = (
    OnlineStateConfig("slow", 0.06, 0.15, 16.0, 1.5, 0.68, 0.48, 0.82),
    OnlineStateConfig("balanced", 0.10, 0.22, 20.0, 1.7, 0.62, 0.42, 0.79),
    OnlineStateConfig("responsive", 0.14, 0.30, 24.0, 1.9, 0.56, 0.36, 0.76),
)


@dataclass
class TeamState:
    elo: float = 1500.0
    offense: float = 0.0
    defense: float = 0.0
    margin_form: float = 0.0
    total_form: float = 45.0
    win_rate: float = 0.5
    volatility: float = 10.0
    sos_elo: float = 1500.0
    games: int = 0


def _neutral_site(game: dict[str, object]) -> bool:
    if "neutral_site" in game and game["neutral_site"] is not None:
        return bool(game["neutral_site"])
    value = str(game.get("location", "") or "").strip().lower()
    return value in {"neutral", "neutral site", "neutral_site"}


class OnlineTeamState:
    """Sequential opponent-adjusted state with offseason regression."""

    def __init__(self, config: OnlineStateConfig) -> None:
        self.config = config
        self.season: int | None = None
        self.league_ppg = 22.5
        self.state: dict[str, TeamState] = {}

    def _team(self, name: str) -> TeamState:
        if name not in self.state:
            self.state[name] = TeamState()
        return self.state[name]

    def _regress_offseason(self) -> None:
        c = self.config
        for state in self.state.values():
            state.elo = 1500.0 + c.offseason_elo_carry * (state.elo - 1500.0)
            state.offense *= c.offseason_skill_carry
            state.defense *= c.offseason_skill_carry
            state.margin_form *= c.offseason_form_carry
            state.total_form = 45.0 + c.offseason_form_carry * (state.total_form - 45.0)
            state.win_rate = 0.5 + c.offseason_form_carry * (state.win_rate - 0.5)
            state.volatility = 10.0 + c.offseason_form_carry * (state.volatility - 10.0)
            state.sos_elo = 1500.0 + c.offseason_skill_carry * (state.sos_elo - 1500.0)
            state.games = 0
        self.league_ppg = 22.5 + 0.55 * (self.league_ppg - 22.5)

    def advance_season(self, season: int) -> None:
        if self.season is None:
            self.season = int(season)
            return
        if season < self.season:
            raise DataContractError("online state cannot move backward in season")
        while self.season < int(season):
            self._regress_offseason()
            self.season += 1

    def features(self, game: dict[str, object]) -> dict[str, object]:
        season = int(game["season"])
        week = int(game["week"])
        self.advance_season(season)
        home_team = str(game["home_team"])
        away_team = str(game["away_team"])
        home = self._team(home_team)
        away = self._team(away_team)
        hfa = 0.0 if _neutral_site(game) else self.config.home_field_points
        home_points = self.league_ppg + home.offense - away.defense + hfa
        away_points = self.league_ppg + away.offense - home.defense

        return {
            "season": season,
            "week": week,
            "game_id": str(game["game_id"]),
            "gameday": game.get("gameday"),
            "away_team": away_team,
            "home_team": home_team,
            "online_home_points": float(home_points),
            "online_away_points": float(away_points),
            "online_margin": float(home_points - away_points),
            "online_total": float(home_points + away_points),
            "elo_diff_home": float(home.elo - away.elo),
            "offense_diff_home": float(home.offense - away.offense),
            "defense_diff_home": float(home.defense - away.defense),
            "net_eff_diff_home": float(
                (home.offense + home.defense) - (away.offense + away.defense)
            ),
            "margin_form_diff_home": float(home.margin_form - away.margin_form),
            "total_form_avg": float((home.total_form + away.total_form) / 2.0),
            "win_rate_diff_home": float(home.win_rate - away.win_rate),
            "volatility_avg": float((home.volatility + away.volatility) / 2.0),
            "sos_diff_home": float(home.sos_elo - away.sos_elo),
            "home_games": float(home.games),
            "away_games": float(away.games),
            "games_diff_home": float(home.games - away.games),
            "early_season": float(week <= 4),
        }

    def update(self, game: dict[str, object]) -> None:
        season = int(game["season"])
        self.advance_season(season)
        home_team = str(game["home_team"])
        away_team = str(game["away_team"])
        home = self._team(home_team)
        away = self._team(away_team)
        home_score = float(game["home_score"])
        away_score = float(game["away_score"])
        hfa = 0.0 if _neutral_site(game) else self.config.home_field_points

        predicted_home = self.league_ppg + home.offense - away.defense + hfa
        predicted_away = self.league_ppg + away.offense - home.defense
        home_residual = home_score - predicted_home
        away_residual = away_score - predicted_away

        alpha = self.config.score_alpha
        home.offense += alpha * home_residual
        away.defense -= alpha * home_residual
        away.offense += alpha * away_residual
        home.defense -= alpha * away_residual

        margin = home_score - away_score
        total = home_score + away_score
        form_alpha = self.config.form_alpha
        old_home_form = home.margin_form
        old_away_form = away.margin_form
        home.margin_form = (1.0 - form_alpha) * home.margin_form + form_alpha * margin
        away.margin_form = (1.0 - form_alpha) * away.margin_form - form_alpha * margin
        home.total_form = (1.0 - form_alpha) * home.total_form + form_alpha * total
        away.total_form = (1.0 - form_alpha) * away.total_form + form_alpha * total
        home.volatility = (
            (1.0 - form_alpha) * home.volatility
            + form_alpha * abs(margin - old_home_form)
        )
        away.volatility = (
            (1.0 - form_alpha) * away.volatility
            + form_alpha * abs(-margin - old_away_form)
        )

        if margin > 0:
            home_result, away_result = 1.0, 0.0
        elif margin < 0:
            home_result, away_result = 0.0, 1.0
        else:
            home_result = away_result = 0.5
        home.win_rate = (1.0 - form_alpha) * home.win_rate + form_alpha * home_result
        away.win_rate = (1.0 - form_alpha) * away.win_rate + form_alpha * away_result

        home_elo, away_elo = home.elo, away.elo
        home.sos_elo = (1.0 - form_alpha) * home.sos_elo + form_alpha * away_elo
        away.sos_elo = (1.0 - form_alpha) * away.sos_elo + form_alpha * home_elo
        elo_hfa = 0.0 if _neutral_site(game) else 45.0
        expected_home = 1.0 / (1.0 + 10.0 ** (-((home_elo - away_elo) + elo_hfa) / 400.0))
        multiplier = min(1.6, 1.0 + log1p(abs(margin)) / 6.0)
        delta = self.config.elo_k * multiplier * (home_result - expected_home)
        home.elo += delta
        away.elo -= delta

        self.league_ppg = 0.995 * self.league_ppg + 0.005 * (total / 2.0)
        home.games += 1
        away.games += 1


def build_online_state_features(
    schedules: pl.DataFrame,
    config: OnlineStateConfig,
) -> pl.DataFrame:
    """Build strict pre-week state features for every completed regular-season game."""

    require_columns(
        schedules,
        {
            "season",
            "week",
            "game_id",
            "game_type",
            "gameday",
            "away_team",
            "home_team",
            "away_score",
            "home_score",
        },
        "schedules",
    )
    games = completed_games(schedules).filter(pl.col("game_type") == "REG").sort(
        ["season", "week", "gameday", "game_id"]
    )
    if games.is_empty():
        raise DataContractError("online state requires completed regular-season games")

    engine = OnlineTeamState(config)
    rows: list[dict[str, object]] = []
    week_keys = games.select(["season", "week"]).unique(maintain_order=True)
    for season_value, week_value in week_keys.iter_rows():
        season = int(season_value)
        week = int(week_value)
        block = games.filter(
            (pl.col("season") == season) & (pl.col("week") == week)
        ).sort(["gameday", "game_id"])
        engine.advance_season(season)

        block_games = list(block.iter_rows(named=True))
        for game in block_games:
            row = engine.features(game)
            home_score = float(game["home_score"])
            away_score = float(game["away_score"])
            row["actual_home_margin"] = home_score - away_score
            row["actual_total"] = home_score + away_score
            rows.append(row)

        # All Week W features are frozen before any Week W result updates state.
        for game in block_games:
            engine.update(game)

    return pl.DataFrame(rows).sort(["season", "week", "game_id"])
