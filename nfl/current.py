"""Current/upcoming NFL projections from the independent football model.

The canonical score baseline is always emitted. The validated quarterback residual
structures are also emitted with explicit release labels: margin remains RESEARCH and
total remains SHADOW while forward evidence accumulates. Sportsbook prices are not
accepted by this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import polars as pl

from .contracts import SCHEDULE_REQUIRED, DataContractError, require_columns
from .data import NFLDataClient
from .qb_dataset import build_qb_walkforward_dataset
from .qb_state import QB_PRIOR_DROPBACKS, qb_matchup_signals, team_qb_state
from .qb_validated import ValidatedQBAdjustment
from .ratings import fit_pregame_fair_score

QB_MARGIN_RELEASE_STATE = "RESEARCH"
QB_TOTAL_RELEASE_STATE = "SHADOW"


@dataclass(frozen=True)
class CurrentProjectionAudit:
    season: int
    week: int
    games: int
    training_seasons: tuple[int, ...]
    qb_training_games: int
    qb_margin_release_state: str
    qb_total_release_state: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def unplayed_regular_games(
    schedules: pl.DataFrame,
    season: int,
    week: int,
) -> pl.DataFrame:
    """Return regular-season target games that do not yet have final scores."""

    require_columns(schedules, SCHEDULE_REQUIRED, "schedules")
    return schedules.filter(
        (pl.col("season") == season)
        & (pl.col("week") == week)
        & (pl.col("game_type") == "REG")
        & (pl.col("home_score").is_null() | pl.col("away_score").is_null())
    ).sort("game_id")


def assert_target_week_schedule_integrity(
    targets: pl.DataFrame,
    season: int,
    week: int,
) -> None:
    """Fail closed when an upcoming regular-season slate is structurally impossible.

    An NFL team can appear at most once in a regular-season week. This check protects
    the live projection path from publishing projections when an upstream future
    schedule feed contains duplicate team-week assignments or malformed games.
    """

    require_columns(targets, {"game_id", "away_team", "home_team"}, "target_games")
    if targets.is_empty():
        raise DataContractError(
            f"no unplayed regular-season games found for {season} week {week}"
        )

    duplicate_game_ids = (
        targets.group_by("game_id").len().filter(pl.col("len") > 1).get_column("game_id")
    )
    if duplicate_game_ids.len():
        values = sorted(str(value) for value in duplicate_game_ids.to_list())
        raise DataContractError(
            "target week contains duplicate game IDs: " + ", ".join(values)
        )

    self_games = targets.filter(pl.col("home_team") == pl.col("away_team"))
    if self_games.height:
        values = sorted(str(value) for value in self_games.get_column("game_id").to_list())
        raise DataContractError(
            "target week contains game(s) with the same home and away team: "
            + ", ".join(values)
        )

    teams = pl.concat(
        [
            targets.select(pl.col("away_team").alias("team")),
            targets.select(pl.col("home_team").alias("team")),
        ],
        how="vertical",
    )
    duplicates = teams.group_by("team").len().filter(pl.col("len") > 1).sort("team")
    if duplicates.height:
        values = [str(value) for value in duplicates.get_column("team").to_list()]
        raise DataContractError(
            f"{season} week {week} schedule assigns team(s) to multiple games: "
            + ", ".join(values)
        )


def next_unplayed_regular_week(schedules: pl.DataFrame, season: int) -> int:
    """Return the earliest regular-season week containing any unplayed game."""

    require_columns(schedules, SCHEDULE_REQUIRED, "schedules")
    future = schedules.filter(
        (pl.col("season") == season)
        & (pl.col("game_type") == "REG")
        & (pl.col("home_score").is_null() | pl.col("away_score").is_null())
    )
    if future.is_empty():
        raise DataContractError(f"no unplayed regular-season games found for {season}")
    return int(future.get_column("week").min())


def build_current_qb_projection(
    schedules: pl.DataFrame,
    player_stats: pl.DataFrame,
    season: int,
    week: int,
    adjustment: ValidatedQBAdjustment,
    *,
    score_ridge: float = 8.0,
    qb_prior_dropbacks: float = QB_PRIOR_DROPBACKS,
) -> pl.DataFrame:
    """Project unplayed games using only state known before the selected week."""

    if week < 2:
        raise DataContractError("QB-assisted current projection requires week >= 2")
    targets = unplayed_regular_games(schedules, season, week)
    assert_target_week_schedule_integrity(targets, season, week)

    qb_state = team_qb_state(
        player_stats,
        season,
        week,
        prior_dropbacks=qb_prior_dropbacks,
    )
    qb_map = {row["team"]: row for row in qb_state.iter_rows(named=True)}
    score_model = fit_pregame_fair_score(schedules, season, week, ridge=score_ridge)

    rows: list[dict[str, object]] = []
    for game in targets.iter_rows(named=True):
        home_team = str(game["home_team"])
        away_team = str(game["away_team"])
        if home_team not in qb_map or away_team not in qb_map:
            raise DataContractError(
                f"missing last-observed QB state for {away_team} at {home_team}"
            )

        home_qb = qb_map[home_team]
        away_qb = qb_map[away_team]
        projection = score_model.project(home_team, away_team)
        row: dict[str, object] = {
            "season": season,
            "week": week,
            "game_id": str(game["game_id"]),
            "gameday": game["gameday"],
            "away_team": away_team,
            "home_team": home_team,
            "home_qb_proxy_id": str(home_qb["qb_proxy_id"]),
            "home_qb_proxy_name": home_qb["qb_proxy_name"],
            "home_qb_proxy_last_week": int(home_qb["qb_proxy_last_week"]),
            "away_qb_proxy_id": str(away_qb["qb_proxy_id"]),
            "away_qb_proxy_name": away_qb["qb_proxy_name"],
            "away_qb_proxy_last_week": int(away_qb["qb_proxy_last_week"]),
            "baseline_home_points": projection.home_points,
            "baseline_away_points": projection.away_points,
            "baseline_home_margin": projection.home_margin,
            "baseline_total": projection.total,
        }
        row.update(qb_matchup_signals(home_qb, away_qb))
        rows.append(row)

    frame = pl.DataFrame(rows).sort("game_id")
    adjusted = adjustment.apply(frame)
    return adjusted.with_columns(
        pl.lit("CANONICAL").alias("baseline_release_state"),
        pl.lit(QB_MARGIN_RELEASE_STATE).alias("qb_margin_release_state"),
        pl.lit(QB_TOTAL_RELEASE_STATE).alias("qb_total_release_state"),
    )


def fit_validated_qb_adjustment(
    schedules: pl.DataFrame,
    player_stats: pl.DataFrame,
    training_seasons: tuple[int, ...],
    *,
    start_week: int = 5,
    end_week: int = 18,
    score_ridge: float = 8.0,
    qb_prior_dropbacks: float = QB_PRIOR_DROPBACKS,
) -> ValidatedQBAdjustment:
    """Refit the frozen QB structures on completed historical football seasons."""

    if not training_seasons:
        raise ValueError("training_seasons must not be empty")
    frames = [
        build_qb_walkforward_dataset(
            schedules,
            player_stats.filter(pl.col("season") == season),
            season,
            start_week=start_week,
            end_week=end_week,
            score_ridge=score_ridge,
            qb_prior_dropbacks=qb_prior_dropbacks,
        )
        for season in training_seasons
    ]
    training = pl.concat(frames, how="vertical_relaxed").sort(
        ["season", "week", "game_id"]
    )
    return ValidatedQBAdjustment().fit(training)


def run_current_projection(
    season: int,
    week: int | None = None,
    *,
    training_seasons: tuple[int, ...] | None = None,
    score_ridge: float = 8.0,
    qb_prior_dropbacks: float = QB_PRIOR_DROPBACKS,
    client: NFLDataClient | None = None,
    refresh: bool = False,
) -> tuple[pl.DataFrame, CurrentProjectionAudit]:
    """Load sources, fit frozen QB structures, and project the next selected week."""

    training = training_seasons or tuple(range(season - 4, season))
    if not training or max(training) >= season:
        raise ValueError("training seasons must all precede the projection season")

    source = client or NFLDataClient()
    schedule_seasons = sorted({min(training) - 1, *training, season})
    schedules = source.load_schedules(schedule_seasons, refresh=refresh)
    target_week = next_unplayed_regular_week(schedules, season) if week is None else week
    targets = unplayed_regular_games(schedules, season, target_week)
    assert_target_week_schedule_integrity(targets, season, target_week)

    stats_seasons = sorted({*training, season})
    player_stats = source.load_player_stats(stats_seasons, refresh=refresh)
    adjustment = fit_validated_qb_adjustment(
        schedules,
        player_stats,
        training,
        score_ridge=score_ridge,
        qb_prior_dropbacks=qb_prior_dropbacks,
    )
    projection = build_current_qb_projection(
        schedules,
        player_stats.filter(pl.col("season") == season),
        season,
        target_week,
        adjustment,
        score_ridge=score_ridge,
        qb_prior_dropbacks=qb_prior_dropbacks,
    )
    audit = CurrentProjectionAudit(
        season=season,
        week=target_week,
        games=projection.height,
        training_seasons=training,
        qb_training_games=adjustment.training_rows,
        qb_margin_release_state=QB_MARGIN_RELEASE_STATE,
        qb_total_release_state=QB_TOTAL_RELEASE_STATE,
    )
    return projection, audit
