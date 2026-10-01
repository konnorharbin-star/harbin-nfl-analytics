"""Prospective attachment of the frozen recent-form totals shadow model.

This module is deliberately downstream of the canonical fair score. It rebuilds
current-week recent-form state from completed prior-week PBP, refits the already-frozen
2022-2025 residual structure, and emits separate SHADOW columns. It never replaces the
canonical total used by market intelligence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import polars as pl

from .contracts import DataContractError
from .current import assert_target_week_schedule_integrity, unplayed_regular_games
from .dataset import _pbp_for_games, _regular_season_history
from .recent_form import recent_matchup_signals, team_recent_pbp_features
from .recent_form_dataset import build_recent_form_walkforward_dataset
from .recent_form_shadow import (
    FROZEN_BLEND_WEIGHT,
    FROZEN_FEATURE_SET,
    FROZEN_RECENT_ALPHA,
    FROZEN_RIDGE_ALPHA,
    FROZEN_SELECTION_SEASONS,
    FROZEN_TOTAL_FEATURES,
    apply_frozen_recent_total,
    fit_frozen_recent_total,
)

RECENT_FORM_CURRENT_RELEASE_STATE = "SHADOW"


@dataclass(frozen=True)
class CurrentRecentFormAudit:
    season: int
    week: int
    games: int
    covered_games: int
    coverage: float
    training_seasons: tuple[int, ...]
    training_games: int
    recent_alpha: float
    feature_set: tuple[str, ...]
    ridge_alpha: float
    blend_weight: float
    release_state: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_current_recent_form_signals(
    schedules: pl.DataFrame,
    pbp: pl.DataFrame,
    season: int,
    week: int,
) -> pl.DataFrame:
    """Build strict pregame recent-form signals for one unplayed NFL week."""

    if week < 2:
        raise DataContractError("recent-form current projection requires week >= 2")
    targets = unplayed_regular_games(schedules, season, week)
    assert_target_week_schedule_integrity(targets, season, week)
    history = _regular_season_history(schedules, season, week)
    if history.is_empty():
        raise DataContractError("recent-form current projection has no completed prior games")

    history_ids = history.get_column("game_id").cast(pl.String).to_list()
    history_pbp = _pbp_for_games(pbp.filter(pl.col("season") == season), history_ids)
    features = team_recent_pbp_features(history_pbp, alpha=FROZEN_RECENT_ALPHA)
    feature_map = {
        str(row["team"]): row for row in features.iter_rows(named=True)
    }

    rows: list[dict[str, object]] = []
    for game in targets.iter_rows(named=True):
        home_team = str(game["home_team"])
        away_team = str(game["away_team"])
        if home_team not in feature_map or away_team not in feature_map:
            raise DataContractError(
                f"missing current recent-form PBP state for {away_team} at {home_team}"
            )
        signals = recent_matchup_signals(feature_map[home_team], feature_map[away_team])
        row: dict[str, object] = {
            "game_id": str(game["game_id"]),
            "home_recent_form_games": int(feature_map[home_team]["recent_off_games"]),
            "away_recent_form_games": int(feature_map[away_team]["recent_off_games"]),
        }
        for feature in FROZEN_TOTAL_FEATURES:
            row[feature] = signals[feature]
        rows.append(row)
    return pl.DataFrame(rows).sort("game_id")


def fit_frozen_current_recent_total(
    schedules: pl.DataFrame,
    pbp: pl.DataFrame,
    *,
    training_seasons: tuple[int, ...] = FROZEN_SELECTION_SEASONS,
    score_ridge: float = 8.0,
):
    """Refit the frozen recent-form structure on its fixed 2022-2025 training window."""

    if training_seasons != FROZEN_SELECTION_SEASONS:
        raise ValueError("current recent-form training seasons are frozen to 2022-2025")
    frames = [
        build_recent_form_walkforward_dataset(
            schedules,
            pbp.filter(pl.col("season") == training_season),
            training_season,
            recent_alpha=FROZEN_RECENT_ALPHA,
            start_week=5,
            end_week=18,
            score_ridge=score_ridge,
        )
        for training_season in training_seasons
    ]
    training = pl.concat(frames, how="vertical_relaxed").sort(
        ["season", "week", "game_id"]
    )
    return fit_frozen_recent_total(training), training.height


def attach_current_recent_form_shadow(
    projection: pl.DataFrame,
    schedules: pl.DataFrame,
    pbp: pl.DataFrame,
    *,
    season: int,
    week: int,
    score_ridge: float = 8.0,
) -> tuple[pl.DataFrame, CurrentRecentFormAudit]:
    """Attach the frozen totals SHADOW to current projections without replacing baseline."""

    if season != 2026:
        raise ValueError("the frozen recent-form shadow specification is scoped to 2026")
    signals = build_current_recent_form_signals(schedules, pbp, season, week)
    joined = projection.join(signals, on="game_id", how="left")
    missing = joined.filter(
        pl.any_horizontal([pl.col(name).is_null() for name in FROZEN_TOTAL_FEATURES])
    )
    if missing.height:
        raise DataContractError(
            f"recent-form current signals missing for {missing.height} projected game(s)"
        )

    model, training_games = fit_frozen_current_recent_total(
        schedules,
        pbp,
        score_ridge=score_ridge,
    )
    scored = apply_frozen_recent_total(model, joined).with_columns(
        pl.lit(FROZEN_RECENT_ALPHA).alias("recent_form_alpha"),
        pl.lit(FROZEN_RIDGE_ALPHA).alias("recent_form_ridge_alpha"),
        pl.lit(FROZEN_BLEND_WEIGHT).alias("recent_form_blend_weight"),
    )
    covered = scored.filter(pl.col("recent_form_shadow_total").is_not_null()).height
    audit = CurrentRecentFormAudit(
        season=season,
        week=week,
        games=scored.height,
        covered_games=covered,
        coverage=covered / scored.height if scored.height else 0.0,
        training_seasons=FROZEN_SELECTION_SEASONS,
        training_games=training_games,
        recent_alpha=FROZEN_RECENT_ALPHA,
        feature_set=FROZEN_FEATURE_SET,
        ridge_alpha=FROZEN_RIDGE_ALPHA,
        blend_weight=FROZEN_BLEND_WEIGHT,
        release_state=RECENT_FORM_CURRENT_RELEASE_STATE,
    )
    return scored, audit


def blocked_recent_form_shadow(projection: pl.DataFrame) -> pl.DataFrame:
    """Emit explicit blocked shadow columns while preserving canonical projections."""

    return projection.with_columns(
        pl.lit(None, dtype=pl.Float64).alias("recent_form_total_adjustment"),
        pl.lit(None, dtype=pl.Float64).alias("recent_form_shadow_total"),
        pl.lit("BLOCKED").alias("recent_form_total_release_state"),
        pl.lit(FROZEN_RECENT_ALPHA).alias("recent_form_alpha"),
        pl.lit(FROZEN_RIDGE_ALPHA).alias("recent_form_ridge_alpha"),
        pl.lit(FROZEN_BLEND_WEIGHT).alias("recent_form_blend_weight"),
    )
