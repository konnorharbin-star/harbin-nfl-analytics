"""Stage 2 fair-score and advanced-feature audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .advanced import pregame_pbp, pregame_team_pbp_features
from .data import NFLDataClient
from .ratings import VALIDATED_PRIOR_SEASON_WEIGHT, fit_pregame_fair_score


@dataclass(frozen=True)
class Stage2Audit:
    season: int
    target_week: int
    pregame_pbp_rows: int
    feature_team_count: int
    fair_score_team_count: int
    fair_score_training_rows: int
    fair_score_training_weight: float
    prior_season_weight: float
    league_points: float
    home_field: float
    residual_std: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def run_stage2_audit(
    season: int,
    week: int,
    *,
    client: NFLDataClient | None = None,
    refresh: bool = False,
) -> Stage2Audit:
    """Validate real-source PBP features and the canonical pregame baseline."""

    source = client or NFLDataClient()
    schedules = source.load_schedules([season - 1, season], refresh=refresh)
    pbp = source.load_pbp(season, refresh=refresh)

    history = pregame_pbp(pbp, season, week)
    features = pregame_team_pbp_features(pbp, season, week)
    model = fit_pregame_fair_score(schedules, season, week)

    if features.get_column("team").n_unique() < 2:
        raise RuntimeError("Stage 2 PBP audit found fewer than two teams")
    if len(model.teams) < 2:
        raise RuntimeError("Stage 2 fair-score audit found fewer than two teams")
    if model.league_points is None or model.home_field is None or model.residual_std is None:
        raise RuntimeError("Stage 2 fair-score model did not expose fitted diagnostics")

    return Stage2Audit(
        season=season,
        target_week=week,
        pregame_pbp_rows=history.height,
        feature_team_count=features.get_column("team").n_unique(),
        fair_score_team_count=len(model.teams),
        fair_score_training_rows=model.training_rows,
        fair_score_training_weight=model.training_weight,
        prior_season_weight=VALIDATED_PRIOR_SEASON_WEIGHT,
        league_points=model.league_points,
        home_field=model.home_field,
        residual_std=model.residual_std,
    )
