"""Real-source walk-forward reconstruction audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .data import NFLDataClient
from .dataset import build_walkforward_dataset
from .evaluation import summarize_baseline


@dataclass(frozen=True)
class WalkForwardAudit:
    season: int
    start_week: int
    end_week: int
    games: int
    feature_columns: int
    margin_mae: float
    margin_rmse: float
    margin_bias: float
    total_mae: float
    total_rmse: float
    total_bias: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def run_walkforward_audit(
    season: int,
    *,
    start_week: int = 5,
    end_week: int = 10,
    client: NFLDataClient | None = None,
    refresh: bool = False,
) -> WalkForwardAudit:
    """Reconstruct several historical weeks and report canonical baseline error."""

    source = client or NFLDataClient()
    schedules = source.load_schedules([season - 1, season], refresh=refresh)
    pbp = source.load_pbp(season, refresh=refresh)
    dataset = build_walkforward_dataset(
        schedules,
        pbp,
        season,
        start_week=start_week,
        end_week=end_week,
    )
    metrics = summarize_baseline(dataset)
    feature_columns = len([name for name in dataset.columns if name.endswith("_matchup_advantage")])

    return WalkForwardAudit(
        season=season,
        start_week=start_week,
        end_week=end_week,
        games=metrics.games,
        feature_columns=feature_columns,
        margin_mae=metrics.margin_mae,
        margin_rmse=metrics.margin_rmse,
        margin_bias=metrics.margin_bias,
        total_mae=metrics.total_mae,
        total_rmse=metrics.total_rmse,
        total_bias=metrics.total_bias,
    )
