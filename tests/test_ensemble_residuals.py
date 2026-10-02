import numpy as np
import polars as pl

from nfl.ensemble_residuals import (
    _whole_week_split,
    evaluate_ensemble_fold,
    evaluate_rolling_ensemble,
)


def _dataset(*, perfect_baseline: bool = False) -> pl.DataFrame:
    rng = np.random.default_rng(17)
    rows: list[dict[str, float | int | str]] = []
    for season in (2022, 2023, 2024, 2025):
        for week in range(5, 19):
            for game in range(8):
                signal = float(rng.normal())
                baseline_margin = float(rng.normal(0.0, 5.0))
                baseline_total = float(rng.normal(45.0, 4.0))
                margin_residual = 0.0 if perfect_baseline else 2.5 * (signal > 0.4)
                total_residual = 0.0 if perfect_baseline else -2.0 * (signal < -0.3)
                row: dict[str, float | int | str] = {
                    "season": season,
                    "week": week,
                    "game_id": f"{season}-{week}-{game}",
                    "baseline_home_margin": baseline_margin,
                    "baseline_total": baseline_total,
                    "actual_home_margin": baseline_margin + margin_residual,
                    "actual_total": baseline_total + total_residual,
                    "margin_residual": margin_residual,
                    "total_residual": total_residual,
                    "home_off_plays": 180 + week + game,
                    "away_off_plays": 176 + week + game,
                    "home_def_plays": 182 + week + game,
                    "away_def_plays": 179 + week + game,
                }
                for name in (
                    "epa_per_play_matchup_advantage",
                    "success_rate_matchup_advantage",
                    "pass_epa_per_dropback_matchup_advantage",
                    "rush_epa_per_attempt_matchup_advantage",
                    "explosive_rate_matchup_advantage",
                    "early_down_epa_matchup_advantage",
                ):
                    row[name] = (
                        signal
                        if name == "epa_per_play_matchup_advantage"
                        else float(rng.normal(scale=0.2))
                    )
                rows.append(row)
    return pl.DataFrame(rows)


def test_whole_week_split_keeps_tune_after_core() -> None:
    core, tune = _whole_week_split(_dataset().filter(pl.col("season") < 2024))
    core_max = max(zip(core["season"].to_list(), core["week"].to_list(), strict=True))
    tune_min = min(zip(tune["season"].to_list(), tune["week"].to_list(), strict=True))
    assert core_max < tune_min


def test_perfect_baseline_forces_zero_weight_fallback() -> None:
    result = evaluate_rolling_ensemble(_dataset(perfect_baseline=True), target="margin")
    assert result.eligible is False
    assert result.positive_folds == 0
    assert result.adjusted_mae == result.baseline_mae
    assert all(fold.selected_spec is None for fold in result.folds)


def test_evaluation_outcomes_do_not_select_same_season_spec() -> None:
    frame = _dataset()
    before = evaluate_ensemble_fold(frame, season=2025, target="margin")
    mutated = frame.with_columns(
        pl.when(pl.col("season") == 2025)
        .then(pl.col("actual_home_margin") + 100.0)
        .otherwise(pl.col("actual_home_margin"))
        .alias("actual_home_margin"),
        pl.when(pl.col("season") == 2025)
        .then(pl.col("margin_residual") + 100.0)
        .otherwise(pl.col("margin_residual"))
        .alias("margin_residual"),
    )
    after = evaluate_ensemble_fold(mutated, season=2025, target="margin")
    assert before.selected_spec == after.selected_spec
