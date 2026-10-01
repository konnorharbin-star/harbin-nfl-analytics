# Stage 8 — NFL recent-form PBP research

Stage 8 ports one important NCAA modeling concept without importing an NCAA coefficient: a separate recent-form efficiency state alongside the season-to-date football state.

The canonical NFL fair score is unchanged. Recent form is a research/shadow layer until independent NFL evidence is strong enough to justify any promotion.

## Feature state

For every completed historical game, eligible scrimmage plays are reduced to game-level offense-versus-defense observations for:

- EPA per play;
- success rate;
- pass EPA per dropback;
- rush EPA per attempt;
- explosive-play rate;
- early-down EPA.

Each team then receives separate offense and defense-allowed exponentially weighted moving averages. Target-week and future PBP are excluded by rebuilding each target week from exact completed prior regular-season game IDs.

For every metric, the research dataset emits two matchup signals:

```text
home_matchup = home_recent_offense - away_recent_defense_allowed
away_matchup = away_recent_offense - home_recent_defense_allowed

margin_signal = home_matchup - away_matchup
total_signal  = home_matchup + away_matchup
```

Sportsbook prices are not inputs.

## Development selection

The decay rate, feature subset, ridge regularization and residual blend were selected only from 2022-2025 historical development data. Expanding chronological evaluation folds were 2023, 2024 and 2025.

Candidate recent-form alphas were `0.20`, `0.35` and `0.50`. The residual selector also tested smaller/larger football feature sets, ridge values and blend weights. A real zero-weight fallback was included.

A nonzero candidate was eligible only when:

1. aggregate MAE improved;
2. aggregate RMSE improved; and
3. at least two of three season folds improved both MAE and RMSE.

These folds are development evidence, not a pristine promotion holdout, because 2022-2025 results have already informed NFL model design elsewhere in the repository.

### Margin result

Recent form did not pass the development gate for margin. The selector correctly returned the disabled state:

- blend weight: `0.00`;
- aggregate MAE: `10.1966 -> 10.1966`;
- aggregate RMSE: `13.0138 -> 13.0138`;
- shadow candidate: `false`.

Margin recent form therefore remains disabled.

### Total result

A small totals candidate passed the development selector:

- recent alpha: `0.20`;
- features: EPA/play + success rate;
- ridge alpha: `10.0`;
- residual blend: `0.50`;
- games across 2023-2025 evaluation folds: `624`;
- aggregate total MAE: `10.4926 -> 10.4731`;
- aggregate total RMSE: `13.4427 -> 13.3822`;
- positive folds: `2 / 3`.

Per-season behavior was mixed:

| Season | Games | MAE baseline | MAE adjusted | RMSE baseline | RMSE adjusted | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2023 | 208 | 10.8663 | 10.7923 | 13.7449 | 13.5918 | improved |
| 2024 | 208 | 10.1277 | 10.1574 | 13.2747 | 13.2870 | regressed slightly |
| 2025 | 208 | 10.4839 | 10.4697 | 13.3034 | 13.2653 | improved |

This is enough to freeze a shadow specification, not enough to promote it into the canonical total.

## Frozen 2026 shadow specification

Before reading 2026 shadow results, the totals structure was frozen as:

```text
recent_alpha = 0.20
features = [epa_per_play, success_rate]
ridge_alpha = 10.0
blend_weight = 0.50
training_seasons = 2022-2025
```

The fixed structure is refit on all 2022-2025 development rows and then evaluated on completed 2026 games without retuning.

The first available out-of-time shadow sample contained 16 completed Week 3 games. Its point estimates were:

- baseline total MAE: `12.3192`;
- recent-form shadow total MAE: `11.9330`;
- MAE point improvement: `+0.3863` points;
- baseline total RMSE: `13.8572`;
- recent-form shadow total RMSE: `13.2730`;
- RMSE point improvement: `+0.5843` points.

The paired-bootstrap evidence is not conclusive:

- MAE 95% improvement interval: `[-0.4046, 1.1192]`;
- RMSE 95% improvement interval: `[-0.2264, 1.2427]`;
- minimum shadow sample: `128` games;
- current shadow sample: `16` games;
- evidence status: `SHADOW_INSUFFICIENT_SAMPLE`;
- promotion eligible: `false`.

The positive point estimate therefore is not treated as proof of improvement. The specification remains frozen and separate from the canonical total while 2026 evidence accumulates.

## Release discipline

`nfl/recent_form_shadow.py` never overwrites `baseline_total`. It adds separate shadow fields only. The paired-bootstrap gate can return promotion evidence only when the minimum sample is met, both point improvements are positive, and the lower confidence bounds for both MAE and RMSE improvement are above zero.

The Stage 8 source workflows also use shell `pipefail` and bounded retries. A Python/source failure cannot be hidden by a successful `tee` process and reported as a false-green audit.

## Current decision

- recent-form margin adjustment: **disabled**;
- recent-form total adjustment: **frozen SHADOW candidate**;
- canonical fair score: **unchanged**;
- 2026 outcomes: **not used for retuning**;
- promotion: **not eligible**.
