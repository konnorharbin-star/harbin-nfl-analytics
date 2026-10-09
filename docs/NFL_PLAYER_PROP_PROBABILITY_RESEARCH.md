# Player prop over/under probability research — empirical pregame errors

**Status: experiment only.** This produces statistical probabilities on fixed *synthetic* half-point thresholds. It does not pull sportsbook player-prop prices, prove a real betting edge, or modify betting actions.

## What we have changed

A mean projection of 70 receiving yards does not by itself tell us the chance of finishing over 74.5 yards. This experiment estimates uncertainty separately:

1. Choose mean-forecast windows/shrinkages on **2024 development games** using the existing `nfl.prop_challenger` module.
2. Re-create **2024 pregame** predicted statistics for games from only that season's prior weeks. Match actual later game results *after* the forecast; count missing players as zero (with source uncertainty noted). Standardize errors by a fixed scale depending on stat type and projected mean.
3. Construct independent empirical error distributions for challenger and rolling-three-game baseline, by player position where at least 80 historical errors exist, otherwise a market-wide pool where at least 160 exist.
4. For every hypothetical half-point threshold, compute `P(stat > line)` from the empirical error tail with Beta(1,1) smoothing and 1%-99% tail clipping. The computed probabilities decrease monotonically as the threshold increases; the under complement is one minus the over.
5. Evaluate **2025** outcomes with those fixed **2024** distributions. Report over/under Brier score, log loss, ten-bin expected calibration error, paired Brier difference versus a separately trained rolling-three baseline, and a **descriptive** week-clustered bootstrap interval. This diagnostic counts multiple highly correlated synthetic lines for each player; reported threshold-count `n` is *not* the number of statistically independent observations.
6. For future **2026** kickoffs, publish probabilities on the exact same precommitted synthetic grid, and append a first-seen timestamped (game, player, market, line) row only once. Never backfill after kickoff or rewrite original forecasts.

### Fixed non-bookmaker threshold grid

| Market | Research thresholds (not sportsbook quotes) |
|---|---|
| Receptions | 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5 |
| Receiving yards | 24.5, 39.5, 49.5, 64.5, 79.5, 99.5, 124.5 |
| Rushing yards | 19.5, 34.5, 49.5, 64.5, 79.5, 99.5, 124.5 |
| Passing yards | 149.5, 174.5, 199.5, 224.5, 249.5, 274.5, 299.5 |

The model does not invent odds, lines, bookmakers, player availability verification, or positive EV. A probability is labeled *research only* and has no tradable price.

### Published artifacts

- `docs/player_prop_probability_shadow.json`: development count; held-out *historical diagnostic* scores for challenger versus baseline; Brier/log-loss/calibration/uncertainty by market; forward research coverage; unproven edge statement.
- `docs/player_prop_probability_shadow.csv`: future player/market/threshold over/under probability curves without odds or wager directives.
- `history/player_prop_probability_forward.csv`: immutable earliest pregame research forecast at each fixed threshold.

### Limits and future requirement

The 2025 comparison is not pristine untouched evidence because previous analysis already inspected that season. 2024 hyperparameter selection and residual estimation share development data. The nonparametric distribution may miss playing-time uncertainty, injury reports, opponent quality, correlations and player-role changes. Low Brier score versus the rolling baseline still does **not** mean the model beats a sportsbook.

To establish a market edge, obtain legal **free** origin-timestamped two-sided player odds by player ID/book/threshold, confirm player role/availability at each timestamp, prospectively compare against a no-vig market-only benchmark, grade independent game outcomes and include push/void rules, and demonstrate robust incremental positive out-of-sample value before changing any recommendation policy.

**Zero automatic betting, paid APIs, bookmaker integration or staking.**
