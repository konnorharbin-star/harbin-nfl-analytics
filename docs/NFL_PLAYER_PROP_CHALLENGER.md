# Free NFL player-stat challenger — forward shadow, not an automatic bet

## Why this exists

The existing NFL football model has not demonstrated stable incremental edge over the no-vig market. A full-game total or winning-margin forecast is **not** an adequate player-prop probability model. This challenger introduces a separate, strictly time-ordered player-stat projection built from real NFL weekly statistics.

### Current covered markets

- Receptions, receiving yards, rushing yards, and passing yards.
- Other market families are listed in `docs/market_universe.json`, but they are not priced or given probabilities without their own training/test/evidence.
- The market scanner still lacks a free source of fully origin-timestamped, two-sided sportsbook player props. **These mean forecasts are not bet selections and supply no validated fair price or expected value.**

### Predictive experiment

1. Load freely available nflverse weekly player statistics and regular-season schedule through the repository's existing `NFLDataClient`. Reject mismatching season/week/game ID/team, unsupported positions, duplicate player-game identities, invalid stats and uncompleted game histories.
2. Forecast one target week at a time using *only earlier weeks of that season*. Require at least three observed prior games and a recent appearance on the same team. Never decide whether a player was eligible using the target game's participation data.
3. Compare a three-game rolling-mean baseline against a challenger that combines the player's prior 3, 5 or 8 games with a historical team-position mean. Shrinkage weights {0,1,3,6} are selected by mean absolute error on the **2024 development season only**.
4. Report 2025 player/game market-diagnostic MAE and RMSE for the chosen 2024 configuration, and compare to the rolling-mean baseline. **2025 was examined in broader NFL work and is not a pristine unseen holdout.**
5. Generate 2026 research forecasts only for games whose kickoffs remain in the future. Mark active roster/starting status unverified, price/line unavailable, probability uncalibrated, EV unproven and recommendation `UNPRICED_RESEARCH_ONLY`.
6. Append each (game ID, player ID, stat market) to `history/player_prop_forward.csv` **once**, before kickoff. Future runs must not overwrite that original forecast, even after the game or if the model updates.
7. If the free player feed fails, publish a `BLOCKED_PLAYER_STAT_SOURCE_OR_SCHEMA` status. Never silently reuse old player forecasts or fail the established full-game model publication pipeline.

### Why the method cannot yet claim an edge

- Expected stat values are **not** full distributions; threshold probabilities for over/under props need separately tested count/hurdle distributions, injury/participation and role models.
- An inactive player can score 0 in the evaluation; that prevents postgame participant-selection bias, but a true pregame injury report is still needed to know likely starters and their workload.
- No market-only NFL prop benchmark can be run without original pregame, two-sided, per-book, per-player verified odds.
- Improved MAE against a naive rolling average is not evidence of outperforming sportsbooks; even improvements on 2025 can be noise/previously examined data.
- The next real milestone is a **prospective 2026** locked comparison versus a calibrated no-vig prop market on the same market instrument, with prices executable only after independent manual checking.

### Artifacts

- `docs/player_prop_shadow.json` — status, data quality, development/diagnostic accuracy and prospective coverage.
- `docs/player_prop_shadow.csv` — per-player, per-stat expected-value projections (stat **means**, not betting EV).
- `history/player_prop_forward.csv` — first-seen, immutable research forecasts.

These are built by `python run_player_prop_shadow.py` after the canonical free NFL model and committed through the existing read-only/no-wager publication workflow.

**Always free. Never connect to a bookmaker, fund an account, place a bet, or present an unpriced forecast as a profitable wager.**
