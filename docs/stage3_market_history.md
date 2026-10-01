# Stage 3 Market History Contract

This stage introduces historical sportsbook observations without weakening the independent football-model boundary.

## Required quote provenance

Every stored historical quote must contain:

- `game_id`
- `market_type` (`moneyline`, `spread`, or `total`)
- `side`
- `line` where applicable
- `american_odds`
- `provider`
- `book`
- `captured_at`
- `snapshot_id`
- `source_event_id`

The provider/snapshot/event fields are not cosmetic. They make it possible to reproduce which executable price was used in a simulated decision and prevent unrelated sides from being combined into a synthetic market that never existed.

## Point-in-time selection

Historical evaluation supplies one explicit `decision_time` per game. The selector:

1. rejects every quote captured after that decision time;
2. optionally rejects quotes older than a configured maximum age;
3. requires both sides of a two-way market to come from the same provider snapshot;
4. requires spread lines to be exact opposites and totals to share one total line;
5. falls back to the latest earlier complete snapshot when the newest capture is incomplete.

This means a backtest cannot use a closing price, later line move, or opposite-side quote that was not actually available at the simulated decision time.

## Market comparison and grading

The market backtest layer accepts an already-fitted football probability distribution and already-produced football projections. Sportsbook prices are then used only to compute:

- implied probability;
- proportional no-vig probability;
- football-model probability edge;
- executable-price expected value;
- realized historical win/loss/push result;
- realized net units.

Quote provenance and decision timestamps are preserved on every comparison and graded row.

Research summaries require explicit probability-edge and expected-value thresholds. There are intentionally no default betting thresholds and no automatic promotion claim.

## Still required before production research

This contract does not itself supply a historical odds vendor. A source adapter must prove that `captured_at`, provider event IDs, sportsbook identity, and snapshot identity are genuine point-in-time records before those rows are admitted to chronological profitability or CLV studies.

The next evidence layer should add a source adapter plus chronological edge-bucket / ROI / CLV evaluation. Thresholds must be selected on earlier data and scored on later untouched data.
