# Stage 3 Market History, Provider, and CLV Contract

Stage 3 introduces sportsbook observations without weakening the independent football-model boundary.

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

A backtest therefore cannot use a closing price, later line move, or opposite-side quote that was not actually available at the simulated decision time.

## Historical provider adapter

`nfl/odds_api.py` implements an optional adapter for The Odds API historical NFL endpoint. The adapter is intentionally fail-closed:

- `THE_ODDS_API_KEY` is required for network retrieval;
- historical requests must use timezone-aware decision timestamps;
- the provider's returned snapshot timestamp is stored as the observation time;
- source event IDs, bookmaker keys, provider commence times, and source last-update timestamps are preserved;
- full NFL team names are mapped to nflverse team identifiers;
- event-to-schedule matching requires the same home/away pairing and a provider commence date within one calendar day of nflverse `gameday`;
- ambiguous schedule matches raise an error rather than guessing;
- API responses are cached locally using a query hash that never includes the API key.

The adapter requests featured NFL markets only: moneyline (`h2h`), spreads, and totals. Historical provider access is optional and is not required for the core football model or unit tests.

## Market comparison and grading

The market backtest layer accepts an already-fitted football probability distribution and already-produced football projections. Sportsbook prices are then used only to compute:

- implied probability;
- proportional no-vig probability;
- football-model probability edge;
- executable-price expected value;
- realized historical win/loss/push result;
- realized net units.

Quote provenance and decision timestamps are preserved on every comparison and graded row.

## Closing-line value

`nfl/clv.py` compares a decision quote with a later pre-kickoff quote from the same provider, sportsbook, market, and side.

Probability CLV is:

```text
closing_no_vig_probability - decision_no_vig_probability
```

Positive probability CLV means the market's no-vig probability moved toward the selected side after the simulated decision.

Line CLV is side-aware:

- spread: `decision_line - closing_line`;
- total over: `closing_line - decision_line`;
- total under: `decision_line - closing_line`.

Closing observations are evidence only. They never enter the football score projection or probability fit.

## Line shopping and chronological rule validation

`nfl/market_validation.py` prevents a multi-book backtest from counting the same modeled opportunity once per sportsbook. For each `game_id` and `market_type`, it retains only the available quote with the highest model expected value, using deterministic tie breakers.

Research qualification thresholds are then handled chronologically:

1. choose an explicit probability-edge grid and expected-value grid;
2. evaluate those rules on one validation season only;
3. require a minimum number of validation bets;
4. reject the entire rule family if the selected validation rule has non-positive units or non-positive average probability CLV;
5. freeze the selected thresholds;
6. score a later untouched holdout season exactly once;
7. require minimum holdout volume, positive units/ROI, and positive average probability CLV for the research candidate to pass.

A passing research candidate is not a production staking rule. Portfolio sizing, drawdown controls, live/shadow evidence, and release gates remain later stages.

## Current limitation

The adapter and validation machinery are implemented and covered by synthetic tests, but the repository does not contain a paid historical provider credential. Therefore no historical ROI/CLV claim is made from The Odds API data yet. A live provider audit becomes valid only after `THE_ODDS_API_KEY` is supplied and archived point-in-time snapshots are successfully ingested.
