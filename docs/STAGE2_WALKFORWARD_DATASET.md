# Stage 2 — Chronological Modeling Dataset

This layer converts the fair-score baseline and play-by-play feature state into one row per historical target game, reconstructed exactly from information available before that target week.

## Information boundary

For a target game in season `S`, week `W`:

- target games are completed `REG` games in `S/W`;
- fair-score ratings use completed `REG` games from season `S` with week `< W`;
- PBP features are restricted by the exact game IDs of those earlier regular-season games;
- target-week PBP is never eligible for predictors;
- preseason games are excluded from the regular-season model state;
- sportsbook prices are not loaded or accepted.

The first implementation intentionally uses same-season history only. Cross-season priors, offseason regression, roster changes and quarterback transitions require separate validation before they can influence the score engine.

## Game-level features

For each target game, the dataset stores the independent fair-score projection plus symmetric home-minus-away matchup advantages for:

- EPA per play;
- success rate;
- passing EPA per dropback;
- rushing EPA per attempt;
- explosive-play rate;
- early-down EPA.

For each metric `m`, the matchup advantage is:

```text
(home offense m - away defense m allowed)
- (away offense m - home defense m allowed)
```

It also retains offense/defense historical play counts as data-coverage diagnostics.

## Targets

Historical final scores are used only as evaluation targets:

- actual home margin;
- actual total;
- margin residual versus the independent fair-score baseline;
- total residual versus the independent fair-score baseline.

These targets are not used when constructing that game's pregame feature state.

## Baseline evaluation

`nfl.evaluation.summarize_baseline()` reports margin and total MAE, RMSE and bias. These are football-projection diagnostics, not evidence of betting profitability.

The next modeling step is to train residual adjustments only through expanding chronological splits and compare them against the untouched fair-score baseline. A feature or adjustment should remain disabled if it does not improve genuinely later data.
