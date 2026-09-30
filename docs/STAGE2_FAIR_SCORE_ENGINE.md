# Stage 2 — Fair-Score Engine

Stage 2 begins the actual NFL projection model. The first baseline estimates expected team points from completed football results only; sportsbook prices are deliberately absent.

## Baseline model

For each team-game row:

```text
expected_points = league_points + offense(team) - defense(opponent) + home_field
```

The offense and defense effects are ridge-regularized toward league average. The league scoring intercept and home-field coefficient are not penalized.

This is intentionally a transparent baseline, not the final model. It gives later EPA/success-rate/QB/personnel features an independent benchmark they must beat out of sample.

## Leakage contract

`fit_pregame_fair_score()` calls the Stage 1 `pregame_history()` primitive before fitting. A target in season `S`, week `W` cannot use results from week `W` or later. A dedicated unit test changes the target-week final score from a normal result to an extreme result and verifies that the pregame projection is unchanged.

## Outputs

`FairScoreModel` exposes:

- expected home points;
- expected away points;
- projected home margin;
- projected total;
- learned home-field value;
- offense rating by team;
- defense rating by team;
- net team rating;
- in-sample residual standard deviation for diagnostics only.

## Next Stage 2 work

The baseline must be expanded and evaluated chronologically with:

1. opponent-adjusted EPA/play;
2. dropback and rushing EPA splits;
3. success rate and explosiveness;
4. early-down/neutral-situation efficiency;
5. pace and play volume;
6. field-position/finishing metrics;
7. recency weighting and offseason regression;
8. quarterback state as a separately validated component;
9. walk-forward comparison against the simple ridge baseline.

A feature is not promoted merely because it improves in-sample fit. It must improve untouched chronological validation without introducing future information.
