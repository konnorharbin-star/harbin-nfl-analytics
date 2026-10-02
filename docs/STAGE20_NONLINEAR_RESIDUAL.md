# Stage 20 — NCAA-style linear + nonlinear residual architecture

Stage 20 tests the largest predictive architecture difference that remained between the
CFB and NFL models: the CFB model learns residual corrections with both a standardized
linear model and a shallow histogram gradient-boosting model, while the NFL research
program had previously tested only linear ridge residuals.

## NFL feature contract

The Stage 20 model uses only pregame football information already produced by the
chronological NFL dataset:

- overall EPA matchup advantage;
- success-rate matchup advantage;
- pass EPA/dropback matchup advantage;
- rush EPA/attempt matchup advantage;
- explosiveness matchup advantage;
- early-down EPA matchup advantage;
- canonical baseline margin and total;
- historical offensive and defensive play-count state.

Sportsbook lines, odds, implied probabilities, CLV, betting results, current injuries,
and postgame information are excluded.

## NCAA-style architecture, NFL-only selection

Each target (margin and total) gets two residual learners:

1. standardized ridge regression;
2. shallow histogram gradient boosting.

For every evaluation season, all earlier rows are divided chronologically by whole
season/week blocks into a core fit block and a later tune block. The tune block chooses:

- ridge regularization from 1, 10, and 100;
- the ridge-versus-boost share from 0, .25, .50, .75, and 1;
- the overall residual weight from 0, .25, .50, .75, and 1.

Weight zero is the default. A nonzero specification is admissible only if it improves
both MAE and RMSE on the tune block. The chosen specification is then refit on all
pregame rows before the evaluation season and scored once on that season.

## Rolling development gate

Evaluation seasons are 2023, 2024, and 2025, Weeks 5–18. A target can become a 2026
shadow candidate only if its selected architecture improves both MAE and RMSE in every
evaluation season and in aggregate.

The audit hard-codes:

- canonical score adjustment disabled;
- promotion eligibility disabled;
- 2026 as the reserved prospective season.

Historical success, if any, can only justify a separate persisted 2026 shadow test.
