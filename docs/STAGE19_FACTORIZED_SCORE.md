# Stage 19 — Factorized possessions × scoring-efficiency baseline

Stage 19 changes the fair-score architecture itself rather than adding another residual feature layer.

The canonical model directly regresses team points on offense, opponent defense, and home field. Stage 19 tests the football decomposition:

`expected team points = expected shared game possessions × expected team points per drive`

## Architecture

Shared possessions are estimated from offensive drive counts in nflverse play-by-play. One game-level ridge model learns a league possession rate plus additive team pace effects for both participants.

Scoring efficiency is estimated as team points divided by offensive drives. A separate opponent-adjusted ridge model learns league points per drive, team scoring efficiency, opponent defensive efficiency, and home-field effect.

The final home and away projections share the same expected possession count but receive separate opponent-adjusted scoring-efficiency estimates.

## Information boundary

The factorized model uses the same scoring-history boundary as the canonical baseline:

- completed regular-season games from the immediately prior season enter with the validated `0.10` sample weight;
- current-season games must be completed strictly before the target week and receive full weight;
- PBP is filtered by the exact eligible historical game IDs before drive counts are computed;
- sportsbook spreads, totals, prices, probabilities, CLV, and betting outcomes are never inputs.

Target-week or future PBP therefore cannot affect a historical projection.

## Fixed development gate

The evaluation folds are 2023, 2024, and 2025, Weeks 5–18. These seasons are development evidence, not pristine promotion holdouts.

Only three predeclared ridge strengths are tested: `2.0`, `8.0`, and `32.0`. The same ridge is used for the possession and points-per-drive components and remains fixed across all folds.

A factorized specification can survive only if the same ridge improves all four primary score metrics in every fold and in aggregate:

1. margin MAE;
2. margin RMSE;
3. total MAE;
4. total RMSE.

If no specification clears that gate, the canonical direct-score baseline wins automatically. The audit always keeps `canonical_score_change_enabled=false` and `promotion_eligible=false`; a historical survivor could only justify a separately frozen 2026 shadow experiment.

## Why this is distinct

Earlier NFL experiments tested prior-season weighting, current-season recency, opponent-adjusted PBP residuals, recent-form residuals, situational rates, quarterback state, and drive-level residual features. Stage 19 is different because possession volume and scoring efficiency are the score-generating baseline itself rather than corrections applied after a direct points projection.
