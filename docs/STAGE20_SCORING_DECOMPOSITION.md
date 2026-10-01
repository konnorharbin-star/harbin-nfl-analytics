# Stage 20 — Possession vs non-possession scoring decomposition

Stage 20 tests whether the canonical direct-score model is being hurt by volatile defensive and special-teams scoring that is mixed into final team points.

## Candidate architecture

For every historical team-game, nflverse PBP is used to reconstruct **possession-team scoring** by summing positive changes in `posteam_score_post - posteam_score`. The remainder is defined as:

`non-possession points = final team points - possession-team points`

The repeatable possession-scoring component is modeled with the same opponent-adjusted ridge structure as the canonical fair-score engine. The non-possession remainder is deliberately shrunk much more aggressively.

Three remainder treatments are predeclared:

- `constant`: weighted league-average remainder only;
- `ridge_32`: opponent-adjusted remainder model with ridge 32;
- `ridge_128`: opponent-adjusted remainder model with ridge 128.

The possession-scoring ridge is fixed at 8.0 for every candidate.

## Information boundary

The candidate uses the same pregame history boundary as the canonical baseline:

- immediately prior regular season at the validated `0.10` sample weight;
- current-season completed regular-season games strictly before the target week at weight `1.0`;
- PBP filtered by the exact eligible historical game IDs before any scoreboard deltas are reconstructed;
- sportsbook spreads, totals, prices, probabilities, CLV, and betting outcomes are excluded.

The implementation fails closed if possession scoring cannot be reconstructed for every historical team-game or if reconstructed possession points exceed the final team score.

## Fixed development gate

Evaluation folds are 2023, 2024, and 2025, Weeks 5–18. These seasons are development evidence only.

One predeclared remainder treatment must improve all four primary score metrics in every fold and in aggregate:

1. margin MAE;
2. margin RMSE;
3. total MAE;
4. total RMSE.

If no specification clears the gate, the direct-score baseline wins automatically. The audit hard-codes `canonical_score_change_enabled=false` and `promotion_eligible=false`; a historical survivor could only justify a separately frozen 2026 shadow experiment.

## Rationale

Stages 17–19 found no robust gain from situational residuals, drive-level residuals, or a possessions × points-per-drive replacement baseline. Stage 20 therefore tests a narrower structural hypothesis: whether separating relatively repeatable possession scoring from sparse defensive/special-teams points improves signal-to-noise without adding sportsbook information.
