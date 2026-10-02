# Stage 20 — Decoupled margin and total architecture

Stage 20 tests whether the NFL fair-score engine improves when margin and total are estimated by separate game-level models instead of being forced to emerge from the same pair of team-point regressions.

## Candidate architecture

The research model has two independent ridge-shrunk components:

- **Margin model:** home-field intercept plus a team-strength rating, with the home team entering +1 and the away team -1.
- **Total model:** league-total intercept plus additive team scoring-environment ratings for both participants.

The model uses only completed regular-season schedule results. No sportsbook spread, total, odds, implied probability, CLV, or betting result is accepted as an input.

## Information boundary

For target season/week:

- all completed regular-season games from the immediately prior season may enter at the already validated weight of 0.10;
- current-season games may enter only when their week is strictly earlier than the target week;
- target-week outcomes and future games are excluded.

The canonical direct-score baseline is rebuilt at the same weekly boundary and with the same prior-season weight.

## Fixed rolling gate

Development folds are 2023, 2024, and 2025, Weeks 5–18.

Predeclared ridge values are 2.0, 8.0, and 32.0. Margin and total are allowed to select different ridge strengths because their estimation is intentionally decoupled.

A nonzero side survives only when one fixed ridge improves both MAE and RMSE:

1. in every development fold; and
2. in the aggregate 2023–2025 sample.

If no ridge clears that gate, that side falls back to the canonical direct-score baseline.

## Evidence boundary

Stage 20 is development research only. The evaluator hard-codes:

- `canonical_score_change_enabled=false`
- `promotion_eligible=false`

A surviving historical margin or total candidate would only justify a separately frozen 2026 prospective shadow experiment. Historical selection alone cannot alter canonical projections, probabilities, market edges, or stake sizing.
