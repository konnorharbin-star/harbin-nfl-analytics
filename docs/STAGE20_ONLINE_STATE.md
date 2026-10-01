# Stage 20 — NCAA-style online NFL team state

Stage 20 ports a predictive *concept* from the NCAA model rather than another operational shell component: a sequential pregame team-state vector that is updated only after the relevant football results become available.

The NFL implementation remains independently estimated and separately validated. It does not copy CFB release weights, thresholds, calibrators, or betting conclusions.

## State vector

Before each NFL week, every team carries:

- Elo rating;
- opponent-adjusted scoring offense;
- opponent-adjusted scoring defense;
- recent margin form;
- recent total form;
- exponentially smoothed win rate;
- scoring-margin volatility;
- opponent Elo / strength-of-schedule state;
- games played.

A matchup exposes the NCAA-style derived signals:

- online expected home and away points;
- online margin and total;
- Elo difference;
- offense and defense differences;
- net-efficiency difference;
- margin-form difference;
- average total form;
- win-rate difference;
- average volatility;
- strength-of-schedule difference;
- games-played state;
- early-season indicator.

These signals are downstream corrections to the existing independent NFL fair-score baseline. Sportsbook data is not accepted.

## Strict weekly information boundary

The implementation is stricter than a game-by-game sequential update. All games in Week W are snapshotted from state created by weeks < W. Only after every Week W feature row is frozen are Week W scores allowed to update team state.

This prevents a Sunday afternoon or Monday result from leaking into another Week W historical feature merely because one game finished earlier on the calendar.

Season transitions regress team state toward league priors before the new season is observed.

## Fixed NFL development profiles

Three predeclared NFL response profiles are evaluated: slow, balanced, and responsive.

They differ only in football-state response speed, home-field assumption, Elo K, and offseason carry. These are NFL research profiles, not copied NCAA coefficients.

For each profile, Stage 20 tests standardized ridge penalties 1, 10, and 100 and positive blend weights 0.25, 0.50, 0.75, and 1.0.

Baseline / blend weight 0 is always admissible.

## Rolling gate

The source window is 2021–2025 so 2021 can warm the online state and support the canonical prior used by 2022 projections.

The modeling dataset covers 2022–2025, Weeks 5–18. Evaluation folds are 2023, 2024, and 2025.

For margin and total independently, the exact same state profile + ridge + blend must:

1. improve MAE in every development fold;
2. improve RMSE in every development fold;
3. improve aggregate MAE;
4. improve aggregate RMSE.

If no specification clears all requirements, weight 0 wins automatically and the canonical score remains unchanged.

## Evidence boundary

Stage 20 hard-codes canonical_score_adjustment_enabled=false and promotion_eligible=false.

The 2023–2025 folds are development evidence. A surviving fixed candidate may justify a separate frozen 2026 prospective shadow ledger, but historical selection cannot alter current betting decisions.

The real-source GitHub Action also verifies the schedule-source contract and reruns the strict chronology tests before producing its JSON audit.


## Real-source result

The 2021–2025 real-source audit completed successfully. Each predeclared state profile produced 831 chronological modeling rows, and the evaluation window contained 624 games: 208 each in 2023, 2024, and 2025.

Neither margin nor total produced a fixed state-profile + ridge + blend specification that improved both MAE and RMSE in every development fold. The fail-closed selector therefore returned the baseline / weight-0 state for both targets.

Aggregate baseline metrics over the 624 evaluation games were:

- margin MAE: 10.1966;
- margin RMSE: 13.0138;
- total MAE: 10.4926;
- total RMSE: 13.4427.

The selected outputs are therefore:

- margin state_config = disabled, blend_weight = 0.0, shadow_candidate = false;
- total state_config = disabled, blend_weight = 0.0, shadow_candidate = false;
- canonical_score_adjustment_enabled = false;
- promotion_eligible = false.

Stage 20 creates no 2026 online-state shadow candidate and makes no change to the canonical fair-score model. The negative result is retained because it narrows the useful NFL design space: the NCAA-style sequential state representation does not add robust linear residual value under the predeclared NFL gate.
