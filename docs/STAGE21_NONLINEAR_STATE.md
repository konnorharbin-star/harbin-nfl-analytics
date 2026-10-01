# Stage 21 — Nonlinear online-state residual research

Stage 21 tests the remaining major predictive-model concept used by the NCAA implementation that the NFL research stack had not yet evaluated: a shallow nonlinear residual learner layered on top of an independent football baseline.

Stage 20 showed that the NCAA-style sequential online-state vector did not add robust value through a linear ridge residual. Stage 21 keeps that exact leak-safe feature family fixed and asks a narrower question: do interactions among those state variables contain repeatable NFL signal?

## Feature boundary

The input family is exactly the Stage 20 pre-week online-state vector:

- expected online home/away points, margin, and total;
- Elo difference;
- offense and defense differences;
- net-efficiency difference;
- margin and total form;
- win-rate difference;
- volatility;
- strength of schedule;
- games played and early-season state.

Sportsbook data, CLV, graded results, and 2026 outcomes are excluded.

All Week W state features are frozen before any Week W score updates state.

## Nonlinear candidates

Three predeclared research structures are tested:

- shallow depth-2 gradient boosting;
- an equal ridge / depth-2 boosting hybrid;
- an equal ridge / depth-3 boosting hybrid.

The NFL parameters are predeclared for this research stage rather than inherited as production coefficients from CFB. The candidate residual is then scaled by one of four positive fixed blend weights: 0.25, 0.50, 0.75, or 1.0.

Weight 0 is always the canonical fallback.

## Rolling gate

The source window is 2021–2025 and the modeling rows cover 2022–2025, Weeks 5–18. Evaluation folds are 2023, 2024, and 2025.

For margin and total independently, one exact state profile + nonlinear structure + blend must improve both MAE and RMSE in every development fold and in aggregate. Otherwise the selected candidate is null and weight 0 remains canonical.

The audit also records the best rejected candidate, ranked first by number of fully positive folds and then by aggregate error. This is diagnostic only and cannot override the gate.

## Evidence boundary

Stage 21 hard-codes canonical_score_adjustment_enabled=false and promotion_eligible=false.

The 2023–2025 results are development evidence only. A survivor may justify a separately frozen 2026 prospective shadow ledger. Historical selection alone cannot change current score projections, probabilities, market edges, or stakes.
