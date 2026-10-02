# Stage 22 — NCAA-style nonlinear residual ensemble

Stage 22 tests the largest remaining predictive-architecture gap between the NCAA and NFL repositories. The NCAA model corrects its independent football baseline with a blend of a standardized ridge residual model and a shallow gradient-boosting residual model, then tunes the residual weight chronologically with weight 0 always admissible.

The NFL implementation copies that architecture, not the NCAA coefficients.

## Pregame feature matrix

Each game row joins four independently walk-forward feature families on `season/week/game_id` with one-to-one coverage required:

- core EPA/success/explosiveness/early-down matchup advantages;
- situational PBP matchup signals;
- drive-efficiency/scoring-opportunity matchup signals;
- last-observed quarterback-state matchup signals.

The independent fair-score baseline values may be used as pregame predictors. The feature contract rejects identifiers, actual outcomes, residual targets, market/odds/closing/CLV fields, stake fields, and postgame/result fields.

## Residual learner

For margin and total separately, Stage 22 fits:

1. a median-imputed, standardized ridge residual model; and
2. a median-imputed shallow histogram gradient-boosting residual model.

The candidate correction is a weighted blend of the two learners. NFL-specific development grids are:

- ridge alpha: 10, 30;
- ridge fraction in the learner blend: 0.50, 0.70;
- boost max depth: 2, 3;
- final residual weight: 0.00, 0.25, 0.50, 0.75, 1.00.

The final residual-weight grid always includes exactly zero, so the independent fair-score baseline wins automatically when the tune block cannot improve both MAE and RMSE.

## Nested chronology

This stage uses two rolling development tests:

- train on 2022, tune model specification/weight on 2023, test once on 2024;
- train on 2022–2023, tune on 2024, test once on 2025.

After selection, the chosen specification is refit on all data through the tune season before the later test season is scored. Test-season outcomes never participate in model/specification/weight selection.

The architecture is considered historically promising for a side only if both later test folds improve MAE and RMSE and the combined 2024–2025 test sample also improves both metrics. Even then, Stage 22 cannot change production directly; a fixed specification would have to be frozen separately for 2026 prospective shadow evidence.

## Evidence boundary

Sportsbook data is absent from the combined feature matrix and learner. The evaluator hard-codes:

- `canonical_score_adjustment_enabled=false`
- `promotion_eligible=false`

2022–2025 are development evidence only. The canonical fair score, probability model, market edges, and stake sizing remain unchanged regardless of this historical audit.

## Real-source result

The real nflverse audit built **831** walk-forward rows across 2022–2025 and admitted **66** numeric pregame football features after the leakage contract. Full Ruff/pytest, Stage 1, and canonical audit checks passed.

Neither side survived the nested later-season tests:

- **Margin:** 2023 tuning selected residual weight `0`, so the 2024 test remained the canonical baseline. On the next fold, 2024 tuning selected `ridge_alpha=30`, `ridge_fraction=0.50`, `boost_depth=3`, residual weight `0.25`; that tune improvement failed in 2025, where MAE/RMSE worsened from `10.6069 / 13.1996` to `10.6889 / 13.2820`. Aggregate 2024–2025 MAE/RMSE worsened from `10.2956 / 13.0832` to `10.3366 / 13.1248`. Positive test folds: `0/2`.
- **Total:** 2023 tuning selected `ridge_alpha=10`, `ridge_fraction=0.70`, `boost_depth=2`, residual weight `0.25`; the 2024 test worsened from `10.1277 / 13.2747` to `10.1414 / 13.3847`. The 2024 tune block then selected residual weight `0`, leaving the 2025 test at baseline. Aggregate 2024–2025 MAE/RMSE worsened from `10.3058 / 13.2891` to `10.3126 / 13.3441`. Positive test folds: `0/2`.

Stage 22 therefore creates no new 2026 shadow candidate and makes no canonical-score change. The result is useful evidence that NCAA's nonlinear residual architecture does not automatically transfer to the NFL feature/data regime.
