# Stage 14 — Rolling NFL Candidate Benchmark

Stage 14 creates one common development scoreboard for NFL-native fair-score candidates instead of evaluating isolated model changes with unrelated criteria.

## Chronological folds

The benchmark uses two rolling validation → holdout folds:

- 2023 validation → 2024 holdout
- 2024 validation → 2025 holdout

All candidates use regular-season Weeks 5–18. These seasons are explicitly **development evidence** because prior research in this repository has already inspected them. They are not presented as pristine promotion holdouts.

The 2026 season is excluded from selection and remains reserved for separately persisted prospective/shadow evidence.

## Candidates

The common scoreboard evaluates:

- score-level recency weighting;
- prior-season scoring shrinkage;
- opponent-adjusted play-by-play residuals;
- last-observed quarterback-state residuals.

The already-frozen recent-form PBP total candidate remains a separate 2026 shadow experiment and is not re-tuned by this benchmark.

## Fail-closed selection

Baseline / zero adjustment is always selectable. A nonzero candidate must clear both chronological development folds. Score recency and prior-season changes are whole-model changes and therefore must clear their complete nested gates. Opponent-adjusted PBP and QB residuals are independently fit for margin and total and may qualify by target.

Among candidates that clear every fold, the benchmark reports the weighted relative reduction in MAE/RMSE and identifies a **SHADOW_CANDIDATE**. This does not alter the canonical score. If no candidate clears every fold, the benchmark selects `baseline` and the layer stays disabled.

## Promotion boundary

`reports/candidate_benchmark.json` is research evidence only:

- `canonical_score_adjustment_enabled` is always `false`;
- `promotion_eligible` is always `false`;
- any historical winner must still be frozen before independent 2026 prospective testing;
- market prices never enter candidate selection.

The purpose is to improve the NFL projection on football evidence first, before revisiting sportsbook thresholds or staking policy.
