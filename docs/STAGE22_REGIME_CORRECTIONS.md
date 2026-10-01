# Stage 22 — Targeted residual-regime correction replay

Stage 22 tests whether the two persistent Stage 21 residual patterns are actually correctable out of season. It is a fixed development replay, not an untouched holdout: Stage 21 already used 2023–2025 to identify the regimes.

## Fixed hypotheses

No coefficient search is allowed. For each test season, the correction is the simple mean canonical residual from matching games in strictly earlier development seasons:

- **Margin / away rest edge:** when `ctx_rest_diff_days <= -2`, add the earlier-season mean margin residual to the canonical projected home margin.
- **Total / low projected total:** when canonical `baseline_total < 42`, add the earlier-season mean total residual to the canonical projected total.

Games outside the target regime receive exactly zero correction. Sportsbook inputs are excluded. The canonical score baseline already includes the validated `0.10` prior-season weight.

## Chronology

The dataset is reconstructed for 2022–2025 regular-season Weeks 5–18. Test folds are 2023, 2024, and 2025:

- 2023 correction estimates use 2022 only;
- 2024 estimates use 2022–2023 only;
- 2025 estimates use 2022–2024 only.

Target-season outcomes can never alter that season's correction estimate.

## Fixed gate

A hypothesis may become a future shadow candidate only if, in every 2023–2025 fold:

1. MAE improves within the affected regime;
2. RMSE improves within the affected regime;
3. overall target MAE improves after leaving all non-regime games unchanged;
4. overall target RMSE improves.

Aggregate regime and overall MAE/RMSE must also improve. There is no ridge, blend, shrinkage, or threshold grid, and baseline/zero adjustment remains canonical regardless of the historical result.

## Evidence boundary

`canonical_score_adjustment_enabled=false` and `promotion_eligible=false` are hard-coded. Because the hypotheses were discovered using 2023–2025, even a clean historical replay is only development evidence. A survivor could justify a newly frozen future-only 2026 shadow starting after the specification is merged; it cannot be backfilled into earlier 2026 weeks or promoted directly.

Existing frozen recent-form and QB-total forward shadows remain untouched.
