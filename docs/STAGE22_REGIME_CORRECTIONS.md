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

## Live-source result

The real nflverse audit completed successfully on 831 walk-forward rows across 2022–2025 and 624 evaluation games across 2023–2025. Both fixed hypotheses failed the gate.

### Margin — away rest edge

The estimated correction was not stable across expanding histories:

- 2023: `+1.9640` points from 20 prior matching games; 29 test games; regime MAE `8.5406 -> 9.1078`, regime RMSE `10.5266 -> 11.3234`; fold failed.
- 2024: `-1.2404` points from 49 prior matching games; 39 test games; regime MAE `11.5911 -> 11.5530`, regime RMSE `15.1267 -> 14.9982`; fold passed.
- 2025: `-1.6568` points from 88 prior matching games; 32 test games; regime MAE `8.9469 -> 8.7313`, regime RMSE `11.3915 -> 10.9601`; fold passed.

Across all 624 evaluation games, margin MAE worsened from `10.1966` to `10.2095` and RMSE worsened from `13.0138` to `13.0166`. Within the 100 affected games, MAE worsened from `9.8603` to `9.9410` and RMSE worsened from `12.7632` to `12.7810`. The candidate passed `2/3` folds and is **not** a shadow candidate.

### Total — projected total under 42

The expanding-history correction remained positive but did not transfer consistently:

- 2023: `+2.9531` points from 51 prior matching games; 72 test games; regime MAE `12.1250 -> 12.3483`, regime RMSE `15.4428 -> 15.4420`; fold failed because MAE worsened.
- 2024: `+2.0913` points from 123 prior matching games; 45 test games; regime MAE `10.5556 -> 10.0890`, regime RMSE `13.4122 -> 12.8333`; fold passed.
- 2025: `+2.7843` points from 168 prior matching games; 31 test games; regime MAE `10.0035 -> 10.3586`, regime RMSE `12.5472 -> 12.6473`; fold failed.

Across all 624 evaluation games, total MAE worsened from `10.4926` to `10.5024`, while RMSE improved from `13.4427` to `13.4065`. Within the 148 affected games, MAE worsened from `11.2034` to `11.2446`, while RMSE improved from `14.2719` to `14.1274`. The candidate passed `1/3` folds and is **not** a shadow candidate.

These results show that the Stage 21 signed biases are not sufficient to support a fixed empirical correction. No new 2026 shadow is created and the canonical score remains unchanged.

## Evidence boundary

`canonical_score_adjustment_enabled=false` and `promotion_eligible=false` are hard-coded. Because the hypotheses were discovered using 2023–2025, even a clean historical replay would only have been development evidence. Neither hypothesis survived, so no future-only shadow is justified.

Existing frozen recent-form and QB-total forward shadows remain untouched.
