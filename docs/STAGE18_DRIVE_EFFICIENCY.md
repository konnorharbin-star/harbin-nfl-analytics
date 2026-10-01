# Stage 18 — Drive-level scoring opportunity research

Stage 18 tests whether possession-level football signals improve the independent fair-score baseline after the earlier per-play and situational rate layers failed their fixed chronological gates.

## Candidate signals

The drive layer reconstructs offensive possessions directly from nflverse play-by-play and smooths team state toward contemporaneous league priors. It measures:

- points per drive;
- scoring-drive rate;
- red-zone entry rate;
- points per red-zone entry;
- drive security (one minus turnover-drive rate);
- starting field position;
- first downs per drive;
- drives per game.

These features are intentionally lower-frequency than EPA-per-play. They target scoring opportunity creation, finishing, possession sustain, turnovers, and field-position burden.

## Leakage controls

Every target-week feature state is built from completed same-season regular-season game IDs strictly before the target week. PBP is filtered to those exact game IDs before drive reconstruction. Target-week and future drives cannot enter a historical feature state.

Sportsbook spreads, totals, prices, implied probabilities, CLV, and betting results are not accepted by the drive feature, dataset, or selection modules.

## Fixed rolling gate

Development seasons are 2022–2025. Evaluation folds are 2023, 2024, and 2025, using Weeks 5–18.

A candidate uses one fixed feature bundle and one fixed ridge penalty across every fold. Margin and total are selected independently. A nonzero candidate survives only if:

1. MAE improves in every development fold;
2. RMSE improves in every development fold;
3. aggregate MAE improves;
4. aggregate RMSE improves.

Baseline / zero adjustment is always admissible and wins automatically when no fixed candidate clears all requirements.

## Real-source result

The 2022–2025 nflverse audit completed successfully and verified every drive-reconstruction field, including `drive`, `play_id`, `yardline_100`, `first_down`, turnover flags, and pre/post possession scores. The walk-forward dataset contained 831 rows; the 2023–2025 evaluation sample contained 624 games (208 per season).

No fixed drive feature-bundle/ridge combination improved both MAE and RMSE in all three development folds. The selector therefore chose the disabled baseline for both targets:

- margin baseline/selected MAE: `10.1966 / 10.1966`;
- margin baseline/selected RMSE: `13.0138 / 13.0138`;
- total baseline/selected MAE: `10.4926 / 10.4926`;
- total baseline/selected RMSE: `13.4427 / 13.4427`;
- margin positive folds: `0 / 3`;
- total positive folds: `0 / 3`.

Accordingly, Stage 18 creates no new 2026 shadow candidate and makes no change to the canonical fair score.

## Evidence boundary

The Stage 18 audit hard-codes `canonical_score_adjustment_enabled=false` and `promotion_eligible=false`. The 2022–2025 seasons are development evidence only. A future drive-level design would need a new predeclared specification and the same chronological gate before it could even enter shadow evaluation.
