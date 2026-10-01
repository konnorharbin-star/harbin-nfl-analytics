# Stage 17 — Situational possession-efficiency research

Stage 17 tests whether a compact set of NFL-native situational signals can improve the independent fair-score baseline without using sportsbook information.

## Candidate signals

The feature layer isolates situations not explicitly represented by the earlier generic EPA bundle:

- red-zone EPA;
- third-down success rate;
- third/fourth-and-short success rate (2 yards or fewer);
- offensive sack rate / defensive sack generation;
- early-down pass rate;
- offensive/defensive plays per game.

Sparse rates are shrunk toward the contemporaneous league mean before matchup signals are built. The shrinkage priors are fixed in code and are not tuned against sportsbook outcomes.

## Leakage controls

Every target-week snapshot is reconstructed from completed same-season regular-season game IDs strictly before the target week. PBP is filtered by those exact game IDs before any feature is calculated. Target-week and future PBP cannot enter the feature state.

The fair-score baseline is still fit independently from football scores. Sportsbook lines, prices, implied probabilities, CLV, and betting results are not accepted by the Stage 17 feature or selection modules.

## Fixed rolling gate

Development seasons are 2022–2025. Evaluation folds are 2023, 2024, and 2025 (Weeks 5–18).

A candidate uses one fixed feature bundle and one fixed ridge penalty across every fold. Margin and total are evaluated independently. A nonzero candidate is retained only if:

1. MAE improves in every development fold;
2. RMSE improves in every development fold;
3. aggregate MAE improves;
4. aggregate RMSE improves.

Baseline / zero adjustment is always an admissible result and wins automatically if no fixed candidate clears all requirements.

## Evidence boundary

These seasons are development evidence, not a pristine promotion holdout. The audit hard-codes `canonical_score_adjustment_enabled=false` and `promotion_eligible=false`. Any surviving candidate can only justify a separately frozen 2026 prospective shadow experiment; it cannot alter the canonical fair score directly.

The real-source audit also fails closed if nflverse PBP does not supply the required situational columns, including `sack`, `ydstogo`, and `yardline_100`.
