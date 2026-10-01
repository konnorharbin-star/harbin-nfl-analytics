# Stage 16 — Frozen QB-total forward shadow

Stage 15 produced a stricter fixed-spec result than the earlier nested QB audit:

- **QB margin:** disabled. No single feature/ridge specification improved both MAE and RMSE in every 2023, 2024, and 2025 development fold.
- **QB total:** development-only shadow candidate using QB EPA + CPOE (`quality`), ridge alpha `0.1`, and the existing 75-dropback shrinkage prior.

The total candidate improved both MAE and RMSE in all three development folds, but those seasons were already used for model selection. They are not promotion evidence. Stage 16 therefore freezes the exact specification and records only genuinely prospective 2026 predictions made before kickoff.

## Evidence contract

The forward ledger is `history/qb_total_shadow_predictions_v1.csv`.

A row is eligible only when all of the following are true:

- season is exactly 2026;
- capture time is before kickoff;
- specification version is `stage15_fixed_quality_v1`;
- feature set is `quality` (QB EPA + CPOE);
- ridge alpha is `0.1`;
- QB shrinkage prior is 75 dropbacks;
- training seasons are exactly 2022–2025;
- release state is `SHADOW`.

The first valid snapshot for each game/specification is retained. Reconstructed historical predictions are excluded.

## Grading gate

`nfl/qb_total_forward.py` grades the persisted first-snapshot predictions against final scores and compares them with the same canonical baseline total. The paired bootstrap gate requires at least 128 forward games before promotion evidence is possible. Both MAE and RMSE improvement must have positive lower 95% confidence bounds.

Until that forward gate is satisfied, the candidate remains shadow-only. Even if the forward report later says `PROMOTION_EVIDENCE`, a separate release change is still required before the canonical projection can consume the adjustment.

## Operational separation

The canonical market engine continues to use `baseline_total`. The QB-total shadow is not used for probabilities, sportsbook comparison, EV, Kelly sizing, portfolio allocation, or execution.

The older QB margin research output is forced to a zero correction after the Stage 15 fixed-spec failure. No QB margin shadow is created.

Two dedicated workflows keep ownership explicit:

- `NFL QB Total Forward Capture` records pre-kickoff predictions and is the sole writer of the forward ledger.
- `NFL QB Total Forward Grade` grades completed games and writes `reports/qb_total_forward.json`, `reports/qb_total_forward_graded.csv`, and `docs/qb_total_forward.json`.
