# Stage 15 — Fixed QB Shadow Specification

Stage 14 identified quarterback-state residuals as the only NFL-native margin architecture that cleared both nested development holdouts. However, the selected QB feature set and ridge differed by fold. Stage 15 therefore does not reuse either fold-specific specification.

Instead, Stage 15 evaluates one constant feature set and ridge across rolling 2023, 2024, and 2025 development seasons. Each season is scored strictly from QB state available before that game, with the model trained only on earlier seasons.

## Selection gate

For a nonzero fixed QB specification to survive:

- the same feature set must be used in every fold;
- the same ridge alpha must be used in every fold;
- both MAE and RMSE must improve versus the canonical fair-score baseline in every fold;
- aggregate MAE and RMSE must also improve;
- baseline / zero adjustment remains the fallback if no specification satisfies all conditions.

Margin and total are selected independently. The search uses the existing NFL QB feature families and ridge grid; sportsbook prices never enter this process.

## Evidence status

The 2022–2025 data used here are development evidence. They are not an untouched promotion holdout. Any surviving fixed specification is only eligible to be frozen for a separately persisted 2026 SHADOW ledger.

The audit always reports:

- `canonical_score_adjustment_enabled=false`;
- `promotion_eligible=false`.

A historical fixed-spec result therefore cannot alter the canonical fair score or create production eligibility by itself.
