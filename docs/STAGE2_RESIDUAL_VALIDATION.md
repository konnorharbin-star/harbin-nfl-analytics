# Stage 2 — Residual Validation

The advanced football layer is not allowed to alter the fair-score engine merely because a feature sounds predictive. It must improve later data under chronological validation.

## Model

The residual layer starts from the independent baseline projection and predicts only the remaining error:

```text
adjusted margin = baseline margin + predicted margin residual
adjusted total  = baseline total  + predicted total residual
```

The first candidate is a standardized ridge regression using only the six football matchup advantages already generated from pregame PBP:

- EPA/play;
- success rate;
- pass EPA/dropback;
- rush EPA/attempt;
- explosive rate;
- early-down EPA.

The intercept is not penalized. Feature scaling is estimated from training data only.

## Nested chronology

The default audit reconstructs 2022–2025 regular seasons beginning in Week 5.

1. 2022–2023 are the initial training period.
2. 2024 is used to choose the ridge penalty from a small fixed grid.
3. The model is refit on all data before 2025.
4. 2025 is scored once as the untouched holdout.

The holdout season cannot influence penalty selection. Unit tests explicitly mutate holdout outcomes and verify that the selected penalty is unchanged.

## Fail-closed candidate gate

For margin and total independently, `candidate_pass` is true only when the advanced adjustment improves both MAE and RMSE on the untouched holdout relative to the simple fair-score baseline.

A passing candidate is **not** a production betting signal. It only earns the right to continue to broader multi-season validation, calibration and market comparison. A failing candidate remains disabled rather than being forced into the live score projection.

Sportsbook prices remain outside this entire Stage 2 validation path.
