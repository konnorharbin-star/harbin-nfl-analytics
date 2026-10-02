# Stage 30 — Chronological market-edge shrinkage research

The free 2022–2025 archive shows a clear downstream problem: raw model probabilities
generate large apparent edges but negative realized ROI across moneyline, spread and
totals. Stage 30 tests whether those edges are systematically overconfident.

This stage does **not** put sportsbook information into the football model.

## Model

For an already-produced football probability `p_model` and the two-way no-vig market
probability `p_market`, the research probability is:

```text
logit(p_shrunk) =
    logit(p_market)
    + alpha * (logit(p_model) - logit(p_market))
```

Interpretation:

- `alpha = 0` means use the no-vig market probability;
- `alpha = 1` means use the raw football probability;
- values between zero and one retain only the historically reliable portion of the
  model-vs-market edge.

## Chronology

Each market is evaluated independently.

- development: every archive row before 2024;
- validation: 2024;
- untouched holdout: 2025.

Alpha is selected **only** on development log loss, with Brier score as a tie-breaker.
Validation and holdout outcomes cannot choose alpha. Betting ROI never selects alpha.

## Outputs

`run_market_edge_shrinkage.py` writes:

- `reports/market_edge_shrinkage.json`;
- `docs/market_edge_shrinkage.json`.

For moneyline, spread and total the report contains:

- selected alpha;
- development loss grid;
- raw-model, market-only and shrunk Brier/log loss on 2024 and 2025;
- positive-EV/positive-edge betting diagnostics before and after shrinkage;
- whether shrinkage validates chronologically;
- whether the football model adds incremental probability value beyond the market.

## Release boundary

The archive entry prices are not independently verified, so this experiment is
research-only even if probability calibration improves.

The report hard-codes:

- `canonical_market_probability_change_enabled=false`;
- `betting_policy_change_enabled=false`;
- every market's `canonical_change_enabled=false`.

A successful result can justify a frozen forward shadow candidate. It cannot directly
change the current betting policy or open a release gate.
