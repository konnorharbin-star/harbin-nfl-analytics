# Stage 32 — NFL research-status contract

Stage 32 creates one machine-readable source of truth for NFL model research.

The repository now has many historically tested ideas and several prospective 2026
shadow candidates. Without a consolidated registry, a later stage could accidentally
treat an old near-miss, a superseded architecture result, or an archive-only market
experiment as if it were deployable evidence.

## Inputs

The registry reads only persisted evidence:

- `reports/candidate_benchmark.json`;
- `reports/market_edge_shrinkage.json`;
- `reports/forward_shadow_summary.json`.

It does not rebuild predictions and does not tune any model.

## Reconciliation

The registry makes three distinctions explicit:

1. Stage 14 selected the QB-state **architecture family** for margin development
   evidence, but Stage 15's later fixed-spec gate rejected every QB-margin
   specification. QB margin therefore remains disabled.
2. Stage 30 selected alpha `0.0` for moneyline, spread, and total and found zero
   incremental model-probability value over the no-vig market in the archive.
   That result is research-only and does not inject market data into fair scores.
3. The only model changes that can still mature are the already-frozen prospective
   2026 candidates in the forward-shadow ledgers.

## Outputs

- `reports/research_status.json`
- `docs/research_status.json`

The model card also surfaces a compact copy of the current research decision and active
forward candidates.

## Fail-closed contract

The research-status artifact always keeps:

- `canonical_model_change_enabled=false`;
- `canonical_market_change_enabled=false`.

Missing or invalid source reports cannot create readiness. Historical development
evidence, archive betting diagnostics, or a candidate-family win cannot bypass the
fixed-spec and forward-evidence gates.
