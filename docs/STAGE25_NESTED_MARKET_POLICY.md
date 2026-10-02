# Stage 25 — Nested NFL market-policy calibration

Stage 25 closes a Phase 4 gap between the NFL and NCAA operating models: market
thresholds now have an explicit chronological derivation path instead of remaining
only conservative hand-set defaults.

The fair-score and probability models are unchanged. Sportsbook data remains strictly
downstream.

## Evidence boundary

Only rows with `entry_price_verified=true` may calibrate the production policy.

Archive-final fallbacks, opening spread/total lines paired with final juice, and other
research-only price observations remain available for diagnostics, but they cannot
select thresholds or open a production gate.

With the currently available free 2022–2025 archive, this rule is expected to leave the
NFL policy in PAPER mode because there are no promotion-quality verified entries in
that window. That is intentional fail-closed behavior rather than a missing-data
workaround.

## Nested chronology

When at least three verified-entry seasons exist, Stage 25 splits them as:

1. development: every verified season before the penultimate season;
2. tuning: the penultimate verified season;
3. untouched evaluation: the latest verified season.

If the sample spans fewer seasons but has enough whole-week blocks, it uses a
chronological 50/25/25 whole-week split. Row-order fallback is used only when no
whole-week split is possible.

Threshold candidates are ranked on development data. A candidate must then survive the
tuning block before it is frozen. The untouched evaluation block can reject the frozen
policy, but it is never allowed to choose a threshold.

## Market-specific gates

Moneyline, spread, and total are calibrated independently.

A candidate requires at least 40 development bets. A frozen candidate requires at
least 20 tuning bets with non-negative ROI and non-negative CLV. The untouched
evaluation then requires at least 20 bets with non-negative ROI and non-negative CLV.

Repeated weak weeks may be excluded only when the same week is negative in both the
development and tuning blocks with minimum sample support. Evaluation results can
never create a week exclusion.

## Release behavior

The generated `reports/production_policy.json` remains PAPER unless:

- the historical evidence report is already ROBUST on verified entry-price data; and
- at least two markets pass the frozen untouched evaluation.

Even then, the existing hard release gate still controls real staking. Stage 25 does
not bypass multi-book, monitoring, forward-ledger, CLV, or sample-size requirements.

## Files

- `nfl/policy_calibration.py` — verified-entry filtering, nested split, candidate
  selection, frozen evaluation, and policy persistence.
- `run_policy_calibration.py` — standalone policy derivation runner.
- `tests/test_policy_calibration.py` — fail-closed and untouched-evaluation tests.

The free historical market workflow also regenerates the policy snapshot after each
canonical evidence refresh.
