# NFL / CFB Architecture Parity Plan

Harbin NFL Analytics intentionally mirrors the operational architecture of Harbin Sports Analytics CFB while keeping NFL football data, model weights, calibration, thresholds, and validation evidence independent.

## Shared operational layers

The NFL repository should expose the same major layers as CFB:

1. leakage-safe football data and fair-score projection;
2. calibrated probability distributions;
3. verified market ingestion and line history;
4. policy-driven signal classification;
5. execution-market validation;
6. capped portfolio allocation and drawdown throttles;
7. append-only forward decision ledger;
8. independent grading ledger;
9. evidence/proof reporting;
10. live health and drift monitoring;
11. hard RESEARCH -> PAPER -> SHADOW -> PRODUCTION release gate;
12. canonical machine-readable model report.

## Intentional NFL differences

League-specific football logic remains NFL-native. Quarterback value, roster/depth-chart handling, rest/travel rules, schedule structure, nflverse identifiers, and NFL calibration are not copied from CFB. No CFB coefficient, threshold, or profitability result is treated as NFL evidence.

## Promotion rule

Structural parity does not mean evidence parity. An NFL component is enabled only after NFL chronological validation. Operational modules may exist before the release gate allows real staking; until the evidence gates are met, the model remains PAPER or SHADOW.
