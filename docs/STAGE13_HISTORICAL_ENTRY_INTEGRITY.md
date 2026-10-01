# Stage 13 — Historical Entry-Price Integrity

Stage 13 tightens the free nflverse historical market proof layer without changing the independent football projection or simulated bet economics.

## Why this exists

The free `initial_lines.csv` source can provide separate opening observations. For spread and total, an opening observation supplies the line but not opening juice. A research backtest may pair that opening line with archive-final juice, but that hybrid is not an observed opening entry price and therefore must not count as promotion-quality evidence.

The provenance contract also supports a future source with explicit opening moneyline prices. Such a row could qualify as a verified opening price only when the exact price used by the simulated bet is present in the opening observation. The current nflverse `initial_lines.csv` file does **not** provide moneyline rows.

## Current free-source coverage

The upstream nflverse `initial_lines.csv` currently contains 2021 regular-season spread/total opening lines only. It has no 2022+ rows and no moneyline rows. Therefore the canonical 2022–2025 free historical backtest has zero distinct opening observations and zero promotion-quality opening prices. That is an upstream coverage limitation, not a game-ID matching fallback.

This is intentionally fail-closed: 2022–2025 archive-final prices remain useful for research grading and diagnostics, but they cannot satisfy the historical-entry release gate. A future verified source can populate the same provenance fields without changing the release-gate contract.

## Provenance fields

Historical backtest rows now distinguish:

- `entry_line_observed`: a separate opening line/price observation exists.
- `entry_price_verified`: the American odds used by the simulated bet are themselves an observed opening price.
- `entry_price_stage=archive_open_price`: promotion-quality observed opening price.
- `entry_price_stage=archive_open_line_final_price`: useful research row with an opening line but archive-final juice.
- `entry_price_stage=archive_final_fallback`: no separate opening observation; research-only archive fallback.

Spread and total opening-line rows remain valid for research and CLV-proxy diagnostics but cannot satisfy the historical entry-price release gate when opening juice is unavailable.

## Promotion proof

`reports/evidence_report.json` keeps the broad archive summaries visible, but `ROBUST` status is calculated only from `entry_price_verified=true` rows. The promotion subset must independently satisfy the sample-size, ROI confidence-bound, CLV, multi-market, and multi-season requirements. A positive raw archive backtest cannot promote the model by itself.

Legacy CSVs that contain only `has_distinct_open` fail closed: they may report opening-line coverage for diagnostics, but they contribute zero verified entry-price bets until regenerated with the Stage 13 provenance fields.

## Automation

The `Free NFL Market Backtest` workflow now runs on relevant pull requests and weekly. PR runs publish the full backtest/evidence artifact. Scheduled or manually dispatched `main` runs may commit the canonical files used by the operational release gate:

- `reports/free_market_backtest.json`
- `reports/free_market_bets.csv`
- `reports/backtest_audit.json`
- `reports/evidence_report.json`

The model remains fail-closed: historical evidence never overrides the independent fair-score engine, and production eligibility still requires the separate forward/live portfolio evidence gates.
