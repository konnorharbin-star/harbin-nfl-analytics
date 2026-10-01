# Stage 13 — Historical Entry-Price Integrity

Stage 13 tightens the free nflverse historical market proof layer without changing the independent football projection or simulated bet economics.

## Why this exists

The free `initial_lines.csv` source can provide separate opening observations. For moneyline, the opening observation is itself a price. For spread and total, the opening observation supplies the line but not opening juice. The existing research backtest can still pair that opening line with archive-final juice, but that hybrid is not an observed opening entry price and therefore must not count as promotion-quality evidence.

## Provenance fields

Historical backtest rows now distinguish:

- `entry_line_observed`: a separate opening line/price observation exists.
- `entry_price_verified`: the American odds used by the simulated bet are themselves an observed opening price.
- `entry_price_stage=archive_open_price`: promotion-quality observed opening price.
- `entry_price_stage=archive_open_line_final_price`: useful research row with an opening line but archive-final juice.
- `entry_price_stage=archive_final_fallback`: no separate opening observation; research-only archive fallback.

With the current free source, only qualifying moneyline rows can have `entry_price_verified=true`. Spread and total opening-line rows remain valid for research and CLV-proxy diagnostics but cannot satisfy the historical entry-price release gate.

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
