# Stage 44 — Historical proof input isolation

Stage 44 fixes a test and API isolation defect discovered after the first successful
free ESPN archived-opening evidence build.

Once `reports/verified_market_bets.csv` contained the real 1,632-bet ESPN sample,
unit tests that called `build_evidence_report()` with temporary fixture archive files
silently loaded that repository-global provider file too. Their fixture result therefore
depended on unrelated canonical state.

## Contract

`build_evidence_report()` is now input-isolated:

- archive/backtest inputs come only from the paths supplied by the caller;
- verified provider evidence is omitted unless `verified_bets_path` is explicitly
  supplied.

The production writer remains unchanged in behavior:

`write_evidence_report()` still defaults to
`reports/verified_market_bets.csv` and explicitly passes that path into the builder.

## Why this matters

Pure evidence calculations and unit tests must be reproducible from their declared
inputs. Canonical repository state must not leak into a fixture simply because a default
file happens to exist in the checkout.

## Evidence boundary

This stage does not alter the 1,632 verified ESPN archived-opening bets, their results,
the nested production policy, or any release threshold. It only ensures those rows are
included when explicitly requested by the operational proof path.
