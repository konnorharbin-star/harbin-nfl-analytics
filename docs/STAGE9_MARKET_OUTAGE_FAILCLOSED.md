# Stage 9 — Current-market outage fail-closed behavior

## Incident

The first post-merge Stage 9 operational run successfully built the independent football projection and frozen recent-form shadow, then aborted when the current market collector returned no usable ESPN or optional professional quotes. Because the exception occurred before generated state was committed, valid pre-kickoff shadow evidence would have been lost.

## Contract

A current-market source outage must block betting, not erase independent football output.

The operational pipeline now catches current-market collection failures at the pipeline boundary and emits an explicit blocked market-source state:

- zero market quotes;
- zero sportsbook count;
- zero multi-book coverage;
- source error retained in metadata/data quality;
- no fabricated line or price;
- market intelligence receives an empty quote set;
- portfolio allocation therefore has no executable candidate;
- football projections and the prospective recent-form shadow ledger may still be published and persisted.

The underlying market adapters remain strict and may raise when no usable quotes exist. Only the operational shell converts that source failure into a fail-closed run state.

## Validation

`tests/test_market_outage_failclosed.py` covers both the blocked-market fallback and the normal success path.

The `NFL Model + Operations` workflow now also runs on pull requests that touch this boundary. Pull-request runs execute the full current model and publication validation but do not commit generated state. This gives the outage path an end-to-end live-source check before merge.
