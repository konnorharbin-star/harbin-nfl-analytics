# Stage 33 — Prospective forward-ledger health

Stage 33 adds an integrity gate around every frozen 2026 model-change candidate.

The forward ledgers already reject post-kickoff writes and preserve first snapshots.
This stage adds a cross-ledger audit so a future promotion sample also requires complete
capture coverage after each candidate actually begins.

## Candidate inception

Candidates did not all start at the same instant.

For each ledger, inception is the earliest valid persisted `captured_at`. A game that
already kicked off before that timestamp is outside that candidate's prospective sample
and is never reconstructed or counted as missing.

This matters in the current Week 4 evidence:

- recent-form and QB-total were captured before PIT-CLE and therefore opened all 16 games;
- the probability shadow began after PIT-CLE kickoff and therefore legitimately opens
  only the remaining 15 games.

## Opened-week coverage

Once a candidate has at least one valid snapshot in a season/week, the audit requires
exactly one valid frozen pre-kickoff row for every regular-season game in that same week
whose kickoff is later than candidate inception.

The audit reports:

- inception timestamp;
- opened season/week blocks;
- eligible schedule games;
- captured eligible games;
- capture coverage;
- missing game IDs;
- unexpected game IDs;
- invalid timing rows;
- invalid specification rows;
- duplicate rows.

## Promotion gate

Candidate promotion now requires `forward_ledger_health=PASS`.

A candidate with otherwise positive forward metrics remains non-promotable if its
ledger has missing, duplicate, late, wrong-spec, or schedule-inconsistent rows.

This does not change the canonical fair score, probability model, market policy, or
PAPER release state.

## Outputs

- `reports/forward_ledger_health.json`
- `docs/forward_ledger_health.json`

The audit is refreshed automatically when any frozen candidate ledger changes and on a
scheduled backstop. The Phase 5 forward-shadow summary then refreshes from that result.
