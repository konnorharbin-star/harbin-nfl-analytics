# Stage 34 — Forward betting evidence integrity

Stage 34 hardens the prospective betting-evidence contract used by Phase 5 and the
production release gate. It does not change the canonical fair-score model, market
probabilities, betting thresholds, or PAPER deployment state.

## Executable entry provenance

A PAPER, SHADOW, or BET row can enter the graded forward sample only when all of the
following are true:

- `portfolio_candidate_units > 0`;
- `portfolio_action` is PAPER, SHADOW, or BET;
- `execution_ready` is true;
- `quant_quote_at` exists;
- `decision_at` exists;
- `kickoff` exists;
- the quote timestamp is at or before the decision timestamp;
- the decision timestamp is strictly before kickoff;
- the quote timestamp is strictly before kickoff.

Rows that do not satisfy this contract are not graded. The independent grading audit
also treats invalid quote chronology or non-execution-ready evidence as an error.

## Closing-line coverage

Positive average CLV is not sufficient by itself. A small subset of closing snapshots
cannot represent a large forward sample.

Production-quality forward evidence therefore requires:

- at least 300 independently graded portfolio decisions;
- 100% verified entry-quote coverage;
- 100% execution-ready coverage;
- at least 90% of graded bets with a valid later pre-kickoff closing snapshot;
- non-negative forward ROI;
- positive average execution CLV.

The 90% threshold is evaluated as `clv_samples / graded_bets`.

## Reporting

`reports/live_performance.json` and the Phase 5 summary expose:

- `entry_quote_verified_bets`;
- `entry_quote_coverage`;
- `execution_ready_bets`;
- `execution_ready_coverage`;
- `clv_samples`;
- `clv_coverage`.

`reports/grading_audit.json` is persisted with live grading so timing and provenance
violations remain independently inspectable.

## Fail-closed behavior

Missing coverage fields are treated as zero coverage. Historical backtests do not
substitute for this evidence, and the release gate cannot infer CLV quality from only a
few available closes.
