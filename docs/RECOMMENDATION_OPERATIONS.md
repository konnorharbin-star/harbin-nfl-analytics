# Prospective recommendation operating contract

Implemented 2026-10-10. No predictive weights or release thresholds changed.

## Receipt and publication

`python -m scripts.persist_decisions --sport nfl --source outputs/actionable_betting_board.csv`
(or `--sport cfb --source outputs/edge_priority.csv`) evaluates the existing strict
edge board and adds current model coverage from `docs/latest.csv`.

Immutable records live under `history/recommendations_v1/bets/` and `no_bets/`.
Repeated identical recommendations return the original ID. Each BET has a flat
hypothetical 1-unit stake and retains the exact board row and release evidence.
Incomplete model version, source URL, quote observation time, uncertainty distribution,
or blocked release produces NO BET. Current boards lack some required provenance;
this is a deliberate blocker, not permission to fill it with guesses.

**Before communicating BET, verify the exact receipt in a successful GitHub commit
whose publication time precedes kickoff.** A local file, pending receipt, raw research
signal or BET_READY row is insufficient. Report the receipt ID and quoted price.
The workflow publishes receipts before uploading its reporting artifact. Recalculate
at a changed price and never rewrite an existing receipt.

The legacy portfolio CSV is a separate research ledger. It now appends under an
exclusive writer lock and deduplicates all previous game/market/signature tuples.
Older CSV headers remain intact; complete new rows are retained in immutable
`.events/` sidecars. Legacy PAPER/SHADOW rows are not counted as communicated bets.

## Scheduled grading

`recommendation-records.yml` runs three times daily on free GitHub Actions.
`python -m scripts.grade_recommendations` independently queries public ESPN final
scoreboards only for ungraded existing BET receipts. It requires exact team identity,
final status, plausible integer scores and consistent kickoff. Missing/rescheduled,
ambiguous, conflicting and source-failed outcomes remain ungraded. Moneyline ties
require manual settlement review. No picks are backfilled from outcomes.

Grades are separate immutable events. They require the original receipt's GitHub
creation commit before kickoff. Ordinary result corrections must be explicitly
reviewed and appended as a new correction event; never edit an existing grade.
ESPN's unauthenticated API is undocumented and may become unavailable.

## Dashboard and limitations

`docs/operations.html` shows both repositories' current recommendation counts,
results, units, ROI, drawdown, missing CLV and active release blockers. It links to
permanent receipts and does not turn failed fetches into zero performance.
Research/archive diagnostic results and real-money wagers remain separate.

Closing-quote attachment, new-cohort calibration/market-baseline summaries and
uncertainty intervals remain unfinished. No recommendations currently qualify, so
there is no settled BET receipt on which to verify a real prospective grading cycle.
The scheduled workflow must pass in GitHub before claiming deployed operation.

Highest priorities remain genuine point-in-time free quote/personnel coverage,
validated market-baseline incremental value, and prospective evidence accumulation.
Do not change thresholds to rescue negative archive diagnostics.
