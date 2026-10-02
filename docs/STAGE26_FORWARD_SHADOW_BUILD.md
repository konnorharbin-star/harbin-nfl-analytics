# Stage 26 — Consolidated Phase 5 forward-shadow build

Stage 26 turns the existing prospective NFL evidence streams into one explicit Phase 5
acceptance report. It does not create predictions, rebuild historical shadows, retune a
candidate, or change the canonical score/probability model.

## Inputs

The consolidated report consumes only already-persisted grading outputs:

- `reports/live_performance.json` — portfolio-verified canonical betting decisions;
- `reports/recent_form_forward.json` — frozen recent-form total shadow;
- `reports/qb_total_forward.json` — frozen QB-total shadow;
- `reports/probability_forward.json` — frozen home-win and total-distribution
  probability shadows.

The underlying ledgers remain independently owned by their existing capture and grade
jobs. Stage 26 is read-only with respect to those ledgers.

## Forward evidence rules

Canonical betting evidence requires at least 300 graded portfolio decisions. The gate
then requires non-negative ROI and positive average execution CLV. A reconstructed
historical backtest cannot satisfy this gate.

Each score/probability shadow retains its own minimum sample and statistical gate. The
current frozen candidates require at least 128 prospectively graded games. Stage 26
will not honor a `promotion_eligible=true` flag when the persisted graded sample is
below that candidate's minimum.

The probability experiment remains split into two independent targets:

1. home-win probability calibration;
2. total-distribution calibration.

One target can earn promotion evidence without the other.

## Phase-level output

`nfl/forward_shadow.py` writes:

- `reports/forward_shadow_summary.json`;
- `docs/forward_shadow_summary.json`.

The summary records:

- canonical forward betting sample, ROI, CLV, drawdown, and gate status;
- every frozen candidate's prospective ledger/sample state;
- candidate-specific promotion evidence;
- whether the Phase 5 forward betting gate is ready for release review;
- the explicit prohibition on retrospective reconstruction and forward-outcome
  retuning.

`READY_FOR_RELEASE_REVIEW` means only that the canonical forward betting evidence
satisfied the Phase 5 sample/ROI/CLV rule. It is not a production release.

## Release separation

Stage 26 has no production authority. It hard-codes:

- `production_release_authority=false`;
- `canonical_model_change_enabled=false`.

A shadow candidate that earns promotion evidence still requires a separate explicit
model-release change. Real staking remains controlled by `nfl/release_gate.py` and
its engineering, market-breadth, historical-entry, and live-evidence requirements.

## Automation

`NFL Forward Shadow Summary` runs after each of the three forward grading workflows,
on relevant main-branch report changes, on a schedule, and manually. Pull requests run
the integrity tests and build the report without committing evidence.

This completes the Phase 5 reporting contract while allowing the 2026 prospective
samples to continue accumulating naturally.
