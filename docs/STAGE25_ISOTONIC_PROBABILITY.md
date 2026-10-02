# Stage 25 — Isotonic home-win calibration research

Stage 25 tests one remaining probability-model concept used by the NCAA implementation:
an isotonic calibration map layered on top of logistic home-win probabilities.

This is an NFL-specific research test. It does not copy a CFB fitted calibrator,
release decision, betting threshold, or profitability conclusion.

## Chronological structure

The canonical 2021–2025 pregame projection history is split on complete NFL weeks into:

1. Core — fit logistic candidates.
2. Tune — select logistic regularization.
3. Calibration — fit the selected logistic model and isotonic probability map.
4. Evaluation — score the frozen calibration once.

The evaluation block cannot select logistic alpha, fit the logistic model, or fit the
isotonic map.

## Candidate

The only input is the independent projected home margin. Sportsbook lines, odds,
market probabilities, CLV, graded bets, and current-only context are excluded.

Logistic alpha is selected from 0, 0.1, 1, 10, and 100 on the Tune block. The chosen
logistic form is fit on the Calibration block. Its calibration-block probabilities are
then mapped to observed home-win outcomes using monotonic isotonic regression. Both
raw logistic and isotonic probabilities are scored on the untouched Evaluation block.

## Fixed gate

The isotonic layer survives development only if it improves all three probability
diagnostics on untouched evaluation:

- Brier score;
- log loss;
- expected calibration error (ECE).

A mixed result is a failure. The raw logistic model remains the development candidate
unless isotonic clears all three metrics.

## Evidence boundary

Stage 25 hard-codes:

- selection_uses_evaluation=false;
- canonical_probability_change_enabled=false;
- promotion_eligible=false.

The 2021–2025 sample is development evidence only. Any surviving isotonic layer would
require a separately frozen prospective 2026 ledger. The existing Stage 24 probability
shadow remains unchanged while this audit runs.
