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


## Real-source result

The real nflverse audit reconstructed 1,039 canonical pregame games from 2021–2025 and
reused the Stage 23 whole-week partition: 607 Core, 147 Tune, 134 Calibration, and 151
untouched Evaluation rows.

Logistic alpha selected before evaluation was 0.0. On the 151-game evaluation block:

- raw logistic Brier: 0.229127;
- isotonic Brier: 0.229905, a regression of 0.000777;
- raw logistic log loss: 0.648513;
- isotonic log loss: 0.644847, an improvement of 0.003666;
- raw logistic ECE: 0.086355;
- isotonic ECE: 0.083457, an improvement of 0.002899.

Because Brier score worsened, the isotonic layer fails the fixed all-three-metrics gate.
candidate_pass=false, canonical_probability_change_enabled=false, and
promotion_eligible=false remain enforced.

Stage 25 therefore creates no new 2026 shadow and does not alter the already-frozen
Stage 24 logistic/total-dispersion probability experiment.
