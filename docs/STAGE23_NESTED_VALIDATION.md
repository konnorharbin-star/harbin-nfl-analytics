# Stage 23 — Whole-week nested probability validation

Stage 23 tightens the NFL validation architecture without changing the canonical score or probability model.

The existing NFL research stack already uses chronological season-level holdouts. This stage adds the stricter four-block structure used by the NCAA modeling workflow while preserving NFL-specific estimation and evidence.

## Whole-week partition

Historical canonical pregame projections are sorted by season and week and divided into four complete-week sections:

1. Core — fit candidate hyperparameter models.
2. Tune — select probability-dispersion scales and logistic regularization.
3. Calibration — fit the selected Gaussian residual distribution and home-win logistic calibrator.
4. Evaluation — untouched until all selections and calibration fits are fixed.

The approximate row targets are 58% / 14% / 14% / 14%, but boundaries move to complete NFL weeks. A season/week block can never be split across two sections.

## Selection boundary

The tune block chooses Gaussian margin residual scale, Gaussian total residual scale, and logistic home-win regularization. The final evaluation block is not visible to those choices.

A regression test mutates only evaluation outcomes and requires the selected scales and logistic alpha to remain identical.

## Calibration boundary

After hyperparameters are selected, the chosen Gaussian score distribution and logistic home-win model are fit on the dedicated calibration block only.

The final evaluation block reports margin and total Gaussian negative log likelihood, 50% and 80% interval coverage, Gaussian home-win Brier/log loss/ECE, logistic home-win Brier/log loss/ECE, selected-vs-unscaled Gaussian NLL differences, and logistic-vs-Gaussian home-win differences.

Evaluation results are reporting-only. They cannot retune hyperparameters or refit the calibration objects being scored.

## Production boundary

Stage 23 is a validation-architecture audit, not a production promotion. It hard-codes selection_uses_evaluation=false, canonical_probability_change_enabled=false, and promotion_eligible=false.

Any future production probability change must be proposed separately after the Stage 23 evidence is recorded. The current canonical model, market edges, policy thresholds, and stake sizing are untouched.

## Real-source result

The 2021–2025 real-source audit reconstructed 1,039 canonical pregame games and produced complete-week partitions of 607 Core rows, 147 Tune rows, 134 Calibration rows, and 151 untouched Evaluation rows.

Hyperparameters selected before evaluation were:

- margin Gaussian scale: 1.0;
- total Gaussian scale: 0.9;
- logistic win regularization alpha: 0.0.

On the untouched 151-game evaluation block, selected Gaussian margin NLL was unchanged at 3.96630 because the selected margin scale remained 1.0. Total NLL improved slightly from 4.03286 to 4.03231 with the 0.9 total scale.

Gaussian home-win calibration reported Brier 0.23065, log loss 0.65123, and ECE 0.09639. The separately calibrated logistic model reported Brier 0.22913, log loss 0.64851, and ECE 0.08636. Relative to Gaussian, logistic improved Brier by 0.00152 and log loss by 0.00272 on this evaluation block.

Selected Gaussian interval coverage was 53.6% for the nominal 50% margin interval and 81.5% for the nominal 80% margin interval. Total coverage was 51.7% and 75.5% for the nominal 50% and 80% intervals, respectively.

These results validate the chronology architecture and provide development evidence for future probability research. They do not change the canonical probability model: selection_uses_evaluation=false, canonical_probability_change_enabled=false, and promotion_eligible=false remain enforced.
