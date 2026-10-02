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
