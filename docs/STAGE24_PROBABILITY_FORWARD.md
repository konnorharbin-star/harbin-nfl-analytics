# Stage 24 — Prospective probability-calibration shadow

Stage 24 converts the Stage 23 probability findings into true 2026 forward evidence.
It does not change the canonical score model, canonical probabilities, market edges,
policy thresholds, portfolio allocation, or stake sizing.

## Frozen Stage 23 forms

Stage 23 selected all hyperparameters before its untouched evaluation block:

- margin Gaussian residual scale: 1.0;
- total Gaussian residual scale: 0.9;
- home-win logistic regularization alpha: 0.0.

The untouched Stage 23 evaluation showed only development-level improvements: total
Gaussian NLL improved slightly, and the logistic home-win model modestly improved Brier
score and log loss versus the Gaussian home-win probability. Those results do not
justify retrospective promotion.

For Stage 24, the selected forms are frozen. They are refit once on canonical pregame
projections from 2021–2025, all of which precede the 2026 forward season. No 2026
outcome may alter the forms, scales, alpha, or training window.

## Prospective ledger

The capture job writes the first valid pre-kickoff snapshot for each 2026 game to
history/probability_shadow_predictions_v1.csv.

Each row stores:

- capture and kickoff timestamps;
- canonical baseline margin and total;
- baseline Gaussian home-win probability;
- frozen logistic home-win probability;
- baseline total-distribution mean and sigma;
- frozen 0.9-scale total-distribution mean and sigma;
- frozen specification and training-season provenance.

Post-kickoff snapshots, wrong seasons, incomplete rows, and duplicate game/spec
snapshots are rejected. Pull-request runs may produce artifacts but cannot persist
forward evidence to main.

## Independent forward gates

Home-win calibration and total-distribution calibration are graded independently.

### Home-win gate

The persisted Gaussian and logistic probabilities are compared on the same non-tied
games using:

- Brier score;
- log loss;
- ECE for reporting;
- paired bootstrap intervals for Brier and log-loss improvement.

### Total-distribution gate

The unscaled and 0.9-scale Gaussian total distributions are compared using per-game
negative log likelihood and a paired bootstrap interval for NLL improvement.

Each side requires at least 128 forward games before it can produce promotion evidence.
A positive point estimate alone is insufficient. Promotion evidence requires every
required paired 95% lower confidence bound for that side to be above zero.

## Evidence boundary

The grader never reconstructs historical forward predictions. It only grades rows that
were persisted before kickoff under the exact frozen Stage 23 specification.

Even if a forward gate eventually reports PROMOTION_EVIDENCE, Stage 24 still hard-codes
canonical_probability_change_enabled=false. Promotion would require a separate explicit
model-release change after the evidence is reviewed.

The current betting model therefore remains unchanged while the 2026 probability
experiment accumulates prospectively.
