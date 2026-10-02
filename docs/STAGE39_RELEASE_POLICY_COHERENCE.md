# Stage 39 — Release/policy state coherence

Stage 39 closes a Phase 6 state-machine mismatch between the hard release gate and the
portfolio allocator.

Before this stage, the allocator required both:

- `release_gate.production_eligible == true`; and
- `production_policy.deployment_mode == "production"`.

The hard release gate itself did not check the calibrated policy mode. In a future
fully-mature evidence state, the release report could therefore claim PRODUCTION while
the allocator still refused real stakes because the frozen nested policy remained
PAPER.

## Production policy gate

The hard release gate now reads `reports/production_policy.json` and requires:

- `deployment_mode == "production"`; and
- at least two enabled markets in the frozen policy.

This is a separate check named `production_policy`.

The policy is still created only by the existing nested chronological calibration:

1. development data defines candidate threshold grids;
2. tuning data selects the frozen rule;
3. untouched evaluation data decides whether that market remains enabled;
4. at least two markets must pass before the calibrated policy can enter production
   mode.

The release gate does not retune or reinterpret that result.

## Release states

The state machine is now coherent:

- RESEARCH: engineering/data prerequisites fail;
- PAPER: historical verified-entry evidence is not yet robust;
- SHADOW: historical evidence may be robust, but one or more production prerequisites
  such as policy validation, multi-book breadth, or forward evidence still fail;
- PRODUCTION: engineering, verified historical evidence, validated frozen policy,
  current market breadth, and forward portfolio evidence all pass.

## Allocator parity

The allocator already required the production policy mode. Stage 39 makes the release
report and model card express the same prerequisite, so a published
`production_eligible=true` can no longer disagree with executable staking logic.

## Model card

The model card now exposes:

- `production_policy_mode`;
- `production_policy_ready`.

## Model boundary

This stage changes release-state accounting only. It does not change football scores,
probabilities, market prices, threshold selection, Kelly sizing, portfolio caps, or any
sample/ROI/CLV requirement.
