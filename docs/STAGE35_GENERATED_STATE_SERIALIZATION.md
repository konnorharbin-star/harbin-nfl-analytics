# Stage 35 — Generated-state writer serialization

> Superseded by Stage 36. GitHub Actions concurrency retains at most one running and
> one pending run per group, so a single shared writer group can cancel legitimate
> pending writers. Stage 36 restores workflow-scoped concurrency and reconciles
> cross-workflow push races with bounded fetch/rebase/push retries instead.

Stage 35 removes a repository-level race between independent GitHub Actions that write
generated NFL state back to `main`.

The model, grading, line capture, forward-shadow, research-status, and historical
research workflows previously used separate concurrency groups. Two successful jobs
could therefore finish at nearly the same time, both rebase onto the same `main`
commit, and then race to push. The second push could be rejected with a ref-lock or
non-fast-forward error even though its model work was valid.

## Single writer lane

Every workflow that contains `git push` now uses the same generated-state concurrency
key for non-PR runs:

`nfl-generated-state-<branch>`

For normal production automation this resolves to:

`nfl-generated-state-main`

Main-branch generated-state runs use `cancel-in-progress: false` through the event
expression, so a later writer queues behind the active writer instead of cancelling it.

## Pull request behavior

Pull-request validation remains workflow-scoped:

`nfl-<workflow>-pr-<number>`

PR runs use `cancel-in-progress: true`, allowing a newer commit on the same PR to
supersede stale validation without blocking unrelated PR checks.

## Workflow-run behavior

For workflows triggered by another workflow, the source workflow's
`head_branch` determines the generated-state lane. A workflow-run sourced from
`main` therefore joins the same `nfl-generated-state-main` queue as scheduled,
push, and manual main-branch writers.

## Protected writer inventory

`tests/test_workflow_writer_serialization.py` discovers every workflow containing
`git push` and requires it to be registered and to implement the shared concurrency
contract.

The current writer inventory is:

- candidate benchmark;
- forward ledger health;
- forward shadow summary;
- free market backtest;
- line capture;
- live grading;
- market-edge shrinkage;
- model + operations;
- probability forward capture;
- probability forward grading;
- QB-total forward capture;
- QB-total forward grading;
- research status.

Adding a new writer without updating the serialization contract fails CI.

## Model boundary

Stage 35 changes orchestration only. It does not alter fair scores, probabilities,
market edges, betting thresholds, portfolio sizing, evidence thresholds, or the PAPER
release state.
