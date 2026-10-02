# Stage 36 — Generated-state push reconciliation

Stage 36 replaces the Stage 35 shared-concurrency writer lane after live post-merge
validation showed that GitHub Actions keeps only one running and one pending run in a
concurrency group. Additional pending writer workflows were cancelled, which prevented
push races but also dropped legitimate refresh work.

## Workflow-scoped concurrency

Each generated-state writer again uses its own concurrency group, scoped to the source
branch. A newer run of the same workflow may replace an older one, but unrelated writer
workflows can all execute.

This is appropriate because repeated runs of the same writer are superseding refreshes,
while different writers own different evidence or operational products.

## Shared push reconciler

Every writer commits its generated changes locally and then calls:

`bash scripts/push_generated_state.sh`

The helper performs a bounded reconciliation loop:

1. fetch the latest target branch;
2. rebase the local generated-state commit onto the fetched tip;
3. attempt an explicit `HEAD:<branch>` push;
4. if another writer wins the push race, fetch/rebase and retry;
5. if the rebase encounters a real content conflict, abort and fail rather than
   choosing a side or overwriting another writer.

The default retry budget is eight attempts with short increasing backoff.

## Target branch

The helper accepts an explicit target branch and otherwise resolves from the GitHub
Actions ref environment. Normal automated operation writes `main`; manual dispatches
on another branch retain branch-local behavior.

## Regression tests

`tests/test_generated_state_push_reconciliation.py` protects three properties:

- every workflow that writes generated state uses the shared reconciler;
- two stale writers with non-overlapping files are both preserved after reconciliation;
- two writers editing the same file cause a fail-closed conflict and the already-pushed
  remote version is not overwritten.

The writer inventory remains explicit so adding a new generated-state writer without
the reconciler fails CI.

## Model boundary

Stage 36 changes repository write coordination only. It does not change fair scores,
probabilities, market inputs, betting thresholds, portfolio sizing, evidence thresholds,
or the PAPER/SHADOW/PRODUCTION release logic.
