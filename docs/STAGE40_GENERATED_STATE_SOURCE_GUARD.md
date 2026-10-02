# Stage 40 — Generated-state source revision guard

Stage 40 closes a race that remained after Stage 36 push reconciliation.

Stage 36 correctly allowed concurrent generated-state writers to fetch, rebase, and retry
when another generated writer reached `main` first. It also failed closed on true
content conflicts.

A different race remained possible:

1. a workflow starts from source revision A;
2. source/workflow/test changes are merged as revision B;
3. the old workflow finishes later and commits generated outputs built with revision A;
4. the reconciler rebases that generated commit onto revision B;
5. stale artifacts can briefly overwrite outputs produced by newer code.

That happened after Stage 39. The older model run published first; the newer Stage 39
run correctly refused a conflicting rebase, leaving the branch temporarily with stale
generated reports even though the Stage 39 source code was present.

## Source-revision contract

Every generated-state push now records the parent of its generated commit as the
writer's source revision.

Before rebasing, the shared push helper fetches the target branch and inspects all files
changed between that source revision and the current target head.

The writer may continue only when every intervening change is generated state:

- `outputs/**`
- `history/**`
- `reports/**`
- generated publication files under `docs/` with JSON, HTML, CSV, or PNG extensions

If any newer source, workflow, script, test, or documentation-source file is present,
the generated commit is considered superseded. The helper exits successfully without
publishing stale artifacts.

## Why superseded writers exit successfully

A stale writer is not an operational failure: a newer source revision already exists
and will trigger its own canonical refresh. Treating supersession as success avoids
retry storms while preventing old calculations from crossing a newer source boundary.

## Existing reconciliation behavior retained

Stage 40 does not remove Stage 36 behavior:

- non-conflicting generated-state writers still rebase and retry;
- true generated-file content conflicts still fail closed;
- no writer force-pushes or overwrites another writer.

## Model boundary

This is orchestration only. It changes no football model, probability, market signal,
policy threshold, portfolio rule, evidence gate, or publication schema.
