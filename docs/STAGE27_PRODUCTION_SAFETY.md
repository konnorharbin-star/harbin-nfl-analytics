# Stage 27 — Production safety hardening

Stage 27 closes the remaining Phase 6 execution-state gap found during the production
audit. The release gate, policy, bankroll throttle, execution validator, concentration
caps, decision ledger, grading and publication systems were already present. This
stage makes their interaction fail closed when a nominal production release later
encounters an execution or bankroll safety failure.

## Explicit execution minimum

The NFL policy now carries
`min_market_book_count_for_execution=1` explicitly rather than relying on the
execution validator's fallback default. A future evidence-backed policy can raise this
minimum without changing execution code.

This per-market execution minimum is separate from the release gate's broader
multi-book coverage requirement.

## Production quote failure

Before Stage 27, an invalid production quote could be denied real stake but still
receive paper allocation and consume game/team/market/book/slate capacity.

That is no longer allowed.

When the production gate and production policy are both open:

- non-executable or stale quotes remain `PASS`;
- candidate allocation remains zero;
- real stake remains zero;
- the quote cannot consume concentration capacity;
- the execution failure is preserved in `portfolio_limit_reason`.

Paper and shadow research retain their ability to exercise portfolio planning without
real stake.

## Explicit HALTED mode

A production release can be invalidated downstream by bankroll safety state. Examples
include a missing required live ledger or the drawdown hard stop.

The portfolio summary now distinguishes this state:

```text
production gate open
+ production policy active
+ bankroll/history safety failure
= mode: halted
```

The summary also records:

- `production_gate_open`;
- `production_eligible` after downstream safety checks;
- `production_block_reason`.

In HALTED mode no candidate or real-stake allocation is permitted.

## Committed exposure across repeated runs

Production concentration limits now include already-authorized future `BET` decisions
from `history/portfolio_decisions_v1.csv`. This prevents a later model refresh from
treating earlier open bets as if their risk had disappeared.

For the current target season/week, prior future BET rows reserve:

- slate/week units;
- game units;
- selected-team exposure;
- market exposure;
- sportsbook exposure;
- kickoff-window exposure;
- max-bet-count capacity.

An already-committed game/market cannot receive a second production stake on a later
run. Completed/past-kickoff rows do not reserve future capacity.

Production also requires the committed-exposure ledger to be readable and internally
valid. If that ledger is missing or contains an invalid open BET row for the target
period, an otherwise open production release becomes `HALTED` rather than assuming
zero prior exposure.

## Regression coverage

The operational parity tests now prove that:

1. a stale production quote receives zero candidate units and stays PASS;
2. a production bankroll hard stop reports HALTED;
3. a missing required committed-exposure ledger reports HALTED;
4. a prior future BET reserves next-run slate/correlation capacity;
5. the same game/market cannot be re-approved while that BET remains open;
6. the explicit market-book and committed-ledger requirements remain policy defaults.

These are release-safety tests, not betting-edge evidence.

## Phase 6 status

With Stage 27, the production shell contains the controls specified by the build plan:

- frozen policy loading;
- fractional Kelly;
- game/team/market/book/kickoff/slate concentration caps;
- drawdown and adverse-run throttles;
- executable-book and timestamp requirements;
- stale-quote protection;
- explicit downstream production HALT;
- decision ledger with repeated-run committed exposure reservation;
- independent grading and CLV;
- monitoring/health/audit snapshots;
- publication reconciliation;
- hard RESEARCH -> PAPER -> SHADOW -> PRODUCTION release gates.

Architecture completion does not mean production release. The model remains locked
until its independent NFL evidence gates pass.
