# NFL Step 2 — Observed Same-Book Timing (Research Only)

The NFL model previously had downstream BET_NOW / WAIT / PASS diagnostics.
That legacy field remains unchanged and is not a verified betting strategy.
It could label BET_NOW when **no prior same-book quote existed**.
This phase adds a separate, evidence-gated **edge_timing_action**.

## Chronological evidence

The new NFL timing module consumes the verified history in
history/market_snapshots.csv. A usable prior quote must:

- Match game, market, selected side and canonical sportsbook. Never mix books.
- Have an aware capture timestamp at least 10 minutes and at most 12 hours
  before the selected current quote, and no later than the model decision time.
- Be strictly before kickoff, with the same game's kickoff within 60 seconds.
- Have valid American odds and an observed line for spreads/totals.

It chooses the **latest eligible prior capture**, not the best price.
Capture time indicates when the system observed the quote; it does NOT
guarantee the sportsbook actually offered or held that price then.
No post-kickoff observations or official closing-line claims are made.

## Direction of movement

- Spread: current selected-side spread minus previous selected-side spread
- Total OVER: previous total minus current total
- Total UNDER: current total minus previous total
- Moneyline and same-line price: the reduction in implied break-even
  probability from previous American odds to current odds, in percentage points

A move is material at 0.5 spread/total points or 1 percentage point of
implied break-even price. Contradictory material spread and odds movement
is classified MIXED_LINE_PRICE and does not justify a timing action.

## Three research-only outcomes

- **BET_NOW_RESEARCH** requires a Step 1 supported NFL edge, contemporaneous
  verified and sane sportsbook quote, reliable QB and injury context,
  positively weighted NFL chronological holdout shrinkage, a conservative
  same-line price cushion and materially worsening observed same-book terms.
- **WAIT_MONITOR** requires the same evidence gates and improving terms.
  Improvement is observed, not assumed to persist.
- **PASS** is the default on unvalidated edges, stale or missing prices,
  unavailable prior observations, conflicting line/odds moves, or no material
  movement.

All are shadow observations, not instructions to place bets. Every row
explicitly sets edge_timing_staking_authorized=false.

The NFL 2024/2025 validation currently selects market-only alpha=0 and
marks all three market regimes unreliable. Thus current live timing
signals should correctly be PASS even when the raw displayed model EV
appears very positive.

## Price limits

Only a qualifying supported candidate can have
edge_timing_same_line_min_american_odds calculated from its
heldout-shrunk probability. It requires at least 1% probability-based
expected value and a 2 percentage point cushion over the American
odds break-even probability at **the current selected line**.

The model does **not** infer a bet-to spread or total line. Repricing
a different line requires calculating the football probability again
at that line. Hence edge_timing_bet_to_line is deliberately null.

These price cushions and line movement thresholds are diagnostic,
not historically validated trading thresholds.

## Publication and safeguards

Every model run publishes three identical artifacts to outputs/ and docs/:
edge_timing.csv (all assessed rows), edge_timing_signals.csv (qualified
research-only timing observations; may be empty) and
edge_timing_report.json (coverage, action counts, source limitations).

Timing metadata is included under meta.edge_timing in current_model.json,
and README links the new reports. The enrichment happens **after**
NFL portfolio allocation and after Step 1 evidence triage.
It does not change historical data, projections, production staking,
portfolio actions, previous decision-intelligence actions, or release gates.

Next phase: forward-register the first timing decision and compare
same-book future quotes with predeclared timing baselines. No observed
movement alone proves executable profit, positive CLV, or future trends.
