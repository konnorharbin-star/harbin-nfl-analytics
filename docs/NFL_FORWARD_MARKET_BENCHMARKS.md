# NFL Step 5 — Predeclared Market and Flat-Unit Baselines

**Research only. No live betting approval, changed probability model,
retrained classifier, Kelly exposure, new market enablement or threshold.**

The existing Step 4 forward ledger freezes first pre-kickoff 2026
game/market selections, their market price, and the paired two-way
no-vig market probability. Independent grading attaches settled results.
This step runs AFTER grading, never inside the pregame model.

## Benchmarking population: same exact frozen observations

Input files are `reports/forward_edge_graded.csv` and
`reports/forward_edge_validation.json`. The benchmark refuses to run
if the source summary doesn't reconcile to the CSV, if duplicate
game/market rows appear, if a quote or source timestamp is no longer
valid, or if a graded result or payoff doesn't reconcile with the
frozen odds/handicap. Missing or unverified quotes, pending games,
and pushes remain visible in cohort denominators. Pending rows never
receive imputed results.

**Critical limitation:** the stored first side and bookmaker were
chosen by the existing NFL model's maximum raw EV **before kickoff**.
A market-only probability assigned to that *same selected side* is a
conditional forecast accuracy benchmark, **not an independently
selected market-only betting system**. To evaluate an independent
market-only selection rule later, the model must separately pre-freeze
all valid book/side offers before kickoff and prespecify its selection
algorithm. No game is reconstructed from ESPN archives.

## Frozen forecast specifications

For every nonpush, complete quote, the actual binary result is graded
against the following frozen, 2026 pre-specified forecasts on the
**identical decided cohort**:

| Name | Probability on locked selected side |
|---|---|
| `market_only` | Same-quote sportsbook two-way no-vig probability |
| `market_plus_25pct_model` | 75% market, 25% frozen model |
| `market_plus_50pct_model` | 50% market, 50% frozen model |
| `raw_model` | 100% frozen model probability |

No weights are fitted on 2026 outcomes. No weights are selected by
2024/2025 subgroup profits or changed after seeing a game finish.
All four predictors are scored using Brier score, log loss and fixed
10-bin descriptive ECE. Positive
`market_minus_predictor_brier` or
`market_minus_predictor_log_loss` means the predeclared
candidate forecast beat the market-only baseline *on this paired
model-selected cohort*. This is not sufficient proof of a
tradeable edge. ECE is a descriptive finite-sample quantity.

Metrics are split by all markets, moneyline, spread and totals
separately, and also reported by **kickoff week and market**. No subgroup
is mined and promoted from these reports.

## Frozen economic reference policies

All paper rules use the same model-selected first-side offer,
**flat 1 unit** for each qualifying eligible quote, and settled
payoffs at the captured pregame odds:

- `no_bet_market_only`: never place a hypothetical bet; 0 units,
  0 exposure, ROI undefined
- `same_selected_side_flat`: one hypothetical unit per valid,
  graded pregame candidate (model-selected side, NOT market-selected)
- `model_ev_positive`: same side, only frozen raw EV > 0
- `model_edge_2pp_and_positive_ev`: raw EV > 0 and model probability
  minus book no-vig probability >= 2 percentage points
- `model_edge_5pp_and_positive_ev`: raw EV > 0 and difference >= 5 pp

The rules are **fixed upfront and never optimized on completed 2026
games**. Report counts, hypothetical returns per bet and net units
per eligible settled opportunity alongside the no-bet baseline. A
PASS recommendation is included if its predeclared research rule would
have bet; **no actual bettor wager or fill is inferred**, and no
paper stake changes the operational decision ledger.

## Confidence and promotion restrictions

95% **paired, full kickoff-week cluster bootstrap** intervals
are only calculated for a cohort after at least:

- 100 nonpush paired results for Brier/log loss or 100 settled
  opportunities for fixed-rule return;
- 80 distinct games; and
- 8 distinct kickoff weeks.

Each cluster resamples every quote from the same season/kickoff week,
preserving the correlation of multiple markets from one game.
A missing interval means **insufficient evidence**, not an estimated
zero variance. All intervals are unadjusted for the four
forecast and five paper-rule comparisons, are conditional on the
model-selected side, and are research diagnostics, not automatic
promotion tests. These 2026 games cannot be retuned and then
simultaneously cited as an untouched forward holdout.

No model winning claims, new allocations, release-gate changes,
production bet recommendations, or changes to the weekly PNG
are authorized by this report.

## Operations

The existing **NFL Live Shadow Grading** workflow calls
`run_forward_market_benchmarks.py` immediately after the independent
forward-grade step and publishes the following files consistently to
`reports/`, `outputs/`, and `docs/`:

- `forward_market_benchmarks.json`: methodology, evidence status,
  integrity checks and all paired metrics.
- `forward_market_benchmark_forecasts.csv`: overall and market-specific
  baseline/Brier/log-loss comparisons.
- `forward_market_benchmark_policies.csv`: five fixed shadow policies,
  including the no-bet benchmark and full quote/settlement denominators.
- `forward_market_benchmark_weeks.csv`: predeclared kickoff-week
  trends by market and forecast specification.

The grader owns these artifact files; the NFL model workflow
explicitly excludes them from its own commits, just as it does
for Step 4 forward grades. A new workflow push or successful
post-model `workflow_run` refreshes the report.

**Forward-evidence state as of introduction (October 7, 2026,
Central Time):** 45 first-snapshot valid Week 5 game/market
candidates and **0 completed forward games**. Thus forecast
Brier/log-loss figures and realized paper outcomes are
appropriately null/PENDING_FORWARD, not displayed as artificial
accuracy or profitability.

**Next:** collect independent market-first side/quote selections at
the same time as the model candidates, plus enough new week blocks
to evaluate release-gated fixed strategy changes on data that
has not yet been used for model or policy development.
