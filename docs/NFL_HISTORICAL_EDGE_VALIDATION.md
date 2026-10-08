# NFL Step 3 — Fixed Historical Edge Failure Diagnosis

This phase investigates **why positive raw expected value does not survive
historical results**. It adds a reproducible, read-only research audit over
already-frozen NFL 2024 validation and 2025 diagnostic holdout results.
It does not optimize new betting thresholds or revise football forecasts.

## Evidence

Inputs:
- reports/verified_market_bets.csv
- reports/verified_market_backtest.json
- reports/market_edge_shrinkage.json
- reports/regime_edge_reliability.json

**Important distinction:** The "verified" archived dataset stores ESPN
provider-labeled *opening and closing stages*; entry_price_verified means
the provider supplied an archive price. It does NOT imply that the price was
independently captured from a sportsbook before kickoff. In the 2024/25 data,
entry_timestamp_verified is false. Therefore observed hypothetical betting
profits/ROI and archived CLV proxies are not executable realized profits or
certified closing-line value. The source's false timestamp flags are
reported explicitly and force fail-closed evidence status.

Model probabilities are derived from the separate chronologically trained
football distribution. No betting-market information enters fair-score
training. The audit runs AFTER those independent projections and ONLY reads
previously committed historical outcomes and market prices.

## Frozen segments and leakage controls

No parameters or segments are chosen by 2025 performance. Each game/market
has one selected archived best-quote candidate. Identical game/market/season
duplicates fail rather than contributing multiple apparent wins. Invalid
probabilities, inconsistent model edge/EV arithmetic and inconsistent unit
payoffs are EXCLUDED, counted and cause a fail-closed report status.

Predeclared descriptive cohorts:
- All selected archived game-market candidates
- 2024 validation and 2025 **already-inspected** diagnostic holdout separately
- Moneyline, spread, total separately
- All selected candidates versus fixed positive-raw-edge sample
  (probability_edge > 0 AND expected_value > 0)
- Market-specific raw edge bands: nonpositive, (0,2pp), [2,5pp),
  [5,10pp), [10,20pp), 20pp+
- Market-specific modeled win probability: below 50%, [50,55),
  [55,60), [60,65), [65,70), 70%+

Group results are observational checks, not promotion rules. **The
historical 2025 games have already been analyzed repeatedly**: any newly
identified subgroup requires separately collected future data and cannot
claim untouched holdout validation.

## What gets measured

For each cohort:

- Mean raw modeled win probability and observed **nonpush** win frequency
- Same-row two-way no-vig sportsbook probability and realized-minus-market
  win-frequency gap
- Probability overconfidence (mean predicted probability minus win rate)
- Paired Brier and log-loss error of model versus **the same matched
  sportsbook no-vig probability**, on identical nonpush cases
- Positive model edge retention relative to observed nonpush results
- Mean raw model EV, hypothetical archive-price ROI and the **EV
  realization gap** (model's forecast EV minus observed archived payoff)
- No-vig-market-only EV at the same bookmaker's quoted odds, to expose vig
- Push accounting, games, weeks, quote timestamp coverage and integrity

Three 95% uncertainty intervals use deterministic **season × kickoff-week
block bootstrap**, resampling entire weeks instead of independent tickets
which may share a game. Intervals are withheld until 100 cases, 60 distinct
games and 8 week blocks. They are descriptive conditional-on-selected-
archived-prices intervals, not multiplicity-adjusted proof of winning.
Positive-looking subgroups are **not** used to select a betting policy.

The audit flags:
- Model overconfidence relative to observed results
- Model Brier performance worse than the no-vig market
- Positive raw EV accompanied by negative archived simulated ROI
- Gap between the model's projected EV and archived unit returns
- Zero holdout-selected NFL model-to-market weight (alpha=0)
- NFL market-regime unreliability
- Absence of independently timestamped executable entries

## Operational report and guardrails

The dedicated research workflow runs on PR/push, manual invocation,
weekly schedule, and completed upstream model/reliability evidence. It
publishes matching artifacts to reports/, outputs/, and docs/:

- historical_edge_failure.json: global diagnostics, per-market failure
  attribution and fixed descriptive cohort results
- historical_edge_segments.csv: auditable summary table with descriptive
  metrics, withheld/available block-bootstrap confidence intervals and
  provenance status

The NFL model's generated README links these reports. Neither the
canonical projection, market probabilities, signal policy, thresholds,
market enablement, NFL release gate, portfolio stakes, live bet ledger nor
published moneyline/spread/total selections are modified.

**Current evidence priority:** fix calibration and historical price
provenance before treating any raw 10%+ EV claim as an actionable
sportsbook edge. Independent contemporaneous pricing and new forward
evaluation are prerequisites for meaningful performance claims.

**Next step (separate phase):** forward-register NFL edge hypotheses and
collect matched pre-kickoff, same-book entry and market-close snapshots
to test calibration, price availability, missed wagers, and the market-only
benchmark. No historical subgroup is auto-promoted.
