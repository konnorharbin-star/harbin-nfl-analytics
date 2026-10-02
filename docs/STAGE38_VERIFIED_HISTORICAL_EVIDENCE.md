# Stage 38 — Verified historical market evidence bridge

Stage 38 connects the existing optional timestamped historical sportsbook adapter to
the canonical NFL proof, policy-calibration, and release-gate path.

The free nflverse archive remains the default research source. It is still useful for
broad historical diagnostics, but archive-final prices and opening lines without the
actual entry juice cannot satisfy promotion-quality entry-price evidence.

## Optional provider boundary

The timestamped path uses the existing The Odds API historical adapter and requires an
explicit API key. It is not required for the free PAPER model, is never invoked by pull
requests or schedules, and is only run through an explicit manual workflow dispatch.

Provider responses are cached locally for reproducibility during a run. API credentials
are never written to reports or committed.

## Frozen historical decision policy

The verified backtest uses a fixed, predeclared timing contract:

- entry query: 60 minutes before kickoff;
- closing query: 5 minutes before kickoff;
- selected quote must be no more than 15 minutes old at the simulated decision time;
- every decision and provider snapshot is timezone-aware;
- the latest complete two-way snapshot available at the decision time is selected;
- later quotes are never eligible for the entry decision.

These timestamps are research-policy inputs, not tuned on historical outcomes.

## Football-model separation

The football projection dataset is rebuilt with the same leak-free weekly fair-score
path used by the free archive backtest.

For each historical week:

1. fit the football distribution only on earlier games;
2. select timestamped sportsbook entry snapshots as of the frozen decision time;
3. compare the already-built football distribution with each available book;
4. line-shop the best executable side per game/market;
5. grade at the exact observed entry price;
6. attach same-book later pre-kickoff closing CLV.

Sportsbook values remain downstream and never enter the fair-score features.

## Promotion sample

The canonical proof layer now combines only rows with explicit verified entry-price and
entry-quote provenance. Research-only archive fallbacks remain excluded.

ROBUST historical evidence still requires:

- at least 1,000 verified bets;
- positive lower bound of the verified-sample ROI confidence interval;
- positive average verified CLV;
- at least 90% verified historical CLV coverage;
- positive evidence in at least two markets;
- positive evidence in at least two seasons.

Stage 38 does not weaken any of those gates.

## Nested policy calibration

The same verified rows are available to the existing nested chronological policy
calibrator. Development/tuning data may select thresholds; the later evaluation sample
remains untouched until release evaluation.

A verified provider sample does not automatically enable a market or production. The
existing evaluation and release gates still decide that.

## Automation

`NFL Verified Historical Market Backtest` has two modes:

- pull request: fixture/unit validation only, with no external historical API calls;
- manual dispatch: requires `THE_ODDS_API_KEY`, fetches/caches timestamped snapshots,
  writes the verified historical evidence, refreshes canonical proof/policy files, and
  commits them through the shared generated-state push reconciler.

A successful manual run triggers `NFL Model + Operations` so the release gate and
publication state are recalculated promptly.

## Outputs

- `reports/verified_market_bets.csv`
- `reports/verified_market_backtest.json`
- `reports/evidence_report.json`
- `reports/production_policy.json`

## Release boundary

Free operation remains fully functional in PAPER without this optional provider.
PRODUCTION remains locked until historical verified evidence, current multi-book
breadth, forward sample/ROI/CLV, engineering, and all operational gates pass.
