# NFL Step 1 — evidence-first market edge discovery

**Research-only downstream diagnostic. Does not create bets or alter the release gate.**

The existing NFL fair-score projection, probability distribution, two-way no-vig
market comparison, historical regime validation and conservative production
policy remain independent. This addition adds a transparent *evidence triage*
to the already selected best-book candidate for each game and market. It is
not a fresh hyperparameter search, a refitted score model or a new validated
trading strategy.

## What is and is not an edge

- **Raw model probability edge** (`quant_edge`) = model win/cover/total
  probability minus the paired book's no-vig probability. It is not a
  realized historical advantage.
- **Raw model EV** (`quant_ev`) uses the corresponding quoted American odds.
  Large positive numbers can arise from miscalibrated model probabilities.
- **NFL fixed chronological shrinkage** uses the alpha chosen by the
  independent NFL 2024 validation/untouched 2025 holdout experiment, via
  `nfl.market_shrinkage.shrink_probability`. It transforms model-to-market
  differences in *log odds*. When alpha=0, the eligible reference becomes
  the paired same-book no-vig market; the raw model edge is removed.
- **Shrunk quote EV** is a *diagnostic estimate* using the frozen alpha and
  the actually observed book price. It is not guaranteed positive realized
  betting value or a tradeable market opportunity.

The previous NFL research reports already show significant reasons for caution:
`reports/regime_edge_reliability.json` currently marks **moneyline, spread,
and total** market-level regimes `UNRELIABLE`. The
`reports/market_edge_shrinkage.json` holdout currently selects **alpha=0**
and `MARKET_ONLY_PREFERRED` for those three markets; its conclusion is
`NO_INCREMENTAL_MODEL_VALUE`. The independent ESPN archive-verified
backtest shows **negative overall historical ROI**, but its provider-labeled
opening and closing observations do **not** provide independently timestamped,
contemporaneously executable wagers. This evidence *cannot* be used to
claim a newly verified betting edge. Recheck these files as they refresh.

## Evidence tiers

- `SUPPORTED_RESEARCH`: Only a clean and still-current sportsbook offer,
  positive raw and NFL-holdout-shrunk EV, independent market regime reliability,
  candidate regime reliability, positive shrinkage alpha, enabled policy
  market, model calibration and contextual certainty, and no contraindications.
  This is **not** production authorization even if it ever occurs.
- `RAW_PRICE_DISAGREEMENT`: Valid price with possible raw discrepancy but
  residual non-hard research questions. No stake authorization.
- `EVIDENCE_OR_CONTEXT_BLOCKED`: The quoted raw discrepancy failed a
  model-reliability, archive-proof, policy, shrinkage, market disagreement,
  contextual injury/QB, or related evidence gate.
- `NO_VERIFIED_PRICE`: The book, quote, market sanity, timestamp, or
  pre-kickoff freshness condition failed. This is not actionable.
- `NO_POSITIVE_RAW_EDGE`: The existing raw model signal was non-positive or
  incomplete.

Tiers sort a *research watch/audit board*, not an approved betting slip.
`edge_discovery_score` is a **capped raw discrepancy ordering aid**, not a
forecast of realized return or a learned betting threshold. Missing or
unreadable proof fails closed; no data is silently replaced with imagined odds,
implied probabilities, player news or validation samples.

## Auditable outputs

Every NFL operational model refresh now provides identical CSV/JSON copies
to `outputs/` and `docs/`:

- `edge_priority.csv`: all current selected game/market rows, sorted by
  observed evidence tier and capped raw disagreement, with exact blockers.
- `edge_supported.csv`: strict holdout-supported *research-only* rows.
  Expected to be empty until NFL-specific evidence improves.
- `edge_watchlist.csv`: residual raw research discrepancy candidates.
- `edge_exclusions.csv`: all excluded/unsupported and nonpositive rows,
  including market-policy and injury/context blockers.
- `edge_discovery_report.json`: category counts, data-source status,
  source provenance, and categorical evidence limitations.

The files also preserve the first-class model fields, book, market, side,
line, odds, quote timestamp, matched no-vig probability, and the independent
portfolio signal and approved stakes. The same evidence summary is embedded in
`outputs/current_model.json` metadata. Neither bet quantities nor
`portfolio_action` are modified by the research annotation.

## Next requirements to demonstrate genuine NFL advantage

Accumulate NFL-only chronologically split entry/CLV/forward evidence with
verified **same-book contemporaneous entry and comparable closing** prices,
then measure calibration, profitability with confidence intervals, subgroup
stability, market-only baseline lift, and player/context certainty.
A method that only selects the highest raw model EV after observing all
prices is not sufficient. The sportsbook market never enters the fair-score
training inputs, and the NFL release gate remains authoritative.
