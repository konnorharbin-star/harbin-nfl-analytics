# All-game moneyline experiment v1

The prior forward comparison is conditional on model-selected sides. This experiment
freezes every supported upcoming game with an independent home-win probability and a
complete named-book two-way moneyline, even when the game has no selected bet.
A game has one observation, not one per book or per complementary side.

## Frozen hypotheses

The reference is proportional de-vigging of a **same-book** home/away price pair.
Three candidates are declared before the new prospective cohort settles:

1. The existing independent football model probability.
2. 75% market / 25% independent model probability (explicit downstream pricing layer).
3. `sigmoid(1.1 * logit(market_probability))`, a fixed favorite/longshot-calibration hypothesis.

No coefficients or betting thresholds are fitted to this experiment. The prospective evaluation ends at March 1, 2027 UTC; later games require a new declared experiment. Results seen before that cutoff are descriptive and cannot be used to select or promote a winner. Book choice is
fixed: prefer DraftKings by name, otherwise alphabetically choose a named book. Never
choose a book by apparent EV, outcome or model agreement. Anonymous book IDs are excluded.
Collectors may observe advertised stale prices; collector time is **not** source quote
origin time. All receipts explicitly leave origin verification false. Only observed
pairs at most 120 minutes old and model publications at most six hours old are captured.
An unavailable/stale market blocks capture and appears in coverage counts.

## Prospective integrity and scoring

`history/market_first_v1/forecasts/` contains atomic immutable first-seen JSON records.
Repeated runs cannot overwrite a game, including after model or price changes. Source
publication time, model version, source-data hash, experiment code hash and original
paired price are preserved. Grades are separate immutable JSON records. A forecast
must have exactly one Git commit touching it, committed between capture and kickoff,
before grading can proceed. Edits, late publication, unresolved identity or non-final
scores block grading. ESPN final results must match both teams and kickoff; ties are
omitted from conditional binary probability scoring and are never silently settled.

The existing three-times-daily recommendation workflow captures and grades this separate
research cohort. NFL collection uses the existing free ESPN adapter and stored line
history; CFB uses existing published complete market pairs. Ten final-score requests
per run are allowed. Forecasts remain zero-stake research observations, not BET receipts,
and do not change recommendation counts or release gates.

Report Brier, log loss and descriptive ECE by model version and season. Paired confidence
intervals resample whole season/week clusters (10,000 draws, fixed seed). Bonferroni
intervals cover three candidates times two loss metrics. Intervals require at least
100 games and eight week clusters. Sparse ECE is withheld below 100 games. These are
fixed-horizon exploratory summaries through games kicking off before March 1, 2027 UTC: repeated dashboard inspection is not a sequential
promotion test. Any future promotion needs a separately frozen evaluation window and
all existing economic, quote-provenance and release requirements.

## Historical diagnostic

`reports/market_first_archive_diagnostic.json` reports fixed candidates by chronological
season on already-inspected historical data. It is **not an untouched holdout** or proof
of executable returns. NFL input is the existing full moneyline archive cohort, with
selected away sides restored to home orientation; proper binary losses are symmetric.
CFB pairs are reconstructed through the existing canonical archive matcher and joined
to chronological model predictions. Missing paired prices or probabilities are excluded
and counted. Opening-field/final-fallback archive quotes remain unverified for execution.
No ROI, EV or CLV claim is made from these unverified prices.

Reproduce NFL: `python -m scripts.market_first_experiment --sport nfl --historical reports/free_market_bets.csv --out reports/market_first_archive_diagnostic.json`

Reproduce CFB: `python -m scripts.build_cfb_moneyline_diagnostic --out /tmp/cfb-market-first.csv`, then `python -m scripts.market_first_experiment --sport cfb --historical /tmp/cfb-market-first.csv --out reports/market_first_archive_diagnostic.json`.
The source download is cached only locally; no paid API or wagering integration is used.

The fixed calibration hypothesis has inconsistent historical gains. The independent
model and blend generally fail to improve on the market. No credible betting advantage
has been established, and no production candidate is enabled by this experiment.
