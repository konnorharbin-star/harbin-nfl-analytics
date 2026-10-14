# Regulation and overtime score challenger

Every run of **NFL MODEL — Our Scores, Spreads, Totals and Player Props** now also
archives `reports/overtime_score_challenger.json`. The original published scores,
player forecasts, exact-contract rankings and immutable forward journals retain
their existing model lineage. This report is a separate score-only candidate.

Run locally with:

```bash
python -m scripts.nfl_overtime_research_run --season 2026
```

The challenger uses free nflverse football data. It verifies the fourth-quarter
terminal score against the final score for games without overtime; overtime games
must have tied regulation scores. Rows after quarter four never enter regulation
targets or the possession kernel. Completed games must be at least 72 hours old
at capture. Missing regulation endpoints, conflicting durations and unsupported
scoring sequences are counted in the source audit, rather than replaced by zeros.

It fits our offense/defense ridge model to **regulation** points with ridge 8 and
equal historical weights, then samples paired home/away fitted residuals. Integer
scores are rounded and clipped at zero. These residuals and their resulting tie
frequency have not been calibrated on held-out games. This differs from the
existing coupled drive/player simulation and cannot be substituted into its
player probabilities.

Only tied regulation draws receive overtime. The resolver implements ordinary
completed possessions under the [2026 official NFL Rulebook, Rule 16](https://operations.nfl.com/rules-officiating/2026-nfl-rulebook):
an opening score allows a response; an opening empty possession enables sudden
death; an opening defensive safety or defensive touchdown can end the game;
unnecessary winning-touchdown tries are omitted. The regular-season clock ends
after ten minutes, including an unfinished response. A modeled event later than
the remaining clock does not award a future score. An incomplete event stream
raises an error. Postseason rules are unsupported.

The kernel uses pooled **regulation** possessions and time of possession as an
uncalibrated duration proxy. It does not estimate overtime strategy, field
position, kickoff choice/recoveries, timeout management or clock administration.
Penalties, unusual kicking plays, defensive try returns and double changes of
possession require a richer event model. Source versions downloaded today are
not verified historical vintages. The report retains the sanitized score history
and possession kernel plus an input hash for replay; it is not an immutable
first-seen forward cohort.

`regulation` and `with_overtime` summarize paired draws, showing score, spread,
total, win and tie changes. The source snapshot contains no bookmaker fields.
`used_for_bet_ranking`, `player_props_updated`, `market_accuracy_proven` and
`promotion_earned` remain false. Source failures write an explicit blocked report;
the optional workflow step can fail without interrupting existing forecasts.

Promotion requires a separately frozen model lineage and chronological/forward
comparison of score errors and probability calibration against both the existing
model and contemporaneous markets. Rule handling alone establishes no pricing
edge. Existing release gates remain intact.
