# Video method intake — October 14, 2026

Source: [Mayo Media Network / Rob Pizzola](https://www.youtube.com/watch?v=LzcTB0lOF74).
The public player metadata, publisher description and chapter outline were readable.
The public caption endpoint returned empty responses; the spoken transcript was
not available. No claim here attributes an exact formula or coefficient to the speakers.

The outline identifies free football data (14:00), market comparison (19:20),
distinct prop modeling (22:45), leakage control (26:15), variance (45:10), and
efficiency statistics (50:15–54:45). These are research directions, not evidence
that a particular betting strategy is profitable.

## Incorporated experiment

`nfl.clean_efficiency_research` tests a separate football-only residual correction
of the independent fair margin and total. Existing EPA/recent-form features remain
available; this experiment adds explicit play cleaning and neutral-state filtering:

- Regulation pass/run plays with a valid down; exclude kneels, spikes, dead-ball
  plays and two-point conversions. Include sacks and scrambles in dropbacks.
- Keep turnover plays. Do not retrospectively delete costly turnovers to manufacture
  apparent efficiency. No claim that this fully removes luck.
- Use absolute pre-play score difference at most 14 and more than 120 seconds left
  in the half; avoid spread-adjusted win-probability features.
- Aggregate passing EPA, designed-run EPA and success (`EPA > 0`) on offense and
  defense. Missing inputs remain excluded and counted. Require at least four prior
  games with observed plays per metric and at least 40 plays.
- Use the last eight completed, 72-hour-mature games; shrink each metric toward
  the prior-only league mean by 100 plays. Defense stores efficiency allowed,
  so larger defensive EPA is weaker, not stronger.

Those filters, sample minima and shrinkage settings are our declared research
specification, not the video's unpublished model. Definitions were checked against
[the primary nflfastR guide](https://nflfastr.com/articles/beginners_guide.html).
No paid PFF, FTN, SharpStack, API account or wagering integration was added.

Fit coefficients on 2023 weeks 5–18 (208 games). Select ridge 10/100 and blend
0/.25/.5/1 on 2024 weeks 5–18 (208 games), then refit the selected structure on
2023–24. Zero adjustment is always eligible. Diagnose 2025 weeks 5–18 (208 games)
without changing that selection. This season was already inspected elsewhere in
the project; it is **not pristine out-of-sample evidence**. Contemporary nflverse
EPA/source revisions also lack verified historical publication vintages.

| Target | Selected blend | Baseline MAE | Challenger MAE | MAE change |
| --- | ---: | ---: | ---: | ---: |
| Home margin | 1.0, ridge 100 | 10.6115 | 10.6222 | +0.0107 |
| Total | 0.0 | 10.4862 | 10.4862 | 0.0000 |

Margin's two-target-adjusted 97.5% game bootstrap interval is
[-0.2308, +0.2535] points; week-cluster interval is [-0.2307, +0.2284].
Both span zero. Margin RMSE also worsened (13.2031 to 13.2701).
Totals retained the baseline because every positive blend lost on tuning.
No probabilities are calibrated by this score experiment; Brier/log-loss/ECE,
actual ROI and CLV remain unavailable. It did not establish market superiority.

## Every main GitHub model run

Update: experimental monitoring is now disabled by default. Enable the manual
`include_experimental_diagnostics` option to run it; see
[default model simplification](DEFAULT_MODEL_SIMPLIFICATION.md).

The NFL MODEL workflow now emits `reports/clean_efficiency_forward.json` with
prior-game IDs, sample counts, baseline/challenger margin and total, exclusion
counts, capture time and frozen-model hash. A compressed football-only source
snapshot is archived with its uncompressed SHA-256 for replay. Failures remain
visible in the optional research step. Forecasts use the fixed 2023–24 coefficients;
there is no fit or parameter search on 2026 outcomes.

The existing coupled model, props, immutable first-seen cohorts and betting
ranking are unchanged. This challenger has zero betting weight. All candidates
remain research/PASS. No forced per-game bets, profitable-edge claim, or staking.

Reproduce:

```sh
python -m scripts.nfl_clean_efficiency_run study --out reports/clean_efficiency_video_study.json
python -m scripts.nfl_clean_efficiency_run forecast --out reports/clean_efficiency_forward.json
```

Frozen parameters: `research/clean_efficiency_frozen.json`.
Diagnostic evidence: `reports/clean_efficiency_video_study.json`.
Further improvements require separately declared candidates and future evidence;
changing filters after looking at 2025 would make additional results exploratory.
