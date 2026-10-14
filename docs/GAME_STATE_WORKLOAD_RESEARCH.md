# Quarter/state workload challenger: completed experiment

The public description of Rufus Peabody's NFL prop process starts with team opportunity volume, game-state-dependent run/pass mix and player workload shares. Source: https://www.unabated.com/post/pie-for-dinner-rufus-peabodys-nfl-player-prop-process (accessed October 14, 2026). No proprietary formula or published profitable NFL code was obtained. This is our own statistical implementation of the disclosed idea, using our independent fair margin rather than sportsbook spreads.

The challenger estimates leading/tied/trailing occupancy separately for each regulation quarter. Multinomial logistic coefficients use past-only independent fair margins, home/away and quarter interactions. Expected state mix weights each team's prior six-game conditional dropback rates, shrunk toward that quarter's overall prior rate. Expected opportunity volume is the prior mean by quarter. At least four prior team games are required. A zero-blend baseline remains eligible.

Fit: 2024 weeks 5–12 (229 team-games). Tune: weeks 13–18 (186). Select shrinkage from 32/128 opportunities and blend from 0/.5/1 on tuning Brier; selected 32 and 1. Refit occupancy on 2024 only, freeze settings and evaluate 2025 weeks 5–18: 414 team-games across 208 distinct games. Each target's fair score and workload history excludes target/future weeks and source games younger than 72 hours. Source publication vintages are not verified. Previously examined 2025 is a diagnostic, not pristine confirmation.

| Metric | Prior-quarter baseline | State challenger |
|---|---:|---:|
| Dropback probability Brier | 0.2409577931 | 0.2408913546 |
| Log loss | 0.6750506057 | 0.6749067941 |
| Expected dropback-count MAE | 6.6639694 | 6.6936540 |

Brier delta is -0.0000664385; paired game-cluster 95% interval spans zero (exact interval in the JSON report). The small probability gain is inconclusive; workload-count error worsened. **Promotion not earned.** These are regulation dropbacks, including sacks/scrambles, not actual passing-attempt prop probabilities. No sportsbook-price comparison, ROI, CLV or profitable edge follows from these results. Official participant/void and full-game OT gaps remain.

Files: `nfl/game_state_workload.py`, `scripts/nfl_game_state_workload_run.py`, `research/game_state_workload_2024_frozen.json`, `reports/game_state_workload_diagnostic_20261014.json`. Numeric coefficients are plain JSON, not binary estimator artifacts. The report contains source hash, exclusions, rejected examples and all tuning candidates. Scrimmage play duplicates and missing game-state source are rejected. Unsupported quarters, kneels and spikes are excluded explicitly.

Scheduled coupled runs generate a separate workload monitoring sidecar from the frozen 2024 coefficients and current prior-week NFL data. The full-slate JSON joins this sidecar without using it for prices, betting ranks, qualification or original frozen predictions. A sidecar failure is visible as unavailable metadata and in its run step, while existing model collection can continue. This is monitoring, not a new graded or promoted prop model. Old coupled and selected-contract cohort fingerprints and predictions remain unchanged.

Reproduce the historical diagnostic:

```bash
python -m scripts.nfl_game_state_workload_run study
```

Monitor upcoming workload:

```bash
python -m scripts.nfl_game_state_workload_run forecast --out reports/game_state_workload_forward.json
```

Operational full-slate research is deployed. A complete independently calibrated, profitable betting recommendation model is still not established. Inconclusive challenger evidence is retained rather than advertised as an edge.
