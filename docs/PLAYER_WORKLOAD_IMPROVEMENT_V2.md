# Player workload improvement v2

The old coupled model allocated receptions using four-week target shares and
allocated carries using four-week carry shares. This could overreact to a small
recent sample and did not use a player's own longer receiving workload.

The new predictive family blends each player's original expected receptions or
carries 50/50 with their own prior eight observed game rows. It requires at least
three observations and no more than two weeks of staleness. Missing history
retains the baseline mean; a team with no usable history retains original weights.
The blended workload is normalized within each team, including any unallocated
mass, and supplied as routing weights to the unchanged joint simulator.

This preserves receptions/completions, rushing attempts, yards and touchdown
accounting in each simulated game. It does not paste unrelated means onto old
percentiles. The same score/drive draws still generate player production.
The receiving weight is a completion-routing probability, not a newly verified
target share or roster/active assertion. QB allocation is unchanged.

## Selection and diagnostic evidence

Rolling player estimates always use earlier weeks in the same season/team.
Candidate windows were 3/5/8 observed games and blends 0/.5/1. Selection used
2024 weeks 13–18; earlier games supply rolling training state. Each family's
tuning objective combines count error and yards error (yards scaled by 20 for
receiving and 10 for rushing). Zero blend was eligible. Both families selected
eight games and .5 blend.

The current model was then replayed with these weights for all 208 regular-season
games in 2025 weeks 5–18, at 2,000 draws and seed 20261013. All comparisons used
identical matched player/game/stat rows. Missing outcomes were not treated as zero.

| Metric | Original MAE | New MAE | Matched rows |
| --- | ---: | ---: | ---: |
| Receptions | 1.18279 | 1.13236 | 4,092 |
| Receiving yards | 15.23613 | 15.00758 | 4,092 |
| Carries | 1.24352 | 1.22409 | 4,092 |
| Rushing yards | 7.37970 | 7.23836 | 4,092 |

These all-position averages include small/zero workloads. Prior-role subgroup
results are also retained in the JSON; they should not be compared to differently
defined cohorts. Games and passing distributions were exactly unchanged in all
208 same-seed replays. Receiving/rushing TD errors worsened slightly; their
six-family-adjusted intervals span zero. We do not claim improved TD accuracy.

Game-cluster descriptive 99.167% intervals for the four main error differences
are below zero. These intervals account for six reported stat families, not the
entire exploratory research history. We inspected other candidates, including
QB recency and unconstrained individual means. The QB change was not adopted;
its improvement interval spans zero. The unconstrained mean approach was not
deployed because it can violate team totals.

2025 was previously inspected. Source publication vintages and active lineups
are unverified. This is a modest retrospective improvement, not a pristine
holdout, calibrated prop probability, executable market edge or profitability test.
See `reports/player_workload_shrinkage_diagnostic.json` for all results.

## Default use and lineage

Both root forecast commands and the main NFL MODEL workflow now generate the
baseline first and then the workload-v2 forecast. Published predictions and
contract comparisons use the same v2 scan/replay. Mandatory situational searches
remain active; unverified web text is not converted into numerical adjustments.
Other speculative experiments remain off by default.

Original six predictive fingerprint files and v1 forecasts/cohorts are preserved.
V2 fingerprints include the parent files, workload code, CLI and frozen settings.
New originals use `history/player_workload_v2_2026`; exact-contract selections use
`history/full_slate_workload_v2_2026`. Both receive automatic mature grading.
V1 continues to receive grading. The models' forecasts for the same game are
paired comparisons, not additional independent games; never pool their counts.
Prior original receipts are not replaced or relabeled.

The v2 replay contains complete revised roles and parent drives for exact replay,
with parent identity, source-history hash, new capture time and model fingerprint.
The current source-stat snapshot and adjustment audit are archived. Missing
player/team source identities are counted and excluded; absent player histories
fall back rather than becoming invented zeros. Any such source failure is visible.

No betting gate, qualification threshold or price/lineup requirement is relaxed.
All current recommendations remain research/PASS. Game-score accuracy, QB roles,
injuries/weather effects, final overtime and market superiority remain unfinished.
