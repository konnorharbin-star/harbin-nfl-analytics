# BetMGM Edge Lab — v4.22.0 Historical Snap-Count Stat-Row Proxy

**2026-10-12 UTC. Research-only evidence/model sidecar, not a BetMGM betting system.**

## What was built

- [scripts/nfl_prop_snap_stat_row_diagnostic.py](../scripts/nfl_prop_snap_stat_row_diagnostic.py): pregame forecasted presence of an NFL player-stat row based on *strictly earlier completed regular-season game* PFR offensive snaps. The forecast population consists of players identified by the existing strictly prior-week player-stat eligibility rules, deduplicated to one game/player rather than counting each prop market as an independent observation.
- [tests/test_nfl_prop_snap_stat_row_diagnostic.py](../tests/test_nfl_prop_snap_stat_row_diagnostic.py): blocks same-week snap leakage; differentiates absence of an official stat row from confirmed DNP and void; checks GSIS/PFR unique crosswalk and nonzero-imputation; guarantees first-seen 2026 forecast identity.
- [.github/workflows/edge-lab-prop-snap-stat-row-proxy.yml](../.github/workflows/edge-lab-prop-snap-stat-row-proxy.yml): autonomous 2026-scheduled first-seen forward observations, prior development, historic diagnostic and source consistency checks.
- [history/player_prop_stat_row_proxy_v1_2026.jsonl](../history/player_prop_stat_row_proxy_v1_2026.jsonl): first **304** 2026 player-game original pregame source-row-observation proxy probabilities as of **2026-10-12T11:43:15.149334Z**; horizon 10 days; **not** implied active/eligible status or valid prop settlement/EV.
- [history/player_prop_stat_row_training_2024_v1.json](../history/player_prop_stat_row_training_2024_v1.json): **pinned original 2024 training coefficients**, model/loss specification, origin cloud job and SHA256 **6fed24b417f9f6beb116c5323692b32a08ce97f9be35c84116a876a18df1a66c**. Subsequent runs recompute 2024 fit and fail closed on any revision instead of silently updating 2026 model. Original 2026 cohort retained if developer training ever needs a new version.

This is a **new independent auxiliary model**, not a revision of existing player-stat value distribution, old v1/v2 threshold, first-seen probability journals, or paper/wager ledger.

## Training and diagnostics

Prediction target is binary **a matching player exists in official nflverse game-stat rows**. Target 0 is **MISSING official stats row**, which does **NOT** independently establish DNP, injury, inactivation, void, or a final paid over/under selection. Statistical target is different from BetMGM eligibility.

2024-only development trained fixed 5-feature ridge-logistic regression on **4,632** matched unique player-games after requiring at least two observed prior-game snaps, with unique GSIS/PFR identity. Model features: intercept, mean previous four observed offense snaps / 60, last observed snaps / 60, number observed / 4, first-to-last observed trend / 60. All source snap data restricted to *strictly prior completed team-game weeks*; no next-game PFR snaps. Player eligibility/previous stat rows from prior weeks only. Ridge penalty 50, fixed before 2025 diagnostic; 2025 outcomes did not refit full model.

2025 retrospective diagnostic **4,657 unique player-games in 224 NFL games** from the same eligibility rule:

| Fixed 2024 model | 2025 Brier (lower better) |
|---|---:|
| Historical 2024 unconditional stat-row rate | 0.14104667 |
| Count-only model using number of prior snap rows, without workload magnitude | 0.13940551 |
| Full prior-game snap workload model | **0.13245844** |

Full model 2025 log loss **0.42693174** vs historical rate **0.45580575**. Its Brier improvement vs simple constant baseline **-0.00858823**, game-cluster resampled 95% bootstrap interval **[-0.01032469, -0.00695605]** (400 resamples). Incremental improvement over the 2024-fitted **count-only** prior snap history model **-0.00694707**, 2025 game-cluster 95% bootstrap interval **[-0.00861090, -0.00522519]**.

This is credible *retrospective evidence about an official-stat-row recording task*. However, 2025 has already been inspected in previous research. Its positive conditional source-row signal is NOT a fresh untouched betting test and cannot be called a winning-player-prop edge. PFR historical original row publication timestamp metadata is not independently verified, meaning these are reconstructed prior-week feature rules, not independently proven as-of offer-time records. Historical cohort condition requiring two prior PFR player snaps may select a high-availability subset; it does not generalize automatically to unobserved/injured/rookie/traded populations or prop listing universe.

Cloud evidence:
- 10 original leakage/identity tests and actual 2025/2026 source study: https://github.com/konnorharbin-star/harbin-nfl-analytics/actions/runs/38244401097
- Stronger count-only baseline reproducible cloud evaluation: https://github.com/konnorharbin-star/harbin-nfl-analytics/actions/runs/38244630239
- First original 2024 beta pinned and checked: https://github.com/konnorharbin-star/harbin-nfl-analytics/actions/runs/38244788300
- Model SHA full-beta original first-seen: 6fed24b417f9f6beb116c5323692b32a08ce97f9be35c84116a876a18df1a66c.

## What is NOT established

1. This model does not establish that a player will be ACTIVE on kickoff or play the required snaps to settle at BetMGM.
2. Missing official stats row is a data presence problem and cannot simply be treated as **zero stats** or as a sportsbook VOID ticket.
3. No independently established original U.S. BetMGM account-accepted price/contract timestamp; no real/true estimated EV.
4. We have not proven that multiplying official-stat-row presence and player-stat conditional CDF yields valid OVER/UNDER bet settlement probabilities. That construction requires independent resolution of missing rows and void/DNP eligibility.
5. 2025 outcome comparisons are retrospective, source-vintage uncertain and subject to prior inspection. Actual independent prospective 2026 calibration and user-available BetMGM paper trades have not occurred.
6. Forward distributions, legacy probability journals, spread/total rejected-hypothesis records, and historical failures are preserved.

## Research gates and decision

**Promote this as an auxiliary source-row-coverage research model ONLY; leave all BetMGM actual moneyline and prop trading models blocked.** Record 2026 original player-game first-seen predictions and later grade actual recorded row status (with explicit source completeness verification). For a true prop-edge hypothesis, source separate true participant/playing time and exact book offered terms, then compare prospectively to same-contract no-vig market-only benchmark with pre-registered intervals and paper trades. User account never used, nothing wagered.

Profitable-model registry EMPTY. Betting recommendation PASS. No paid subscriptions or automatic betting.
