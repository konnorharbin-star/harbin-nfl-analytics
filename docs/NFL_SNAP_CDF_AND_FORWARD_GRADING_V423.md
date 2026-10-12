# NFL Snap-Conditioned Prop CDF and Prospective Forward Grader — v4.23.0

**2026-10-12 UTC. Research-only; no demonstrated positive-EV BetMGM betting model.** Active markets NFL two-way moneyline and player props; spreads/totals paused.

## A. New actual player-prop over/under probability test

Implemented scripts/nfl_prop_snap_conditional_cdf.py, tests/test_nfl_prop_snap_conditional_cdf.py and .github/workflows/edge-lab-prop-snap-conditioned-cdf.yml. Real official-source cloud test SUCCESS https://github.com/konnorharbin-star/harbin-nfl-analytics/actions/runs/38246117941 (12 guards passed), familywise retest SUCCESS https://github.com/konnorharbin-star/harbin-nfl-analytics/actions/runs/38246979440 .

PREDECLARED construction: match unique GSIS↔PFR IDs, previous completed NFL-game offensive snaps and truly earlier player-stat point projections; use original 2024 official-stat-row-conditional standardized residuals. Split on original prior PFR offensive snap mean LOW below 35, HIGH at least 35, with 80 minimum position-specific residuals or 160 pooled market minimum; insufficient cells fall back unchanged to original CDF. Fit only 2024, use **same exact 2025 game/player/stat and seven synthetic half-point thresholds** for original-CDF and snap-bin CDF. Original full 2024 CDF SHA256 2f6040e722592c7771d8a75d8022926dbff32b667626a87cf771093664cf8978; immutable new 2024 bucket residual catalog SHA256 **4e7e6458dd926a178abb5db34e7a279563a9177be7fcfcd51f0cbe80fe22a2c6** in history/player_prop_prior_snap_cdf_2024_v1.json. 2024 matching train sample 8,750 player-game-stat rows; 2025 same-cohort diagnostic 8,836 player-game-stat rows, not independent games or authentic BetMGM props.

| 2025 market | player/stat rows | Existing Brier | Snap CDF Brier | Delta snap-minus-existing | 95% game-cluster bootstrap CI | Four-market Bonferroni 98.75% CI |
|---|---:|---:|---:|---:|---|---|
| Receptions | 3324 | 0.10949594 | 0.10903462 | -0.00046132 | [-0.00087077,-0.00003843] | **[-0.00098316,+0.00005069]** |
| Receiving yards | 3290 | 0.09136500 | 0.09095823 | -0.00040677 | [-0.00085506,+0.00008818] | [-0.00096193,+0.00018780] |
| Rushing yards | 1818 | 0.08095280 | 0.08077074 | -0.00018205 | [-0.00078937,+0.00047795] | [-0.00098659,+0.00065014] |
| Passing yards | 404 | 0.20243764 | 0.20318066 | +0.00074302 | [-0.00001512,+0.00157088] | [-0.00029502,+0.00178171] |

Intervals game-cluster bootstrap 400 replicates; thresholds within each game/player correlated. Unadjusted receptions 95% interval excluded zero, but **the four-market familywise-adjusted interval includes zero for ALL four markets**. 2025 already inspected and original PFR historical publication time not independently verified. Passing-yard performance worsened. This is not proof of improvement against BetMGM and is insufficient to promote a new production CDF. Preserve original 2026 forecasts unchanged; new independent research sidecar in history/player_prop_snap_cdf_sidecar_v1_2026.jsonl first frozen **2026-10-12T12:05:38.326332Z**, **698 distinct game/player/market frozen sufficient-statistic groups**; 2 group identities lacked qualifying historical snap coverage. Those are NOT 698 executable odds or tickets.

## B. New 72-hour delayed prospective source grading

Implemented scripts/nfl_prop_stat_row_forward_grader.py, tests/test_nfl_prop_stat_row_forward_grader.py, workflow .github/workflows/edge-lab-prop-stat-row-forward-grade.yml. Cloud SUCCESS https://github.com/konnorharbin-star/harbin-nfl-analytics/actions/runs/38246604672 with 10 passing tests. Reads **304 original 2026 pregame frozen** player-game official-stat-row presence proxy forecasts and immutable 2024 model beta, NEVER recalibrates or edits originals. After kickoff waits >=72 hours and requires completed NFL game, matching original game kickoff, >=4 distinct official stat player rows from each team and >=12 combined before labeling a recorded player-row as present/absent. Absent official stat row is NOT verified DNP, under, player inactive or BetMGM void. Grades preserve original version, source collection timestamp and correction history; true source-stat corrections append new labeled record without rewriting history. First 2026-10-12 actual cloud result: **304 games/player proxies blocked GAME_NOT_COMPLETED; 0 valid outcomes, 0 wagers**. Forward calibration only when mature.

## C. Original source outage transparently resolved

Initial official nflverse Release downloads returned transient 504 Gateway Timeouts on various schedules, snap counts, player stats and GSIS-PFR ID assets. Retrospective estimates were NEVER calculated from those unsuccessful runs. Added finite retries per source/season and hard failure if official data remain inaccessible. Original-source rerun completed successfully. Reference for update cadence and distinction between regular-season PFR snaps and delayed FTN detailed participation: https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html .

## D. Release decision

Version 4.23.0 documents reproducible new research, four-market statistical discipline and a genuine future-score grader, **not** profitable model release. The original player-prop conditional CDF remains official research benchmark; unvalidated snap-conditioned CDF remains sidecar; previously rejected Elo/QB moneyline signals remain rejected. Exact account-accepted U.S. BetMGM sportsbook prices/origin time, player activation/DNP/void and actual settled prop results remain absent. No calculation of tradable true EV, no account use or bet placement. **Profitable edge registry EMPTY; decision PASS.**
