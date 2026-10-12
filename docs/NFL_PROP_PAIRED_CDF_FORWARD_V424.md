# BetMGM Edge Lab v4.24.0 — Genuine Prospective Paired Player-Prop CDF Line Forecasts

**Study date:** 2026-10-12 UTC. **Status:** RESEARCH ONLY / PASS; no confirmed profitable U.S. BetMGM model.

## 1. What was actually implemented and verified

- Source: [scripts/nfl_prop_paired_cdf_forward_grader.py](../scripts/nfl_prop_paired_cdf_forward_grader.py).
- Tests: [tests/test_nfl_prop_paired_cdf_forward_grader.py](../tests/test_nfl_prop_paired_cdf_forward_grader.py), **14 tests passed**.
- Automated cloud workflow: [.github/workflows/edge-lab-prop-paired-cdf-forward-grade.yml](../.github/workflows/edge-lab-prop-paired-cdf-forward-grade.yml). Initial full run https://github.com/konnorharbin-star/harbin-nfl-analytics/actions/runs/38309944526 **SUCCESS**. Hardening pass https://github.com/konnorharbin-star/harbin-nfl-analytics/actions/runs/38310060453 **SUCCESS**.
- Immutable original matched hypothetical half-point forecasts: [history/player_prop_paired_cdf_lines_v1_2026.jsonl](../history/player_prop_paired_cdf_lines_v1_2026.jsonl). **4,886 first-seen line records**, created **2026-10-12T21:20:25.336496Z**, covering exactly **698 original player-game-stat groups × 7 predefined synthetic thresholds**; reference full empirical distribution has 700 source groups, two missing eligible snap source groups excluded. All were before scheduled kickoff. Original existing game/player/stat mean/scale, 2024 fixed residuals and prior-snap categorical source stayed unchanged. The grade journal will be [history/player_prop_paired_cdf_line_grades_v1_2026.jsonl](../history/player_prop_paired_cdf_line_grades_v1_2026.jsonl) only once legitimate postgame sources mature.

Each first-seen record contains **both** frozen `P(over | official player stat row recorded)` for (i) original 2024 residual CDF and (ii) experimental 2024 prior-snap conditioned CDF **at identical full-game player/market/selection/half-point line**. For both systems, only existing 2024 calibrated residual catalogs are queried; no 2026 final stats are used in initial freeze. Original SHA256 of each source and derived record is checked before grading. The original paired probability and its SHA are **never overwritten** on later reruns or source correction.

This is a fixed grid of **synthetic lines**, **not** BetMGM U.S. account-accepted lines or price observations. The line count is heavily dependent across thresholds per player and games. No individual half-point research line constitutes a real bettor-executable contract or an approved profitable bet.

## 2. Genuine prospective grading plan, deployed as scheduled CI

Official NFL game/player stat rows are loaded ONLY after separately committing the originally frozen paired probabilities in GitHub. An **original source pregame freeze-only stage commits first**, then another step fetches the outcome source for grading. This prevents a transient GitHub/nflverse HTTP 504, which was previously observed during source research, from making originally frozen probabilities disappear when postgame data downloads fail.

Grade eligibility requires:
- Exact game ID, original GSIS player ID, explicit full-game stat field and same initial half-point threshold.
- Original first-seen model time earlier than the real target kickoff; original frozen content hash verified.
- NFL schedule indicates completed game; a source provider observation occurs **at least 72 hours after kickoff**.
- Original kickoff equals source schedule kickoff; material moved kickoffs are blocked pending manual as-of review.
- At least **4 distinct player stat rows for each team** and **12 total** confirm minimum source coverage.
- Only an **explicit official player stat row with a finite stat value and matching team** can be labeled over/under against an original half-point threshold. A missing player stat row is excluded, NOT automatically **0**, UNDER, DNP, sportsbook void, or settled bet.
- First eligible grade append-only. Source outcome corrections append a separate revision with original source SHA unchanged.
- Brier score difference candidate-minus-original calculated on exact same eligible rows and threshold; bootstrap resampling **by game**, 400 draws, with both 95% and **98.75% four-market Bonferroni intervals** once there are at least 20 unique complete graded NFL games for that specific stat market.
- **Zero** first official grades as of 2026-10-12T21:21Z: all **4,886** lines were blocked as `GAME_NOT_COMPLETED`. This is expected and confirms no hindsight labels were invented.

CI execution retains scheduled original first-seen capture, daily source rechecks and 2026-only scope; report path `reports/player_prop_paired_cdf_forward_2026.json`.

## 3. Original historic model results unchanged

Prior retrospective 2025 diagnostic from v4.23.0: nominal improvement on receptions, receiving/rushing yards; passing yards became worse. **None of the four signals survived predeclared familywise four-market 98.75% intervals**. 2025 previously inspected. Do not select receptions or any subgroup for approval on these retrospective results. A forward confidence interval or Brier improvement alone still is not evidence of positive *market-priced* BetMGM EV.

Current active market research scope NFL two-way moneylines + player props; spreads/totals paused. Existing Elo/QB winners models remain unprofitable/rejected as market improvement. Never claim that a model can guarantee a mispricing. No paid data APIs or automated wagers.

## 4. Release decision

Version **v4.24.0** is an **evidence pipeline expansion**, not a change to the approved probability model. Original 2024 CDF, experimental 2024 snap-conditional CDF and all earlier original 2026 journals retain their historical first-seen state. The next externally verifiable milestone is independently grading the above original paired **conditional** forecasts **after** a source-complete game actually finishes.

Unverified or absent: prospectively confirmed player active/void, actual BetMGM U.S. account-accepted same-contract price and quote-original timestamp, identical game/participant/period offering, no-vig market-only prop probability reference, financially meaningful uncertainty-adjusted EV, and independent forward paper trading.

**Profitable BetMGM strategy registry EMPTY. Decisions PASS. No wagers.**
