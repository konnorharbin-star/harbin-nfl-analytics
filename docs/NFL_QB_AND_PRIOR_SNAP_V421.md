# BetMGM Edge Lab — v4.21.0 NFL Moneyline QB Challenger + PFR Snap Source

**As of 2026-10-12 11:32 UTC. RESEARCH ONLY.** Scope NFL two-way full-game MONEYLINES and NFL player props only. Spreads/totals and NCAA active betting-discovery deferred. No validated positive-EV BetMGM offers or approved wagering models.

## 1. Pregame QB and recent-form moneyline model: independent test failed

Sources: scripts/nfl_moneyline_qb_efficiency_challenger.py, tests/test_nfl_moneyline_qb_efficiency_challenger.py, .github/workflows/edge-lab-qb-moneyline-challenger.yml.
Cloud research run with 13 tests: https://github.com/konnorharbin-star/harbin-nfl-analytics/actions/runs/38242551708

Available NFL player weekly stats have total passing EPA, attempts, sacks, season, week, GSIS player ID; official source dictionary: https://nflreadr.nflverse.com/articles/dictionary_player_stats.html .

Fixed pre-week features: mean previous six team scoring margins difference, observed team QB passing EPA per dropback from up to five previous completed weeks, identity continuity proxy from prior observed QBs, difference in published rest, neutral/home venue. The QB is *last observed*, not independently verified as the next-game starter. The source feature builder predicts all games for a week before applying any outcome/QB data for that week; incomplete week results cannot update past history. Independent sportsbook-free logistic coefficients and separate archival-market-offset logistic coefficients fitted ONLY on 2019–2022, 1,050 matching non-tie games, fixed ridge alpha 75. Historic 2023–2025 analyzed as retrospective diagnostics; all previously inspected and not pristine forward proof.

| Season | Games | Archived market Brier | Market + preweek QB/form Brier | Standalone QB/form Brier |
|---|---:|---:|---:|---:|
| 2023 | 272 | 0.21870951 | 0.22119237 | 0.24182012 |
| 2024 | 272 | 0.20016162 | 0.20019939 | 0.22973767 |
| 2025 | 271 | 0.21213897 | 0.21297063 | 0.22658303 |

**Both challenger types perform worse than archived market-only probabilities in every historical diagnostic year. Thus QB EPA/recent margin challenger fails approval.** It is nonetheless maintained as a forward **RESEARCH ONLY** comparator and never considered BetMGM value without independent original odds and future validation. Archived historical market price-origin UTC cannot be established.

### Material first-seen horizon correction: two separate immutable journals

Initial v1 code saved **194** original 2026 forecasts across the *entire remaining regular-season schedule* on 2026-10-12 11:26 UTC, beyond a useful weekly recency horizon. Although they were frozen before scheduled kickoff, future many-week priors did not include unplayed intervening games. **Classify that source as a long-horizon exploratory cohort, NOT fresh preweek NFL predictions and NEVER eligible to demonstrate model performance as near-kickoff operational bets.**

Rather than modifying those original records, code now explicitly registers separate **V2_10_DAY_WINDOW** forecast lineage and freezes only kickoffs within the next ten days. New v2 journal: history/moneyline_weekly_qb_challenger_v2_2026.jsonl, original capture UTC **2026-10-12T11:31:49.088444Z**, **15** pregame future games (2026_05_BUF_LA and 14 Week6). New scheduled CI **SUCCESS** with 14 tests: https://github.com/konnorharbin-star/harbin-nfl-analytics/actions/runs/38243173588 .
Original v1 all-season file history/moneyline_weekly_qb_challenger_2026.jsonl retained without deletion or alteration. Neither v1 nor v2 uses accepted BetMGM prices or justifies a bet.

## 2. Real PFR snap counts: new historical workload coverage for future props

Sources: scripts/nfl_pfr_prior_snap_coverage.py, tests/test_nfl_pfr_prior_snap_coverage.py, .github/workflows/edge-lab-prior-pfr-snap-coverage.yml. Verified real-data cloud run **SUCCESS, 12 unit tests**: https://github.com/konnorharbin-star/harbin-nfl-analytics/actions/runs/38243224046 .

Source-of-truth docs:
- https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html : PFR snap-count data updates at 00,06,12,18 UTC during regular season subject to PFR availability, unlike delayed FTN detailed participation.
- https://nflreadr.nflverse.com/articles/dictionary_snap_counts.html : game_id, PFR player ID, team, offense_snap count.
- https://nflreadr.nflverse.com/reference/load_players.html : canonical nflverse GSIS ID plus PFR ID crosswalk, requiring unique reverse/forward association.

First source capture **2026-10-12 11:32 UTC**: **7,277** 2026 PFR snap source rows, matching **700 original player/game/market snapshot groups** across upcoming 15 NFL games. Of these, **698** groups have at least two prior-game observed snaps with unique GSIS-to-PFR identity and source team coverage, **2** missing unique player ID mapping. These are **forecast groups, not independent players, sportsbook prices or wagers**. New append-only original journal: history/player_prop_prior_snap_coverage_2026.jsonl (700 new records). Missing players from covered prior games are NOT assumed inactive and not imputed to zero. All source matching occurs using prior completed team games STRICTLY BEFORE target game week, with 4 most recent team games screened. Future game's participation, starting status and void/settlement condition remain unknown; no unconditional P(win/push/loss) or BetMGM price EV.

Source timestamps capture when model collected this copy, NOT original PFR individual row release time. As-of source record can be revised later. No leakage from target-week snaps. Existing v1/v2/full CDF player probability originals are never recalibrated or overwritten. Crosswalk identity is not an independently authenticated sportsbook listing; U.S. BetMGM exact contract participant still needs separate verification.

## Model registry and blockers

- Archived market baseline still better than all tested Elo, QB EPA and historical margin challengers, so NONE approved.
- New historically observed snap workloads can inform the next strictly pre-registered prospectively validated player opportunity/participation model, but this version is NOT itself that model.
- For actual bet approvals, require time-origin authenticated and account-offered U.S. BetMGM exact prices, precise same contract, pre-kickoff independent model evidence, DNP/void and settlement, market-only calibration, uncertainty, and prospective paper ROI/CLV.
- No automatic wagering. No paid APIs. Zero true EV projections from accepted account prices. Decision PASS.

**Release v4.21.0 is a source-coverage, chronological horizon integrity and research-negative-results release, not a profitable model upgrade.**
