# Unified NFL Game Score + Player Prop Simulation — Research Design

STATUS: RESEARCH DESIGN ONLY. Not a fitted model, a measured accuracy improvement, an approved bet, or a guaranteed edge. Registered first-seen source cohorts and rejected hypotheses remain immutable.

## Objective
Build a football-first, bookmaker-neutral generative forecast with game total, team scores, winner, margin, player passing/rushing/receiving volume, touchdowns and exact-stat outcome probabilities all deriving from ONE coherent simulated game. Compare to current canonical models before any promotion. Compare sportsbook market prices only AFTER freezing the pregame football forecasts. Do not replace canonical forecasts merely because a new model sounds sophisticated.

## Audit findings and why
Existing canonical score is a ridge-regularized team offense/defense scoring model (nfl/ratings.py); additional candidates exist for possessions x points/drive (nfl/factorized_score.py), direct margin/total (nfl/decoupled_score.py), QB/roster, empirical player CDFs (nfl/prop_probability.py; scripts/nfl_player_prop_full_distribution.py) and time-maturity audits.
Existing reports/qb_total_forward.json at an early forward snapshot: 17 scored games, canonical total MAE 10.78245 points, frozen QB total challenger MAE 10.87418 points (worse 0.09173). The sample is not a basis for promotion.
Existing 2024–2025 opponent-adjusted PBP candidate results were not consistently improved across validation/holdout folds (reports/candidate_benchmark.json). More factors alone cannot fix accuracy.
The 698 frozen 2026 single primary prop anchors and 4,886 correlated synthetic threshold lines from 15 original games are NOT 698 independent samples or executable sportsbook predictions. Preserve original hashes/first-seen snapshots.

## Stage 0: As-of input reliability — highest priority
Collect a reproducible UTC as-of feature snapshot BEFORE a game: schedule/kickoff and venue, roof/outdoors, QB starter status and confidence, offense and defense, offensive line and defensive front, skill-player availability, scheme/personnel, rest/travel, coaching changes, official injury and practice reports, NWS-issued forecasts for outdoor stadium wind/rain/temp/roof, time-available NFL play-by-play and player boxscore inputs. Store original source URI, original issue time if published, capture time, immutable data hash, missingness and player/team ID crosswalk.
Free quantitative sources: nflverse public data releases, nflreadpy, NFL official schedules and injury reports; weather.gov NWS API. Do not use observed future/postgame weather or final stat corrections as pregame features. Separate direct play-level participation (2023-onward nflverse participation may be postseason-released only) from in-season public snap counts. Never invent route share, recent injuries, live starters or objective coefficients. Missing inputs trigger explicit scenarios or coverage blockers.

## Stage 1: Shared drives and team scoring process
Model latent tempo and correlated number of possessions using past neutral-situation seconds/play, plays/drive, neutral pass tendency, offense/defense matchup strength, late-game score effects and sensible shrinkage.
Generate each drive's number of plays, field position, drive efficiency, turnover/field-goal/TD/no-score result and relevant scoring details; draw home/away with common game environment and game-state-dependent strategy. Score must be nonnegative and integer-valued. Include defensive/ST scores and overtime rules where modeled/validated.
Compute total = home points + away points, home margin = home minus away on EACH simulation draw; never overwrite total with unrelated point estimate. Produce calibrated full distribution, not merely a mean. Existing score and factorized model remain baseline/shadow alternatives.

## Stage 2: Shared play-volume and game-script generator
Derive team pass attempts, sacks, scramble and designed rush opportunities from simulated possessions and situation-adjusted pass rate. QB injury uncertainty and lineup updates generate scenarios rather than unvalidated subjective point modifiers.
Keep score and volume correlated: trailing team may pass more only if demonstrated in chronologically evaluated data; simulated score state is allowed, actual future game result is not.

## Stage 3: Hierarchical player workload and production
Allocate snaps/routes where observed, pass targets, catches and rush attempts among active skill players with shrinkage toward team/position and player opportunity priors. Use development-tested count/share distributions (e.g. hierarchical binomial/Dirichlet-multinomial or negative binomial) only when demonstrated appropriate out of sample.
Conditional yardage distributions need zero mass and overdispersion/tails assessed per market. Predict rushing yards given carries, receiving yards given targets/catches and passing yards given attempts/completions. Respect identities: individual receptions cannot sum above team completions, rush attempts cannot exceed generated team rush attempts, and player touchdowns must be reconciled with team touchdown events. Passing yards and team net passing yards differ because sacks are accounted for differently.
Separate P(active), conditional performance, rule-specific void probability and full unconditional distribution. Missing official stat rows are not automatically wager-settled under zero. Market rules and activation proof are a separate layer.
Generate correlated game/props directly from shared draws; do not multiply individual over probabilities for same-game parlays.

## Stage 4: Probability and scenario diagnostics
Draw sufficient independently seeded Monte Carlo samples for numerically stable probability estimates; report Monte Carlo standard errors and versioned seeds. 10,000–50,000 draws are engineering targets, NOT evidence of predictive accuracy. Parameter, lineup and model structural uncertainty should exceed simulated sampling noise. Scenario weights and weather/roster effects must be fitted and stress-tested, never fabricated.
Per game: expected/median home and away points, full total/margin distribution; 50/80/95% prediction intervals; straight-up P(home/away/tie); P(cover/push/loss) on an exact queried line; for each eligible player, volume distributions, P(over/under/push), independently flagged DNP/void scenario; pairwise covariance across market outcomes.

## Stage 5: Sequential ablation and independent evaluation
First replay canonical scores and props with prior-as-of features. Then test added possession/drive distributions, QB and pass/run efficiencies, player volume allocation, matchup/lineup data and forecasts one feature FAMILY at a time. Reject additions without replicated incremental gain; penalize complex models and high-missingness variables.
Train with rolling-origin expanding windows: each game's features/coefs must be available at its cutoff; no information from later kickoffs including games in same week. 2024–2025 previously examined seasons are exploratory, not a newly pristine holdout. Register separate future 2026+ challenger cohorts BEFORE kickoff; preserve old frozen forecasts.
Score diagnostic: home/away points, total, margin MAE/RMSE, bias, tail frequency, winner Brier/log-loss, full predictive log score/CRPS and coverage at 50/80/95% by game total, season phase, QB status, weather/roof and favorites.
Prop diagnostic: bias/MAE for observed stat, workload error, CRPS/log score, exact-line over Brier/reliability, calibration by position and market, role/activation and missingness buckets; score only source-qualified mature postgame results and never count seven related lines as independent games.
Evaluate candidate-minus-baseline paired metrics with game or week clustered uncertainty, multiple-comparison correction and sample adequacy. At least original preregistered minimum samples/gates, plus independent chronologically untouched evidence. No tuning on prospective outcomes from an already registered frozen cohort.

## Required automated assertions
- All source captures and original observed timestamps <= model forecast cutoff < kickoff; no future leakage.
- Stable game/player IDs, one record per original event/market, reject duplicates/misaligned season/week.
- Finite normalized probability vectors and monotone prop tail probabilities over increasing thresholds.
- Each simulation draw yields nonnegative integer scores and exact total = home+away; margin=home-away.
- Generated team plays and player opportunities conserve counts; QB/reception/target accounting and TD scoring reconcile with recorded modeled exceptions.
- Player with unknown status has scenario uncertainty; no silent DNP->zero for sportsbook settlement.
- Reproducible seeded simulations, sampling convergence and calibrated uncertainty reported.
- Every output indicates prediction status separately from bet qualification; without qualified exact sportsbook quote and independent calibration, model EV=NULL and action PASS_RESEARCH_ONLY.
- Original historical research ledger and registered rejection history never overwritten.

## Delivery order
A. Publish as-of data/coverage audit and canonical score error breakdown; prioritize availability and QB IDs.
B. Implement no-price game-level correlated drive/points challenger and benchmark against canonical score.
C. Only after valid game-level accuracy: add coherent pass/rush volume generator.
D. Then integrate player role shares, stat residual distributions, participation/void scenarios.
E. Run rolling backtests, clean prospective forecasts and feature ablations; report failures.
F. Add bookmaker-neutral market consensus benchmark and optional exact matched sportsbook line EV only after the independent football forecast is frozen. Sportsbook preference is NOT necessary to choose a team.

## Public research sources (access dates must be recorded in production)
- nflverse release and updates: https://github.com/nflverse/nflverse-data/releases and https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html
- nflfastR EPA and PBP: https://github.com/nflverse/nflfastR
- NWS weather forecast API: https://www.weather.gov/documentation/services-web-api

RELEASE: This RFC does not change current coefficients, historical results, predictive release gates, bookmaker assumptions, wagering execution, or registered prospective primary prop anchors. PASS continues until evidence supports otherwise.
