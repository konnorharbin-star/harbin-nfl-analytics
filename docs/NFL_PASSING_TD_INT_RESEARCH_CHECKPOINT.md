# NFL score-consistent passing touchdowns and interceptions: research checkpoint

Status: **UNVALIDATED CHALLENGER**, not released to betting or the canonical predictor.

This build adds free nflverse PBP fields `pass_touchdown`, `rush_touchdown`, `interception`, and `two_point_attempt` to each **previously completed, mature, before-as-of** drive. It excludes two-point attempts and rejects drives whose claimed scoring/result cannot be reconciled to exactly one offensive passing/rushing touchdown. Unsupported touchdowns (defensive, special-teams, atypical scores) are **excluded from the typed event library**, not misreported as an offensive touchdown. This keeps the existing score 0/3/7 simplification explicit.

The existing tied home/away score/volume simulation now attaches observed-type prior events to sampled drives:
- Team passing TDs, rushing TDs and interceptions come from the same sampled drives that generate game scores and team pass/rush opportunities.
- Every sampled passing touchdown is assigned to a completed team pass and a simulated target recipient; every sampled rushing touchdown is assigned to a simulated team rushing attempt.
- Passer touchdowns are assigned to completed QB passing attempts, and interceptions to incomplete QB passing attempts. Total QB + OTHER TD/INT counts reconcile to team totals. The player IDs remain previous-four-week QB attempt share **proxies** (not verified starters).
- All passing TD, interceptions, rushing and receiving TD research distributions appear only when every selected historical drive has typed, mature source events. Mixed or missing event fields fail closed or remain unavailable in the legacy source contract.
- The postgame source grader uses actual nflverse player-stat columns: `passing_tds`, `passing_interceptions`, `attempts`, `completions`, `receiving_tds` and `rushing_tds`. Missing individual player rows are not graded as zero.

This is a **game-level event-count bootstrap**, NOT a complete time-resolved pass-play simulator. Passer/receiver TD assignments are conditionally sampled within a *game* rather than tied to the same physical play, and correlations may be poorly calibrated. Defensive TDs, safeties, overtime rule variants, conversions and score 6/8 remain limitations of the original simulator. Avoid publishing an unconditional projection when the expected starter or activity is unknown.

Required next work: source and roster/expected starting quarterback verified from official team/NFL or full ESPN/CBS Sports articles; improve score/drive accounting; validate forecast interval/tail Brier and CRPS on **distinct frozen unseen games**, compared with the canonical baseline before promotion; reconcile game/player outcomes after official/source 72-hour maturity. Testing event conservation is not proof of predictive skill.

No sportsbook dependency, market price, approved bet, promised win rate, automatic account action or estimated economic EV. The original Week 6 forecasts and research log must not be rewritten by this improvement.

Sources: https://nflreadr.nflverse.com/articles/dictionary_pbp.html and https://nflreadr.nflverse.com/articles/dictionary_player_stats.html (public nflverse PBP and player weekly stat field dictionaries); the exact available source and observation time must still be recorded per future prospective scan.
