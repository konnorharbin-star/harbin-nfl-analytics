# NFL paired-score shadow — implementation checkpoint

This is the **first engineering prototype** under issue #194, not the possession, drive or player-prop engine. It does **not** improve or replace the production canonical model and cannot assert a profitable betting edge.

Components:
- `nfl/joint_score_shadow.py`: deterministic, paired historical home/away score-error bootstrap using **originally frozen** football-only point forecasts.
- `scripts/nfl_joint_score_shadow_run.py`: manual CLI with an explicit independently verified historical-pairs file; refuses missing, stale or lookahead input.
- `tests/test_joint_score_shadow.py`: synthetic invariant and chronology tests. These do not represent observed predictions or betting records.

The paired-residual method is an engineering baseline. It shares home/away error samples to retain empirical within-game dependence and derives integer score, total, margin and optional threshold push odds from the **same draw**. It neither fits possessions and per-drive events nor predicts player statistics. A zero-clipping or residual-data selection effect could bias predictions; both require empirical out-of-sample analysis before promotion.

The required historical-pairs source is **not currently provided** by the original score archive. Each JSONL entry needs a unique `game_id`, `season`, `week`, the *original* pregame `forecast_captured_utc`, `kickoff_utc`, independently verified `final_first_verified_utc` at least 72 hours after kickoff, `independent_final_source_verified:true`, cryptographic file/source hashes `original_forecast_sha256` and `final_source_sha256`, model pregame `forecast_home_points` / `forecast_away_points`, and integer official `actual_home_points` / `actual_away_points`. Require >=64 distinct games and >=8 different weeks **completely eligible before the specific target forecast capture**. Format validation and the Boolean source attestation do not independently confirm an official source; a separate real-source ingestion/verification job is needed. Never fabricate the history, backdate it or use games that hadn't been completed/graded by target forecast time.

Example invocation **only after collecting qualified historical pairs**:

```bash
python -m scripts.nfl_joint_score_shadow_run \
  --frozen history/score_forward_2026.jsonl \
  --verified-paired-history /path/to/qualified_asof_paired_games.jsonl \
  --game-id <ORIGINAL_FUTURE_GAME_ID> \
  --draws 10000 --seed 20261012 \
  --total-line 47.5 --home-spread -3.5 \
  --out reports/joint_score_shadow.json
```

The sample original Week 5 score freezes in the present repo are past kickoffs as of October 12 late evening; this command is **not an active current-game forecast**. New first-seen score records must exist strictly before future kickoffs. The CLI will give a research PASS blocker rather than supplying an arbitrary hypothetical distribution.

**Next engineering work (not yet delivered here):**
1. Build a verifiable historical as-of score paired-results adapter; evaluate frozen predictions and original source timestamps against official completion and missingness.
2. Compare this empirical paired-score simulation against the original calibrated score distribution on genuinely out-of-sample games (MAE, log/CRPS, calibration and zero-clipping sensitivity). Candidate cannot be promoted on synthetic regression tests.
3. Fit a common possessions and empirical per-drive scoring process from prior NFL PBP; add turnover/defensive scores and correct OT. Then link pass/run plays and player usage/yardage generation. Do not label the score-level bootstrap a true possession simulator.
4. Freeze a separate joint-model prospective cohort without changing the original registered game and prop forecasts.
5. Only after validated score/prop predictive improvement, compare against optional bookmaker-neutral exact market quotes.

All outputs are RESEARCH_ONLY, EV null, wager_authorized false. No sportsbook needed for original football predictions, no paid APIs, no auto-wagers.
