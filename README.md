# Harbin NFL Analytics

Leakage-safe NFL projection, probability, market-analysis, backtesting, risk-management, grading, monitoring, and release-control platform.

> **Current state:** the football/probability/market research stack is implemented through Stage 3, and the CFB-style operational shell is now present: policy, capped Kelly, portfolio controls, line capture, forward decision ledger, independent grading, evidence reports, monitoring, health, model card, canonical reporting, and hard `RESEARCH -> PAPER -> SHADOW -> PRODUCTION` gates. The remaining major parity gap is Stage 4 NFL context: timestamp-safe injuries/personnel, rest/travel, stadium/roof, and weather. Until those and the independent evidence gates pass, the model must remain non-production.

## Design

The NFL platform intentionally mirrors the architecture of `harbin-sports-analytics` while keeping NFL football data, coefficients, probability calibration, betting thresholds, and release evidence independent.

1. **Data foundation** — schedules/results, play-by-play, team statistics, rosters, injuries, depth charts, caching, schema contracts, and anti-leakage primitives.
2. **Fair-score engine** — NFL scoring ratings and NFL-specific dynamic features. Sportsbook prices do not enter the football score projection.
3. **Probability + market layer** — score distributions, win/cover/total probabilities, no-vig market comparison, free nflverse archive research, free ESPN current markets, optional richer provider provenance, line movement, CLV, and chronological market-rule evidence.
4. **Context layer** — quarterback state, personnel/injuries, offensive-line availability, rest, travel, stadium/roof, and weather. QB state exists now; the broader point-in-time context layer remains under construction.
5. **Risk + execution layer** — policy-driven PASS/LEAN/BET/STRONG classification, capped fractional Kelly, game/team/market/book/kickoff/slate caps, drawdown throttles, quote validation, and fail-closed stake approval.
6. **Proof + release layer** — historical evidence reports, line-capture ledger, portfolio-decision ledger, independent live grading, monitoring, health, model card, canonical reporting, and hard release gates.

Structural parity does **not** transfer CFB evidence to the NFL. An NFL feature, threshold, risk setting, or release state must be earned with NFL data.

## What is implemented now

### Stage 1 — data foundation

The free source layer uses `nflreadpy`/nflverse for schedules, play-by-play, team statistics, rosters, injuries, and depth charts. Source frames are cached locally as parquet and validated against minimum platform contracts.

A live 2025 source audit validated 285 schedule rows, 285 completed games, 570 team-game rows and 48,771 PBP rows.

### Stage 2 — independent football model

The canonical fair-score model estimates team scoring from completed football games only:

```text
expected_points = league_points + offense(team) - defense(opponent) + home_field
```

Team offense/defense effects are ridge-regularized. The active season's completed pregame team-game rows receive full sample weight. The immediately prior regular season is a validated low-weight scoring prior: each prior-season team-game row receives weight `0.10`. Postseason, seasons older than `N-1`, target-week results and future results are excluded.

The `0.10` prior was selected on 2024 data and then scored once on the untouched 2025 holdout. Over 256 holdout games it improved all four score-error gates versus the same-season-only baseline:

- margin MAE: `10.707` → `10.607`;
- margin RMSE: `13.452` → `13.340`;
- total MAE: `10.753` → `10.585`;
- total RMSE: `13.549` → `13.380`.

The model outputs projected home points, away points, margin and total without reading sportsbook prices.

The PBP layer adds pregame-only matchup features for EPA/play, success rate, passing EPA/dropback, rushing EPA/attempt, explosive-play rate, and early-down EPA. Historical game-level datasets are reconstructed independently week by week using only exact prior regular-season game IDs for PBP features. Preseason games and target-week PBP are excluded from regular-season predictors.

The current/upcoming-game pipeline emits the canonical fair score separately from research/shadow quarterback adjustments. Research or shadow output is never silently substituted for the canonical baseline.

### Stage 2 candidate evidence retained

A simple exponential recency-weighting candidate selected a 12-week half-life on 2024. On the untouched 2025 holdout it improved MAE slightly but worsened both margin and total RMSE, so it failed closed and was not promoted.

The first six-feature linear PBP residual candidate also remains disabled after the scoring-prior promotion. Against the promoted baseline on the untouched 2025 holdout:

- margin MAE: baseline `10.607` vs adjusted `10.671`;
- margin RMSE: baseline `13.200` vs adjusted `13.265`;
- total MAE: baseline `10.484` vs adjusted `10.632`;
- total RMSE: baseline `13.303` vs adjusted `13.413`.

A second opponent-adjusted EPA/success-rate candidate also failed the untouched 2025 holdout and remains disabled. Negative results are retained as evidence: advanced-looking features do not enter the canonical score simply because they are available.

### Stage 3 — probability and market layer

The probability layer fits Gaussian residual distributions around independently generated margin and total projections. Distribution parameters come only from earlier chronological football residuals. Sportsbook lines are thresholds evaluated afterward; they are not football-model inputs.

The first dispersion-scaling experiment selected scale `1.0` for both margin and total. On the untouched 2025 holdout of 208 games, the research Gaussian baseline recorded approximately:

- margin NLL: `4.00844`;
- total NLL: `4.00855`;
- home-win Brier score: `0.23405`;
- margin 50% / 80% interval coverage: `46.15% / 75.96%`;
- total 50% / 80% interval coverage: `54.81% / 79.81%`.

A direct logistic home-win calibration finished marginally worse on the untouched holdout and remains disabled.

The sportsbook layer supports American/decimal conversion, implied and no-vig probabilities, ML/spread/total model probabilities, executable-price EV, model-vs-market probability edge, timestamp provenance, grading, and one selected opportunity per game/market in research summaries.

### Free historical and current markets — same philosophy as NCAA

The primary historical-market path is free. `nfl/free_market.py` uses nflverse/nfldata schedule-market fields for archived moneylines, spreads, totals and available prices, plus the public nflverse `initial_lines.csv` when a distinct opening spread/total exists.

The archive path follows the same discipline as the NCAA model:

- reconstruct football projections before attaching market data;
- use a distinct archived opening line when available;
- otherwise label the archive-final value as an explicit fallback;
- use `-110` only when a spread/total exists but the archived side price is missing;
- never invent a moneyline;
- grade one selected side per game/market;
- report units, ROI, drawdown, uncertainty and available opening-to-archive-final CLV proxy;
- never relabel an archive-final fallback as a verified opening or official close.

For live weeks, `nfl/espn_market.py` uses free ESPN NFL scoreboard/Core odds. The Odds API remains optional enrichment for richer timestamped/multi-book research; it is not required for the core model.

### CFB-style operational shell

The NFL repository now has the same major operational boundaries as the NCAA repository:

```text
football data
-> independent fair score
-> chronological probability distribution
-> verified current market
-> policy signal
-> capped fractional Kelly
-> execution validation
-> portfolio concentration controls
-> forward decision ledger
-> independent grading
-> historical/live evidence
-> monitoring + health
-> hard release gate
-> canonical report
```

Key controls:

- `nfl/policy.py` supplies conservative NFL-specific PASS/LEAN/BET/STRONG research thresholds and capped fractional Kelly. These defaults are **not** claimed to be production-validated thresholds.
- `nfl/execution_market.py` fails closed on missing/invalid odds, required line, sportsbook provenance, quote timestamp, future timestamp, or stale quote.
- `nfl/portfolio.py` applies slate, game, team, market, book and kickoff-cluster caps plus drawdown/trailing-performance throttles. PAPER/SHADOW allocations can be recorded, but real `portfolio_stake_units` remain zero unless the hard production gate is open.
- `nfl/line_history.py` writes timestamped forward market snapshots and uses explicit UTC kickoff times so pre-kickoff CLV evidence cannot depend on the runner's local timezone.
- `nfl/decision_ledger.py` persists only cap-constrained PAPER/SHADOW/BET decisions and appends a new row only when the executable portfolio state changes.
- `nfl/grading.py` grades the earliest persisted decision per game/market against final NFL scores and can attach a later same-book pre-kickoff CLV observation.
- `nfl/proof.py` separates the broad free archive research sample from the stronger promotion sample. Archive-final fallbacks are excluded from verified-entry promotion evidence.
- `nfl/monitoring.py`, `nfl/health.py`, `nfl/model_card.py`, and `nfl/reporting.py` provide the same operational observability pattern as the CFB system.
- `nfl/release_gate.py` is a hard deployment gate. No weighted readiness score can override missing historical entry integrity, live evidence, context coverage, or other hard blockers.

### Release states

The model follows the same deployment-state concept as NCAA:

- **RESEARCH** — one or more engineering/data/context/calibration hard gates fail.
- **PAPER** — engineering gates pass, but historical promotion evidence is not robust enough.
- **SHADOW** — robust NFL historical evidence exists, but independent live/forward evidence is not yet sufficient for production.
- **PRODUCTION** — engineering, verified historical entry evidence, market breadth, portfolio-verified forward grading, ROI/CLV requirements, and live sample requirements all pass.

No state is manually promoted merely because the code runs.

### Current intentional blocker: Stage 4 context

The QB subsystem exists, but broad timestamp-safe NFL context is not yet complete. Until injuries/personnel, rest/travel, stadium/roof, and weather are implemented with point-in-time-safe contracts, the canonical pipeline deliberately reports partial context coverage and the release gate remains closed. This is preferable to silently filling historical games with current information.

## Anti-leakage rules

For a target game in season `S`, week `W`, team state must be created only from information known before kickoff. Target-week/future results and PBP are excluded. The prior-season scoring state is restricted to the immediately preceding completed regular season.

Sportsbook data remains downstream. Timestamped simulations may use only quotes observable at the simulated decision time. Free archive opening/final values remain labeled by their actual archive stage. Current injuries/depth-chart/weather information cannot be backfilled into historical games without a timestamp-correct source.

## Quick start

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
ruff check .

# source/fair-score/probability audits
python run_stage1.py 2025 --refresh
python run_stage2.py 2025 10 --refresh
python run_walkforward.py 2025 --start-week 5 --end-week 10 --refresh
python run_prior_audit.py --refresh
python run_probability_audit.py --refresh

# current football-only projection
python run_current_projection.py 2026 --refresh

# free historical market research
python run_free_market_backtest.py --start-season 2022 --end-season 2025 \
  --validation-season 2024 --holdout-season 2025 --refresh

# canonical CFB-style NFL operational run
python harbin_nfl_model.py 2026 --refresh

# independent forward evidence utilities
python capture_lines.py --season 2026 --refresh
python grade_live.py
```

GitHub Actions also provide a canonical NFL model run, hourly line capture, scheduled live grading, free historical backtesting, current-market comparison, and the existing football/probability holdout audits.

## Key files

### Football and probability
- `nfl/contracts.py` — schema and fail-closed data contracts.
- `nfl/data.py` — nflverse ingestion, caching and schedule anti-leakage helpers.
- `nfl/ratings.py` — canonical ridge fair-score baseline and validated prior-season weight.
- `nfl/advanced.py` — leak-free PBP efficiency features.
- `nfl/dataset.py` — chronological game-level modeling dataset construction.
- `nfl/evaluation.py` — baseline football-projection error metrics.
- `nfl/recency.py`, `nfl/priors.py` — candidate validation for recency/prior behavior.
- `nfl/residuals.py`, `nfl/opponent_adjusted.py`, `nfl/oa_dataset.py`, `nfl/oa_residuals.py` — advanced residual research.
- `nfl/probability.py`, `nfl/win_probability.py` — chronological probability research/calibration.
- `nfl/quarterbacks.py` and QB dataset/validation modules — NFL-specific QB state and shadow research.

### Market research
- `nfl/market.py` — odds math, no-vig probabilities and model-market comparison.
- `nfl/free_market.py` — free nflverse opening/archive-final market adapter.
- `nfl/free_market_backtest.py` — free archive grading, ROI/CLV proxy and holdout evidence.
- `nfl/espn_market.py` — free current ESPN market adapter.
- `nfl/odds_api.py` — optional richer historical provider adapter.
- `nfl/clv.py`, `nfl/market_history.py`, `nfl/market_backtest.py`, `nfl/market_validation.py` — timestamped provider research and validation.

### CFB-style operational parity
- `nfl/market_intel.py` — canonical current market-intelligence schema.
- `nfl/policy.py` — signal policy and capped fractional Kelly.
- `nfl/execution_market.py` — executable quote validation.
- `nfl/portfolio.py` — concentration caps and bankroll/drawdown throttles.
- `nfl/line_history.py` — forward market snapshot ledger.
- `nfl/decision_ledger.py` — append-only cap-constrained decision ledger.
- `nfl/grading.py` — independent forward grading.
- `nfl/proof.py` — historical evidence/promotion sample report.
- `nfl/monitoring.py` — live readiness and distribution drift.
- `nfl/release_gate.py` — hard deployment state machine.
- `nfl/health.py` — source/model operational health.
- `nfl/model_card.py` — machine-readable model card.
- `nfl/reporting.py` — canonical report/publication bundle.
- `nfl/pipeline.py` — canonical end-to-end operational run.
- `harbin_nfl_model.py` — primary operational entrypoint.
- `capture_lines.py` / `grade_live.py` — forward evidence entrypoints.

See `docs/parity_plan.md` plus the Stage 1–3 documents under `docs/` for architecture and validation contracts.

## Non-negotiable model rules

- No sportsbook line may leak into the independent fair-score engine.
- No target-week/future result may enter a pregame feature.
- Missing data lowers confidence or blocks a path; it is not silently invented.
- Historical evaluation is chronological, not random train/test shuffling across time.
- A learned adjustment stays disabled if it fails later untouched data.
- Current context is not backfilled historically without timestamp-correct evidence.
- A timestamped sportsbook backtest may use only quotes observable at the simulated decision time.
- Archive-final values are never promoted by calling them verified openings or official closes.
- Closing prices/CLV are evaluation evidence, not football-model features.
- Multiple books do not create multiple independent bets on the same modeled game/market.
- No uncapped Kelly and no stake outside the production-controlled portfolio path.
- NFL thresholds, weights, calibration, and release evidence remain independent of CFB.
- A model reaches production only through independent evidence, not because code executes successfully.
