# Harbin NFL Analytics

Leakage-safe NFL projection, probability, market-analysis, backtesting, context, risk-management, grading, monitoring, publication, and release-control platform.

> **Current state:** the NFL repository now closely mirrors the operating architecture of `harbin-sports-analytics`: independent fair scores, chronological probabilities, free/optional multi-book markets, context, policy, capped Kelly, portfolio controls, forward line/decision ledgers, independent grading, audit reports, monitoring, health, model card, weekly publication, and hard `RESEARCH -> PAPER -> SHADOW -> PRODUCTION` gates. Structural parity does **not** transfer NCAA betting evidence to the NFL. The NFL remains fail-closed until its own historical-entry and forward live/shadow evidence satisfy the release gate.

## Design

The NFL platform intentionally mirrors the NCAA model's architecture while keeping NFL data, coefficients, context behavior, probability calibration, betting thresholds, and release evidence independent.

1. **Data foundation** — schedules/results, play-by-play, team/player statistics, rosters, injuries, depth charts, caching, schema contracts, and anti-leakage primitives.
2. **Fair-score engine** — NFL scoring ratings and NFL-specific dynamic research features. Sportsbook prices never enter the football score projection.
3. **Probability + market layer** — score distributions, win/cover/total probabilities, no-vig comparison, free nflverse historical research, free ESPN current markets, optional multi-book enrichment, line movement, CLV, and chronological market-rule evidence.
4. **Context layer** — quarterback state, injuries/personnel, depth-chart/roster availability, offensive-line/skill/defensive availability summaries, rest, travel, stadium/roof, and weather. Context remains post-prediction confidence/risk unless an NFL point-in-time holdout validates a score adjustment.
5. **Risk + execution layer** — policy-driven PASS/LEAN/BET/STRONG classification, capped fractional Kelly, game/team/market/book/kickoff/slate caps, drawdown throttles, quote validation, and fail-closed stake approval.
6. **Proof + release layer** — historical evidence, backtest/runtime audits, forward market snapshots, portfolio decisions, independent grading, monitoring, health, model card, weekly publication, and hard release gates.

## What is implemented

### Stage 1 — data foundation

The free source layer uses `nflreadpy`/nflverse for schedules, play-by-play, team/player statistics, weekly rosters, injuries, and depth charts. Source frames are cached locally as parquet and validated against platform contracts.

A live 2025 source audit validated 285 schedule rows, 285 completed games, 570 team-game rows and 48,771 PBP rows.

### Stage 2 — independent football model

The canonical fair-score baseline estimates scoring from completed football games only:

```text
expected_points = league_points + offense(team) - defense(opponent) + home_field
```

Team offense/defense effects are ridge-regularized. The active season's completed pregame rows receive full weight. The immediately prior regular season is a validated low-weight scoring prior (`0.10` per prior-season team-game row). Postseason, older seasons, target-week outcomes and future outcomes are excluded.

That `0.10` prior was selected on 2024 and then evaluated on the 2025 holdout. Over 256 holdout games it improved all four score-error gates versus the same-season-only baseline:

- margin MAE: `10.707` -> `10.607`;
- margin RMSE: `13.452` -> `13.340`;
- total MAE: `10.753` -> `10.585`;
- total RMSE: `13.549` -> `13.380`.

The PBP layer includes pregame-only EPA/play, success rate, pass EPA/dropback, rush EPA/attempt, explosive-play rate and early-down EPA. Historical game-level features are reconstructed independently week by week from exact prior regular-season game IDs.

Several advanced candidates remain disabled because later holdout evidence did not improve the promoted baseline. Negative results are retained; an advanced-looking feature is not promoted merely because it is available.

### Quarterback layer

NFL QB state is a first-class subsystem rather than an arbitrary manual point adjustment. Current projections emit last-observed QB state and validated QB research/shadow structures separately from the canonical baseline. The release labels remain explicit so a research adjustment cannot silently replace the fair score.

### Stage 3 — probability and market layer

The probability layer fits Gaussian residual distributions around independently generated margin and total projections. Distribution parameters come only from earlier chronological football residuals. Sportsbook lines are evaluated afterward as thresholds.

The initial dispersion experiment selected scale `1.0` for margin and total. On the 2025 holdout of 208 games, the research Gaussian baseline recorded approximately:

- margin NLL: `4.00844`;
- total NLL: `4.00855`;
- home-win Brier: `0.23405`;
- margin 50% / 80% coverage: `46.15% / 75.96%`;
- total 50% / 80% coverage: `54.81% / 79.81%`.

A direct logistic home-win calibration finished marginally worse and remains disabled.

The sportsbook layer supports American/decimal odds conversion, implied/no-vig probabilities, ML/spread/total model probabilities, executable-price EV, probability edge, price provenance, grading, and one selected opportunity per game/market.

### Historical market research

The primary historical path is free. `nfl/free_market.py` uses nflverse/nfldata schedule-market fields plus public `initial_lines.csv` when distinct opening values exist.

Rules match the NCAA evidence philosophy:

- build the football projection before attaching market data;
- use a distinct archived opening when available;
- otherwise label archive-final values as explicit fallbacks;
- never invent a moneyline;
- grade one selected side per game/market;
- report units, ROI, drawdown, uncertainty and available CLV proxy;
- never relabel an archive-final fallback as a verified opening or official close.

`nfl/backtest_runtime.py` adds NCAA-style runtime diagnostics without changing selection: week-block bootstrap ROI intervals, probability-equivalent CLV, and market/season/week/role/location segment reports.


`nfl/policy_calibration.py` now mirrors the NCAA nested policy-calibration contract. Only independently verified entry-price rows may select thresholds. Verified evidence is split chronologically into development, tuning, and untouched evaluation blocks; the evaluation block can reject a frozen policy but can never choose one. When verified entry data is absent, `reports/production_policy.json` remains explicitly PAPER and the hard release gate continues to block real staking.

### Current markets and line shopping

`nfl/espn_market.py` remains the free primary source. `nfl/pro_market.py` adds an optional professional/multi-book layer using The Odds API when `THE_ODDS_API_KEY` is configured.

The canonical current-market path:

- tries free ESPN first;
- optionally adds professional books;
- preserves sportsbook update timestamps where available;
- counts unique sportsbook identity rather than feed/provider identity;
- evaluates every quote downstream of the independent projection;
- retains one best executable quote per game/market;
- records per-market book count and game-level multi-book coverage;
- uses the same aggregation path for scheduled forward line capture.

A duplicated sportsbook exposed through two providers does not count as two books.

### Stage 30 — market-edge reliability research

The broad free archive currently shows negative betting ROI despite large raw model
edges. `nfl/market_shrinkage.py` therefore tests a downstream log-odds shrinkage of
model probabilities toward the no-vig market. Alpha is selected before the 2024
validation and frozen into the untouched 2025 holdout. Betting ROI does not select
alpha, sportsbook probabilities never enter the fair-score model, and the experiment
cannot change the canonical policy while archive entry prices remain unverified.

### Stage 31 — persisted candidate benchmark

The rolling fair-score candidate benchmark is now a persisted research artifact rather
than a dormant manual workflow. It compares recency, prior-season shrinkage,
opponent-adjusted PBP residuals, and quarterback-state residuals on the common
2023->2024 and 2024->2025 development folds. Baseline remains selectable, and any
nonzero winner is only development architecture evidence; the later fixed-spec and
forward gates remain authoritative.

### Stage 32 — research-status contract

`nfl/research_status.py` consolidates the candidate benchmark, market-edge shrinkage
result, and Phase 5 forward ledgers into one fail-closed research registry. It records
which ideas are rejected, superseded, or still accumulating prospective evidence and
surfaces the current research decision in the model card. The registry cannot enable a
canonical model or market change.

### Stage 33 — forward-ledger health

`nfl/forward_ledger_health.py` audits inception-aware capture completeness for every
frozen 2026 candidate. A future candidate promotion now requires exactly one valid
pre-kickoff snapshot for every eligible game in each opened week, with no late,
duplicate, wrong-spec, or schedule-inconsistent rows. Games that kicked off before a
candidate's first persisted snapshot remain outside that candidate's prospective sample
and are never backfilled.

### Stage 34 — forward betting evidence integrity

Forward betting grades now require execution-ready, timestamp-valid entry provenance.
Phase 5 and the hard release gate also require 100% entry/execution coverage and at
least 90% valid later pre-kickoff closing-snapshot coverage before positive CLV can
support release. The independent grading audit is persisted with live evidence.

### Stage 35 — generated-state writer serialization

The first attempt used one shared GitHub Actions concurrency lane for all writers.
Post-merge validation showed that GitHub retains only one pending run per concurrency
group, so legitimate pending writers could be cancelled. Stage 36 supersedes this
approach.

### Stage 36 — generated-state push reconciliation

Generated-state workflows now keep workflow-scoped concurrency and all use
`scripts/push_generated_state.sh`. Non-conflicting cross-workflow push races are
fetch/rebase/retry reconciled, while true content conflicts fail closed instead of
overwriting another writer. CI simulates both cases against a local bare Git remote.

### Stage 37 — ESPN multi-book breadth recovery

The free ESPN collector now always treats Core odds as optional provider enrichment and
preserves one row per game × market × canonical sportsbook instead of collapsing all
providers to one market row. Core outages fall back to complete scoreboard quotes, and
the live diagnostic reports actual normalized sportsbook breadth.

### Stage 38 — verified historical market evidence

The optional timestamped historical provider is now connected to the canonical proof,
nested policy calibration, and release gate. A manual API-key workflow can build
fixed-time pre-kickoff entry/closing evidence without changing the fair-score model.
Free nflverse operation remains the default PAPER path; historical promotion still
requires the full verified-sample ROI, CLV, breadth-across-markets/seasons, and sample
gates.

### Stage 39 — release/policy state coherence

The hard release gate now requires the frozen nested policy itself to be in production
mode with at least two validated markets. This matches the allocator's existing safety
rule, so the model card, release report, and executable staking state can no longer
disagree about PRODUCTION eligibility.

### Stage 40 — generated-state source revision guard

Generated-state writers may still reconcile with other generated writers, but they now
stop cleanly if newer source/workflow/test changes have landed since the run began.
This prevents stale calculations from rebasing across a newer model revision and
temporarily replacing outputs produced by newer code.

### Stage 4 — current NFL context

The context layer now mirrors the NCAA current-only philosophy:

- nflverse injury reports filtered by target week and as-of timestamp;
- latest admissible player status supersedes older reports;
- separate QB injury risk;
- modern timestamped and legacy weekly depth-chart support;
- weekly roster availability and active-QB checks;
- top-depth, offensive-line, skill-position and defensive injury-risk summaries;
- home/away rest-day context;
- deterministic travel miles and timezone-shift context;
- neutral-site venue lookup;
- indoor/closed-roof weather bypass;
- free Open-Meteo outdoor forecasts near kickoff;
- temperature, precipitation, wind, gust and bounded weather-risk diagnostics.

Current context refuses historical-season attachment by default. Missing context lowers coverage or blocks a path; it is not converted into a favorable assumption. `score_adjustment_enabled` remains false until a point-in-time NFL historical test demonstrates improvement.

### CFB-style operational shell

The canonical path now follows the same shape as NCAA:

```text
football data
-> independent fair score
-> chronological probability distribution
-> canonical current market aggregation
-> line shop / no-vig / EV
-> policy signal
-> capped fractional Kelly
-> execution validation
-> portfolio concentration controls
-> forward decision ledger
-> independent grading
-> historical + forward evidence
-> monitoring + health + audit snapshot
-> hard release gate
-> canonical JSON/CSV + weekly HTML/PNG publication
```

Key controls:

- `nfl/policy.py` — NFL-specific PASS/LEAN/BET/STRONG research policy and capped fractional Kelly.
- `nfl/execution_market.py` — fails closed on invalid odds/line, sportsbook provenance, market-book count, stale/future quote, or a quote that is not strictly pre-kickoff.
- `nfl/portfolio.py` — slate/game/team/market/book/kickoff caps, committed-open exposure reservation, drawdown throttles, and fail-closed production halts. Real stake stays zero unless every production safety check passes.
- `nfl/line_history.py` — timestamped forward market snapshots with explicit UTC kickoff.
- `nfl/decision_ledger.py` — append-only cap-constrained PAPER/SHADOW/BET decisions; open production BETs are reserved against later-run exposure caps.
- `nfl/grading.py` — independent grading of the earliest eligible decision per game/market and later pre-kickoff CLV observations.
- `nfl/proof.py` — separates broad archive research from stronger promotion-quality entry evidence.
- `nfl/monitoring.py`, `nfl/health.py`, `nfl/model_card.py` — operational observability.
- `nfl/release_gate.py` — hard state machine; a weighted score cannot override failed evidence gates.
- `nfl/render.py` — NCAA-style one-row-per-game weekly HTML/PNG board with release-state labeling.

### Phase 5 — consolidated forward-shadow validation

The prospective evidence streams now have one read-only acceptance layer in
`nfl/forward_shadow.py`. It consolidates the portfolio-verified betting ledger,
recent-form total shadow, QB-total shadow, and the two probability-shadow targets
without rebuilding historical predictions or retuning candidates on 2026 outcomes.

The generated `reports/forward_shadow_summary.json` keeps candidate promotion evidence
separate from canonical betting evidence. The canonical forward gate requires at least
300 graded portfolio decisions with non-negative ROI and positive CLV; each frozen
score/probability candidate retains its own 128-game statistical gate. The consolidated
report has no authority to change the canonical model or open PRODUCTION.

### Phase 6 — production safety shell

The production allocator now fails closed at the final execution boundary as well as
at the release gate. When production is open, a stale/non-executable quote remains
`PASS` with zero candidate allocation, so it cannot consume concentration capacity.
If a downstream bankroll/history safety rule blocks an otherwise open production
release, the portfolio reports `mode=halted`, zero allocation, and an explicit
`production_block_reason`.

Repeated production runs also reserve already-authorized future `BET` decisions from
the append-only decision ledger before sizing new opportunities. Existing open stakes
count against slate, game, team, market, book, kickoff-window, and max-bet limits, and
the same game/market cannot be re-approved while that production bet remains open.
Production fails closed if the committed-exposure ledger is missing or invalid.

### Monitoring gate separation

Operational monitoring now reports both an all-in live-readiness score and an
engineering-readiness score. The engineering score excludes sportsbook breadth because
multi-book consensus is already a separate production gate. This prevents a valid
single-book PAPER/SHADOW build from being mislabeled RESEARCH while preserving the
independent >=75% verified multi-book requirement for PRODUCTION.

### NCAA-style audit suite

The NFL repository also has explicit machine-readable audits:

- `nfl/feature_audit.py` — feature parity, leakage-name checks, missingness and drift;
- `nfl/backtest_audit.py` — quote provenance, opening/final separation and CLV integrity;
- `nfl/grading_audit.py` — strict pre-kickoff forward decisions and closing chronology;
- `nfl/portfolio_audit.py` — cap enforcement, execution eligibility and production-gate checks;
- `nfl/audit_snapshot.py` — consolidated feature/backtest/grading/portfolio/health/release snapshot.

The `NFL Audit Snapshot` GitHub Actions workflow produces the same PASS/WARN/FAIL style used operationally by the NCAA project.

## Release states

- **RESEARCH** — one or more engineering/data/context/calibration hard gates fail.
- **PAPER** — engineering gates pass, but historical promotion evidence is not robust enough.
- **SHADOW** — robust NFL historical evidence exists, but independent forward evidence is insufficient for production.
- **PRODUCTION** — engineering, verified historical entry evidence, market breadth, portfolio-verified forward grading, ROI/CLV and sample requirements all pass.

No state is manually promoted merely because the code executes successfully.

The weekly HTML/PNG board always prints its release state. A PAPER/SHADOW opportunity may be displayed for evaluation, but it is not labeled as a production bet.

## Anti-leakage rules

For a target game in season `S`, week `W`, team state must be created only from information known before kickoff. Target-week/future results and PBP are excluded. Prior-season scoring state is restricted to the immediately preceding completed regular season.

Sportsbook data remains downstream. Timestamped simulations may use only quotes observable at the simulated decision time. Current injuries/depth/weather cannot be backfilled into historical games without timestamp-correct evidence. Closing prices and CLV are evaluation evidence, never football-model features.

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

# current independent projection
python run_current_projection.py 2026 --refresh

# free historical market research + NCAA-style runtime diagnostics
python run_free_market_backtest.py --start-season 2022 --end-season 2025 \
  --validation-season 2024 --holdout-season 2025 --refresh

# optional timestamped historical evidence (requires THE_ODDS_API_KEY)
python run_verified_market_backtest.py --start-season 2022 --end-season 2025

# regenerate the frozen verified-entry market policy from existing evidence
python run_policy_calibration.py
python run_market_edge_shrinkage.py

# canonical NCAA-style NFL operational run + weekly board
python harbin_nfl_model.py 2026 --refresh

# forward evidence utilities
python capture_lines.py --season 2026 --refresh
python grade_live.py
python run_forward_shadow.py
python run_audit_snapshot.py
```

If `THE_ODDS_API_KEY` is absent, the canonical current path still runs on free ESPN data. The optional source only enriches breadth and price shopping.

## Key files

### Football and probability

- `nfl/contracts.py` — schema/fail-closed data contracts.
- `nfl/data.py` — nflverse ingestion, caching and anti-leakage helpers.
- `nfl/ratings.py` — canonical fair-score baseline and validated prior-season weight.
- `nfl/advanced.py`, `nfl/dataset.py` — leak-free PBP features and chronological datasets.
- `nfl/probability.py`, `nfl/win_probability.py` — chronological score/win probability research.
- QB dataset/state/validation modules — NFL-specific quarterback research and release labels.

### Context

- `nfl/injuries.py` — point-in-time injury normalization.
- `nfl/personnel.py` — depth-chart/roster availability.
- `nfl/weather.py` — venue, travel and Open-Meteo weather context.
- `nfl/context.py` — current-only context aggregation and coverage metadata.

### Markets and proof

- `nfl/market.py` — odds math, no-vig probabilities and model-market comparison.
- `nfl/free_market.py`, `nfl/free_market_backtest.py` — free archive adapter and chronological grading.
- `nfl/backtest_runtime.py` — block-bootstrap/CLV/segment runtime diagnostics.
- `nfl/espn_market.py` — free current ESPN adapter.
- `nfl/pro_market.py` — optional multi-book current aggregation.
- `nfl/odds_api.py` — optional timestamped historical-provider adapter.
- `nfl/market_intel.py` — canonical current line shopping and market-intelligence rows.

### Operations

- `nfl/policy.py`, `nfl/execution_market.py`, `nfl/portfolio.py` — policy, execution, concentration limits, bankroll throttles, and explicit production HALT controls.
- `nfl/line_history.py`, `nfl/decision_ledger.py`, `nfl/grading.py` — forward evidence ledgers and grading.
- `nfl/forward_shadow.py` — consolidated Phase 5 prospective evidence and acceptance state.
- `nfl/proof.py`, `nfl/monitoring.py`, `nfl/release_gate.py`, `nfl/health.py`, `nfl/model_card.py` — proof and release controls.
- `nfl/reporting.py`, `nfl/render.py`, `nfl/pipeline.py` — canonical machine + human publication path.
- `harbin_nfl_model.py` — primary operational entrypoint.

See `docs/parity_plan.md` and the staged architecture/validation documents under `docs/` for the development contracts.

## Non-negotiable rules

- No sportsbook line may leak into the independent fair-score engine.
- No target-week/future result may enter a pregame feature.
- Missing data lowers confidence or blocks a path; it is not silently invented.
- Historical evaluation is chronological, not random time-shuffled train/test validation.
- A learned adjustment stays disabled if it fails later holdout data.
- Current context is not backfilled historically without timestamp-correct evidence.
- A timestamped sportsbook backtest may use only quotes observable at the simulated decision time.
- Archive-final values are never promoted by calling them verified openings or official closes.
- Closing prices/CLV are evaluation evidence, not football-model features.
- Multiple books do not create multiple independent bets on the same game/market.
- Multiple data feeds exposing the same sportsbook do not create synthetic multi-book breadth.
- No uncapped Kelly and no stake outside the production-controlled portfolio path.
- NFL thresholds, weights, calibration and release evidence remain independent of NCAA.
- Production is earned through independent NFL evidence, not because the architecture matches NCAA.
