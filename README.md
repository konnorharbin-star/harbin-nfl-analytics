# Harbin NFL Analytics

Leakage-safe NFL projection, probability, market-analysis, backtesting, risk-management, and monitoring platform.

> **Current state:** Stage 2 — independent fair-score research and chronological football-feature validation. The repository is research software. A successful workflow run is not evidence of a profitable betting edge.

## Design

The NFL platform follows the same separation-of-concerns philosophy as the CFB system while keeping NFL training data, features, thresholds, evidence, and release state fully independent.

1. **Data foundation** — schedules/results, play-by-play, team statistics, rosters, injuries, depth charts, caching, schema contracts, and anti-leakage primitives.
2. **Fair-score engine** — opponent-adjusted football ratings and NFL-specific dynamic features. Sportsbook prices do not enter the football score projection.
3. **Probability + market layer** — score distributions, calibrated win/cover/total probabilities, no-vig market comparison, executable-price checks, line movement, and CLV.
4. **Context layer** — quarterback state, personnel/injuries, offensive-line availability, rest, travel, stadium/roof, and weather.
5. **Risk + execution layer** — fractional Kelly, concentration caps, drawdown throttles, price provenance, and fail-closed stake approval.
6. **Proof + release layer** — chronological walk-forward backtests, calibration, ROI/CLV, max drawdown, confidence intervals, shadow/live grading, monitoring, and RESEARCH/PAPER/SHADOW/PRODUCTION gates.

## What is implemented now

### Stage 1 — data foundation

The free source layer uses `nflreadpy`/nflverse for schedules, play-by-play, team statistics, rosters, injuries, and depth charts. Source frames are cached locally as parquet and validated against minimum platform contracts.

A live 2025 source audit validated 285 schedule rows, 285 completed games, 570 team-game rows and 48,771 PBP rows.

### Stage 2 — independent football model

The current fair-score baseline estimates team scoring from completed football games only:

```text
expected_points = league_points + offense(team) - defense(opponent) + home_field
```

Team offense/defense effects are ridge-regularized. The model outputs projected home points, away points, margin and total without reading sportsbook prices.

The PBP layer adds pregame-only matchup features for:

- EPA/play;
- success rate;
- passing EPA/dropback;
- rushing EPA/attempt;
- explosive-play rate;
- early-down EPA.

Historical game-level datasets are reconstructed independently week by week using only exact prior regular-season game IDs. Preseason games and target-week PBP are excluded from regular-season predictors.

### First untouched residual test

The first six-feature linear residual candidate was tuned chronologically on data before the 2025 holdout and **did not improve the independent baseline**, so it remains disabled.

On the untouched 2025 holdout:

- margin MAE: baseline `10.591` vs adjusted `10.642`;
- margin RMSE: baseline `13.182` vs adjusted `13.261`;
- total MAE: baseline `10.538` vs adjusted `10.708`;
- total RMSE: baseline `13.336` vs adjusted `13.452`.

This negative result is retained as evidence: advanced features must earn their place on later data rather than being forced into the projection.

## Anti-leakage rule

For a target game in season `S`, week `W`, rolling team state must be created only from information known before that game. Target-week/future results and PBP are excluded from predictor construction.

Current injury/depth-chart data must never be backfilled into historical games unless point-in-time historical records prove it was known at the simulated decision time.

## Quick start

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
python run_stage1.py 2025 --refresh
python run_stage2.py 2025 10 --refresh
python run_walkforward.py 2025 --start-week 5 --end-week 10 --refresh
python run_residual_audit.py --refresh
```

The corresponding source and holdout audits also run in GitHub Actions.

## Key files

- `nfl/contracts.py` — schema and fail-closed data contracts.
- `nfl/data.py` — nflverse ingestion, caching and schedule anti-leakage helpers.
- `nfl/ratings.py` — independent ridge fair-score baseline.
- `nfl/advanced.py` — leak-free PBP efficiency features.
- `nfl/dataset.py` — chronological game-level modeling dataset construction.
- `nfl/evaluation.py` — baseline football-projection error metrics.
- `nfl/residuals.py` — nested chronological residual-candidate validation.
- `nfl/stage1.py` / `nfl/stage2.py` — source integration audits.
- `nfl/walkforward.py` — week-by-week reconstruction audit.
- `nfl/residual_audit.py` — multi-season validation/holdout audit.

## Non-negotiable model rules

- No sportsbook line may leak into the independent fair-score engine.
- No target-week/future results may enter a pregame feature.
- Missing data lowers confidence or blocks a path; it is not silently invented.
- Historical evaluation is chronological, not random train/test shuffling across time.
- A learned adjustment stays disabled if it fails later untouched data.
- A model reaches production only through independent evidence, not because code executes successfully.
- NFL thresholds, weights, calibration, and release evidence are independent of the CFB system.

See the Stage 1 and Stage 2 documents under `docs/` for the current contracts and validation design.
