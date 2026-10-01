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

The canonical fair-score model estimates team scoring from completed football games only:

```text
expected_points = league_points + offense(team) - defense(opponent) + home_field
```

Team offense/defense effects are ridge-regularized. The active season's completed pregame team-game rows receive full sample weight. The immediately prior regular season is now a validated low-weight scoring prior: each prior-season team-game row receives weight `0.10`. Postseason, seasons older than `N-1`, target-week results and future results are excluded.

The `0.10` prior was selected on 2024 data and then scored once on the untouched 2025 holdout. Over 256 holdout games it improved all four score-error gates versus the same-season-only baseline:

- margin MAE: `10.707` → `10.607`;
- margin RMSE: `13.452` → `13.340`;
- total MAE: `10.753` → `10.585`;
- total RMSE: `13.549` → `13.380`.

The model outputs projected home points, away points, margin and total without reading sportsbook prices.

The PBP layer adds pregame-only matchup features for:

- EPA/play;
- success rate;
- passing EPA/dropback;
- rushing EPA/attempt;
- explosive-play rate;
- early-down EPA.

Historical game-level datasets are reconstructed independently week by week using only exact prior regular-season game IDs for PBP features. Preseason games and target-week PBP are excluded from regular-season predictors.

### Candidate evidence retained

A simple exponential recency-weighting candidate selected a 12-week half-life on 2024. On the untouched 2025 holdout it improved MAE slightly but worsened both margin and total RMSE, so it failed closed and was not promoted.

The first six-feature linear PBP residual candidate also remains disabled after the scoring-prior promotion. Against the promoted baseline on the untouched 2025 holdout:

- margin MAE: baseline `10.607` vs adjusted `10.671`;
- margin RMSE: baseline `13.200` vs adjusted `13.265`;
- total MAE: baseline `10.484` vs adjusted `10.632`;
- total RMSE: baseline `13.303` vs adjusted `13.413`.

These negative results are retained as evidence: advanced features must earn their place on later data rather than being forced into the projection.

## Anti-leakage rule

For a target game in season `S`, week `W`, rolling team state must be created only from information known before that game. Target-week/future results and PBP are excluded from predictor construction. The prior-season scoring state is restricted to the immediately preceding completed regular season and is known before the active season begins.

Current injury/depth-chart data must never be backfilled into historical games unless point-in-time historical records prove it was known at the simulated decision time.

## Quick start

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
python run_stage1.py 2025 --refresh
python run_stage2.py 2025 10 --refresh
python run_walkforward.py 2025 --start-week 5 --end-week 10 --refresh
python run_recency_audit.py --refresh
python run_prior_audit.py --refresh
python run_residual_audit.py --refresh
```

The corresponding source and holdout audits also run in GitHub Actions.

## Key files

- `nfl/contracts.py` — schema and fail-closed data contracts.
- `nfl/data.py` — nflverse ingestion, caching and schedule anti-leakage helpers.
- `nfl/ratings.py` — canonical ridge fair-score baseline and validated prior-season weight.
- `nfl/advanced.py` — leak-free PBP efficiency features.
- `nfl/dataset.py` — chronological game-level modeling dataset construction.
- `nfl/evaluation.py` — baseline football-projection error metrics.
- `nfl/recency.py` — recency-weight research and holdout evaluation.
- `nfl/priors.py` — prior-season scoring-prior research and holdout evaluation.
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
