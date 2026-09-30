# Harbin NFL Analytics

Leakage-safe NFL projection, probability, market-analysis, backtesting, risk-management, and monitoring platform.

> **Current state:** Stage 1 — data foundation. The repository is research software. A successful workflow run is not evidence of a profitable betting edge.

## Design

The NFL platform follows the same separation-of-concerns philosophy as the CFB system while keeping NFL training data, features, thresholds, evidence, and release state fully independent.

1. **Data foundation** — schedules/results, play-by-play, team statistics, rosters, injuries, depth charts, caching, schema contracts, and anti-leakage primitives.
2. **Fair-score engine** — opponent-adjusted football ratings and NFL-specific dynamic features. Sportsbook prices do not enter the football score projection.
3. **Probability + market layer** — score distributions, calibrated win/cover/total probabilities, no-vig market comparison, executable-price checks, line movement, and CLV.
4. **Context layer** — quarterback state, personnel/injuries, offensive-line availability, rest, travel, stadium/roof, and weather.
5. **Risk + execution layer** — fractional Kelly, concentration caps, drawdown throttles, price provenance, and fail-closed stake approval.
6. **Proof + release layer** — chronological walk-forward backtests, calibration, ROI/CLV, max drawdown, confidence intervals, shadow/live grading, monitoring, and RESEARCH/PAPER/SHADOW/PRODUCTION gates.

## Stage 1 source stack

The initial free data layer uses `nflreadpy`/nflverse for schedules, play-by-play, team statistics, rosters, injuries, and depth charts. Source frames are cached locally as parquet and validated against minimum platform contracts.

### Anti-leakage rule

For a target game in season `S`, week `W`, rolling team state must be created only from information known before that game. `nfl.data.pregame_history()` and `nfl.data.assert_strictly_pregame()` enforce the first version of that rule.

Current injury/depth-chart data must never be backfilled into historical games unless point-in-time historical records prove it was known at the simulated decision time.

## Quick start

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
python run_stage1.py 2025 --refresh
```

The Stage 1 audit can also be run from **Actions → Stage 1 Data Audit**.

## Current repository layout

```text
.github/workflows/
  ci.yml
  stage1-audit.yml

docs/
  STAGE1_DATA_FOUNDATION.md

nfl/
  __init__.py
  contracts.py
  data.py
  stage1.py

tests/
  test_contracts.py
  test_leakage.py

run_stage1.py
pyproject.toml
```

## Non-negotiable model rules

- No sportsbook line may leak into the independent fair-score engine.
- No target-week/future results may enter a pregame feature.
- Missing data lowers confidence or blocks a path; it is not silently invented.
- Historical evaluation is chronological, not random train/test shuffling across time.
- A model reaches production only through independent evidence, not because code executes successfully.
- NFL thresholds, weights, calibration, and release evidence are independent of the CFB system.

See `docs/STAGE1_DATA_FOUNDATION.md` for the Stage 1 contract and exit criteria.
