# Stage 1 — NFL Data Foundation

Stage 1 establishes the source-data and anti-leakage contract for the NFL platform. It does **not** claim a betting edge and does not use sportsbook prices to create football ratings.

## Primary source

The initial free data stack uses `nflreadpy`, the Python loader for nflverse data. The platform currently wraps:

- schedules/results
- play-by-play
- weekly team statistics
- weekly rosters
- injuries
- depth charts

Raw source columns are preserved. The model only assumes the minimum schema declared in `nfl/contracts.py`.

## Non-negotiable leakage rule

For a target game in season `S`, week `W`, any rolling football state must be calculated from information known before that game's kickoff. Stage 1 provides `pregame_history()` and `assert_strictly_pregame()` so Stage 2 ratings cannot silently include target-week or future results.

Current-season injury/depth-chart information is **not** allowed to be backfilled into old games. Historical personnel adjustments will require timestamped historical records and their own Stage 4 validation.

## Cache behavior

`NFLDataClient` stores downloaded frames as parquet under `data/cache/`. The cache is intentionally local and is ignored by git. Use `refresh=True` (or `--refresh` in the audit CLI) when a fresh upstream pull is required.

## Stage 1 audit

Run locally:

```bash
python -m pip install -e ".[dev]"
python run_stage1.py 2025 --refresh
```

Or use GitHub Actions → **Stage 1 Data Audit** and supply a season.

The audit verifies:

1. required schedule and PBP fields exist;
2. schedule game IDs are unique;
3. completed games are identifiable from final scores;
4. completed schedule games expand to exactly two team-game rows;
5. play-by-play data contains game identifiers and EPA fields needed for Stage 2 feature engineering.

## Exit criteria for Stage 1

Stage 1 is complete only when:

- CI passes;
- the manual data audit succeeds for multiple historical seasons and the current season;
- schedule/PBP schemas satisfy the contracts;
- pregame leakage tests pass;
- source/cache behavior is deterministic enough for reproducible backtests.

Only then should Stage 2 add opponent-adjusted team ratings and fair-score projections.
