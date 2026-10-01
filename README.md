# Harbin NFL Analytics

Leakage-safe NFL projection, probability, market-analysis, backtesting, risk-management, and monitoring platform.

> **Current state:** Stage 3 — independent football projections, research probability distributions, point-in-time sportsbook comparison, historical-provider ingestion, CLV diagnostics, and chronological market-rule validation. The repository is research software. A successful workflow run is not evidence of a profitable betting edge.

## Design

The NFL platform follows the same separation-of-concerns philosophy as the CFB system while keeping NFL training data, features, thresholds, evidence, and release state fully independent.

1. **Data foundation** — schedules/results, play-by-play, team statistics, rosters, injuries, depth charts, caching, schema contracts, and anti-leakage primitives.
2. **Fair-score engine** — opponent-adjusted football ratings and NFL-specific dynamic features. Sportsbook prices do not enter the football score projection.
3. **Probability + market layer** — score distributions, win/cover/total probabilities, no-vig market comparison, point-in-time price provenance, line movement, CLV, and chronological market-rule evidence.
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

A second PBP experiment explicitly adjusted each efficiency metric for schedule strength by fitting offense/defense ridge decompositions to historical offense-vs-defense game observations. Validation selected the compact EPA + success-rate feature set, with separate margin and total signals, but it also failed on the untouched 2025 holdout:

- margin MAE: baseline `10.607` vs adjusted `10.640`;
- margin RMSE: baseline `13.200` vs adjusted `13.266`;
- total MAE: baseline `10.484` vs adjusted `10.507`;
- total RMSE: baseline `13.303` vs adjusted `13.313`.

These negative results are retained as evidence: advanced features must earn their place on later data rather than being forced into the projection.

### Stage 3 — probability and market layer

The first probability layer fits Gaussian residual distributions around independently generated margin and total projections. Distribution parameters come only from earlier chronological football residuals. Sportsbook lines are thresholds evaluated after fitting; they are not football-model inputs.

The first dispersion-scaling experiment selected scale `1.0` for both margin and total on validation and therefore produced no holdout improvement. On the untouched 2025 holdout of 208 games, the research Gaussian baseline recorded approximately:

- margin NLL: `4.00844`;
- total NLL: `4.00855`;
- home-win Brier score: `0.23405`;
- margin 50% / 80% interval coverage: `46.15% / 75.96%`;
- total 50% / 80% interval coverage: `54.81% / 79.81%`.

A direct logistic home-win calibration also failed the untouched 2025 holdout gate, finishing marginally worse than the Gaussian baseline on both Brier score and log loss. It remains disabled.

The sportsbook layer now supports:

- American-to-decimal and implied-probability conversion;
- proportional two-way vig removal;
- moneyline, spread, and total model probabilities;
- executable-price EV;
- model-vs-no-vig probability edge;
- complete two-way snapshot validation;
- explicit decision-time selection and quote-age limits;
- win/loss/push grading and realized net units;
- one best executable line-shopped opportunity per game/market for research backtests.

Historical quote rows preserve provider, sportsbook, snapshot, source event, capture time, and decision time. Opposite sides are never synthesized across unrelated snapshots.

### Historical odds provider and CLV

`nfl/odds_api.py` adds an optional The Odds API historical adapter for NFL moneyline, spread, and total snapshots. It requires `THE_ODDS_API_KEY`, uses timezone-aware historical decision timestamps, maps provider events to nflverse game IDs, preserves source provenance, and caches responses without putting the API key in cache paths or files.

`nfl/clv.py` compares a simulated decision quote with a later pre-kickoff quote from the same provider/book/market/side. It records no-vig probability CLV plus side-aware spread/total line CLV. Closing information remains evidence only and never feeds the football model.

`nfl/market_validation.py` selects probability-edge and EV thresholds on one validation season, freezes them, and scores a later holdout season once. A research candidate requires minimum holdout volume, positive units/ROI, and positive average probability CLV. Separate Stage 5 risk and release controls will still be required before any real staking path exists.

The provider adapter and validation machinery are tested, but this repository does not contain a paid historical provider credential. Therefore no historical ROI/CLV profitability claim is made from The Odds API data yet.

## Anti-leakage rule

For a target game in season `S`, week `W`, rolling team state must be created only from information known before that game. Target-week/future results and PBP are excluded from predictor construction. The prior-season scoring state is restricted to the immediately preceding completed regular season and is known before the active season begins.

Sportsbook history is also point-in-time. A simulated decision may use only complete provider snapshots captured on or before its explicit decision timestamp. Closing observations and later line moves may be used only for post-decision CLV evidence.

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
python run_oa_audit.py --refresh
python run_probability_audit.py --refresh
python run_win_probability_audit.py --refresh
python run_current_projection.py 2026 --refresh
```

The corresponding source and holdout audits also run in GitHub Actions. Historical provider retrieval is optional and requires `THE_ODDS_API_KEY`; it is intentionally excluded from ordinary CI so secrets and paid API usage are not required to test the core model.

## Key files

- `nfl/contracts.py` — schema and fail-closed data contracts.
- `nfl/data.py` — nflverse ingestion, caching and schedule anti-leakage helpers.
- `nfl/ratings.py` — canonical ridge fair-score baseline and validated prior-season weight.
- `nfl/advanced.py` — leak-free PBP efficiency features.
- `nfl/dataset.py` — chronological game-level modeling dataset construction.
- `nfl/evaluation.py` — baseline football-projection error metrics.
- `nfl/recency.py` — recency-weight research and holdout evaluation.
- `nfl/priors.py` — prior-season scoring-prior research and holdout evaluation.
- `nfl/residuals.py` — nested chronological raw-PBP residual-candidate validation.
- `nfl/opponent_adjusted.py` — schedule-adjusted PBP offense/defense decompositions.
- `nfl/oa_dataset.py` / `nfl/oa_residuals.py` — chronological opponent-adjusted residual evaluation.
- `nfl/probability.py` — research Gaussian score distributions and holdout calibration metrics.
- `nfl/win_probability.py` — direct home-win calibration experiment.
- `nfl/market.py` — odds math, no-vig probabilities, and football-model market comparison.
- `nfl/market_history.py` — reproducible point-in-time quote selection.
- `nfl/market_backtest.py` — market grading and research summaries.
- `nfl/odds_api.py` — optional historical The Odds API adapter.
- `nfl/clv.py` — closing-line value diagnostics.
- `nfl/market_validation.py` — line shopping plus validation/holdout threshold evaluation.
- `nfl/current.py` — current/upcoming independent projection pipeline.
- `nfl/stage1.py` / `nfl/stage2.py` — source integration audits.
- `nfl/walkforward.py` — week-by-week reconstruction audit.

## Non-negotiable model rules

- No sportsbook line may leak into the independent fair-score engine.
- No target-week/future results may enter a pregame feature.
- Missing data lowers confidence or blocks a path; it is not silently invented.
- Historical evaluation is chronological, not random train/test shuffling across time.
- A learned adjustment stays disabled if it fails later untouched data.
- A sportsbook backtest may only use quotes that were actually observable at the simulated decision time.
- Closing prices are evaluation evidence, not football-model features.
- Multiple books do not create multiple independent bets on the same modeled game/market in research summaries.
- A model reaches production only through independent evidence, not because code executes successfully.
- NFL thresholds, weights, calibration, and release evidence are independent of the CFB system.

See the Stage 1, Stage 2, and Stage 3 documents under `docs/` for the current contracts and validation design.
