# Score-error decomposition — research only

This tool audits **archived historical predictions versus settled scores**, independently of bookmaker odds, suggestions or EA ratings. Its outputs are descriptive, not a fitted adjustment.

The data contain settled scores but **do not establish immutable pregame capture timestamps**. Consequently, this report must be labelled \`RETROSPECTIVE_HISTORICAL_REPORT_NOT_VERIFIED_FORWARD\`; it cannot prove out-of-sample or prospective edge.

## Calculation

Given projected home margin M and total T:

- Projected home points = (T + M)/2
- Projected away points = (T − M)/2
- Actual home points = (actual total + actual home margin)/2
- Actual away points = (actual total − actual home margin)/2

Signed error is **projected minus actual**. Each report shows per-game and aggregate margin, total and individual-team errors. Games with total errors of at least 20/30 points are counted as tails. Fixed bands based on *projected* totals and projected absolute margins let us compare error regimes without grouping by outcomes.

\`TOTAL_DOMINANT\` and \`MARGIN_DOMINANT\` mean only that one mathematical error is larger; they do not prove pacing, coaching, quarterback injuries or defensive breakdowns caused it.

## Run (standard library only)

NFL:
\`\`\`bash
python scripts/score_error_diagnostics.py --sport nfl \
  --source reports/free_market_predictions.csv \
  --json-output reports/score_error_diagnostics.json \
  --tails-output reports/score_error_extreme_games.csv
\`\`\`

CFB:
\`\`\`bash
python scripts/score_error_diagnostics.py --sport cfb \
  --source reports/backtest_predictions.csv \
  --json-output reports/score_error_diagnostics.json \
  --tails-output reports/score_error_extreme_games.csv
\`\`\`

Run \`python -m pytest -q tests/test_score_error_diagnostics.py\` to test the accounting. The companion GitHub Actions workflow uploads JSON and extreme-game CSV as research artifacts (no generated-state push; no production scoring change). This workflow must not publish fresh betting recommendations.

## Prioritized next tests

1. Look for repeatable total-error regimes conditioned on **pregame** projected total, uncertainty, pace indicators and available team efficiency.
2. Link **properly timestamped** QB/injury/OL changes to each pregame matchup. Missing starter confirmations must remain unknown.
3. Replay on chronological untouched weeks with source snapshots frozen before kickoff. Compare margin/total MAE and RMSE, probabilistic calibration, game-by-game prediction changes, and tail loss.
4. Promote a change only after repeated holdout/forward improvements and no deterioration in pricing or risk gates. Do not retroactively change Cowboys–Buccaneers projections or prior decisions.

No historical prediction row is re-trained by this script. The production model's fair scores and stakes remain unchanged.
