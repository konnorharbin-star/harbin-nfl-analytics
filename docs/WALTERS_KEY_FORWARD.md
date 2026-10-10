# Walters key-number prospective 2026 shadow

This continues the publicly documented Walters-inspired sequence: independent fair
score first, calibrated discrete outcome likelihoods second, sportsbook pricing
comparison only after all football forecasts are frozen. This is **not** a
reproduction of Billy Walters' private proprietary coefficients.

## Precommitted challenger

The earlier historical key-number study selected **alpha = 1.0** using 2024
tuning data, separately in NFL and CFB. Its 2025 retrospective results were
reviewed, but did not clear the full Brier/log-loss uncertainty screen.
**Do not refit alpha against any 2025 or 2026 results.** The prospective
experiment freezes alpha at exactly 1.0 and compares it with the discrete
Gaussian baseline (alpha 0.0). Both use the same 3/7 excess-margin
multipliers fitted using the earlier complete seasons:

- NFL: 2022–2025; CFB: 2023–2025.
- Never train on the 2026 target-season outcome.
- Keep the raw independent \`model_margin_home\` from \`docs/latest.csv\`.
  NFL kickoff field is \`kickoff\`, CFB is \`date\`. Exact board file bytes and
  training CSV are SHA256-hashed.
- Only 2026 games more than 5 minutes and no more than 7 days away are
  capturable. This guards accidental after-kickoff "predictions."
- A separate first-seen JSON record is created **once per game** at
  \`history/walters_key_forward_v1/forecasts/<game_id>.json\`; never changed.
  Every record freezes unrounded baseline/challenger WIN/PUSH/LOSS likelihoods
  at nine prespecified fixed hypothetical spreads. Neither selected odds,
  consensus prices nor other sportsbook variables are inputs.
- An audit accepts a record for grading **only after** the unchanged original
  file bytes can be traced to exactly one *Git addition commit* timestamped
  between the capture and original kickoff. Git commit timestamps are useful
  integrity checks but not independently certified bookmaker source times.

The scheduled GitHub Actions workflow uses existing free forecast and
historical CSV files. It runs tests, captures future games, grades eligible
completed games against a separately retrieved **ESPN public final
scoreboard**, uploads the report artifact, and commits generated data when
running on \`main\`. Pull requests only test and upload research output.
A concurrent main-branch push conflict is allowed to fail rather than force
overwriting earlier immutable forecasts. GitHub scheduled runs are best
effort and can be delayed, so **missing first captures stay missing**.

The grading report at \`reports/walters_key_forward.json\` stores an audited
source URL and SHA256 of the fetched final-score payload, the pregame Git
publication time, and paired proper scoring rules. No games are graded from
a local reconstructed "historical" result. Prior to 128 graded games across
at least eight distinct kickoff weeks, label evidence
\`FORWARD_INSUFFICIENT_SAMPLE\`. At larger sample sizes provide conservative
week-clustered simultaneous confidence intervals. This **does not** itself
authorize promotion: the earlier 2025 work already used these data.

## What it does not do

This is **not** a sportsbook pricing backtest, is not a claim of verified EV,
CLV, realized ROI, profitability or a warranted stake. Diagnostic spreads
are fixed reference thresholds, not fake book offers. No live betting
recommendation, user-executable sportsbook quote or automatic wager is created.
Neither the production model probabilities nor its current betting safeguards
are touched. Separate book-verified pregame offers are still required before
a market-edge hypothesis is actionable.

To reproduce on a checkout with the expected repo input files:

\`\`\`bash
python -m pytest -q tests/test_walters_key_forward.py
python -m scripts.walters_key_forward --sport nfl \
  --historical reports/free_market_predictions.csv
# College football: --sport cfb \
#   --historical reports/backtest_predictions.csv
\`\`\`
