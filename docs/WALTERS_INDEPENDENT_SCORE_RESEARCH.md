# Walters-inspired independent score calibration — research v1

This experiment continues the publicly described Billy Walters process of making
**an independent fair score first**, before looking at sportsbooks. Public descriptions
of Walters's process emphasize neutral-field team power ratings, game-specific
adjustments, player values/injury impact, home field, travel, weather, and betting
only if the resulting line differs sufficiently from available odds. The private
historical models and exact coefficients are not known.

## Frozen exploratory experiment

Source: tracked historical **score predictions**, not lines or implied sportsbook
probabilities. The experiment targets home margin and total independently.

For an existing independent forecast `x`, fit residual `e = actual - x` on
training data. With training mean `mu` and standard deviation `sd`, define
`z = (x - mu) / sd` and proposed prediction:

`corrected = x + alpha + beta * z`

Ridge shrinkage towards **no adjustment**:
- `alpha = sum(e) / (n + lambda)`
- `beta = sum(z*e) / (sum(z*z) + lambda)` for affine correction
- for the intercept-only family, `beta = 0`
- `lambda in {25,100,400}`; predeclared two families, six total candidates

Train on seasons strictly before **2024**, choose exactly once on **2024** using
minimum tuning RMSE *only among candidates that also improve both tuning MAE
and RMSE over the uncorrected baseline*. Default to **zero correction** otherwise.
Evaluate once on **2025**. No test-season outcomes enter fitting or selection.
Each sport has entirely independent coefficients. The script rejects short and
poorly distributed chronological samples, malformed numeric scores and duplicate IDs.

Run in NFL repo:

```bash
python scripts/walters_score_calibration.py --sport nfl --source reports/free_market_predictions.csv --out reports/walters_score_calibration.json
```

Run in CFB repo:

```bash
python scripts/walters_score_calibration.py --sport cfb --source reports/backtest_predictions.csv --out reports/walters_score_calibration.json
```

Or dispatch the manual, artifact-only Walters Score Research workflow.

## Limitations and release status

This is **retrospective diagnostic research**, not an untouched or true prospective
backtest: the source predictions were not established as immutable pregame
observations, and 2025 archive results have already been inspected during previous
experiments. Earlier seasons may also have been involved in previous model
development. This research **does not** test executable entry lines, real offered
odds, CLV, ROI, calibrated cover probabilities or profitable wagering.

Coefficients MUST NOT be promoted automatically. Next: obtain or create
timestamped, frozen independent score forecasts; measure margin and total MAE/RMSE
in new games; separately obtain genuine, time-matched bookmaker price evidence.
Only then test a predeclared EV threshold against a no-vig baseline and publish
non-executing informational recommendations with immutable paper-bet receipts.
Nothing here places a bet, changes stake sizing, or weakens existing blockers.

References: Billy Walters official handicapping guide:
https://realbillywalters.com/handicapping-system/ and his publisher's
description of `Gambler`: https://www.simonandschuster.com/books/Gambler/Billy-Walters/9781668032862

The emitted report also includes paired season/week bootstrap uncertainty across
2025 games, with familywise 95% Bonferroni intervals for eight predetermined
sport/target/metric comparisons. An apparent MAE or RMSE gain whose interval
crosses zero is **not** evidence of statistically supported improvement.
