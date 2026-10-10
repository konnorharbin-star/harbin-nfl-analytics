# Walters-inspired Key Number / Push Probability Research (October 2026)

## Why this is a distinct research question

Billy Walters-style handicapping starts with an independent game number and
then assesses whether the available betting line is worth taking. For football,
the market price comparison has a settlement problem when the spread is an
integer. An NFL team winning by precisely 3 or 7 produces a push at several
common handicaps, but not at neighboring half-points. The existing
\`nfl.probability.GaussianScoreDistribution\` is continuous, and therefore
gives an integer spread **zero exact push mass**. An economic decision needs
three outcome probabilities (win, push, loss), because
\`EV = P(win) * profit_at_odds - P(loss)\`; pushes return stake.

This **research-only** code does not modify the live probability engine,
approved EV, stake sizes, or any sportsbook recommendation. Nor does this
study claim access to Billy Walters' proprietary original power ratings.

## Frozen research design

Use the current independent margin forecasts and results from the checked-in
historical files. **Do not** use sportsbook lines, odds, game results after
the target season, player injuries observed late, or 2025 outcomes to tune.

- Build a normal residual distribution using an earlier historical season
  training sample.
- Discretize predicted final **home-margin** into integer outcomes from -100
  to +100 with normal continuity corrections. This provides a legitimate
  probability of landing on **every integer**, including a 3/7 push.
- Compare the empirical **absolute victory margin** frequencies of 3 and 7
  against the expected Gaussian frequencies **on the training data only**.
  Build symmetric home/away multipliers with 120 equivalent pseudo-games of
  shrinkage and cap them to [0.60, 2.20]. This avoids pretending a directional
  home edge exists merely because 3 and 7 are common NFL final margins.
- Freeze candidate multiplier strength at one of \`0, 0.5, 1.0\`. Zero means
  **no correction**. Pick nonzero only if it improves **both** three-outcome
  cross-entropy and Brier score on a separate 2024 tuning block. This is
  not selecting a profitable bet threshold.
- Refit distribution parameters on all years strictly before 2025 with the
  frozen strength, then score 2025. **Never choose parameters from 2025**.
- Assess outcomes at fixed synthetic home spread lines
  \`[-7, -6.5, -3, -2.5, 0, 2.5, 3, 6.5, 7]\`. Each fixed line is evaluated
  against every game's true final margin. This is a *forecast-distribution*
  diagnostic, **not an archived sportsbook backtest**.
- Resample complete season/week clusters in paired bootstrap intervals and
  correct conservatively for the two sports and two probability metrics.

NFL chronology: selection fit 2022-23; tune 2024; refit 2022-24; evaluate
2025. CFB is fit 2023; tune 2024; refit 2023-24; evaluate 2025. CFB must
learn independent key-number multipliers; NFL evidence may not set its values.

## Limits and promotion requirements

The 2025 retrospective observations have already been examined in older
research stages; they are **not a pristine holdout**. There is no proof
archived forecast snapshots were originally published before their kickoffs.
Reference handicaps are *synthetic*, not executable bookmaker prices.
Repeated reference handicaps for one game are correlated; week-block
bootstrap addresses dependence partially, not strategy selection bias.
Neither Brier score nor cross-entropy alone demonstrates positive ROI or CLV.

The only allowed output of this workflow is a research report. Even an
improved 2025 diagnostic requires a frozen **forward** specification with
verified point-in-time projections and real independently verified book
prices, actual hypothetical settlements, and prospective evidence. A zero
adjustment / no-bet outcome is expected if development evidence is weak.

Reproduce:
\`\`\`sh
python -m pytest -q tests/test_walters_key_numbers.py
python scripts/walters_key_number_distribution.py --sport nfl \
  --source reports/free_market_predictions.csv \
  --out reports/walters_key_number_distribution.json
# or replace --sport cfb and use reports/backtest_predictions.csv
\`\`\`

GitHub Actions workflow \`Walters Key Number Research\` runs the isolated
tests and grades the archived real-source games. It only uploads an artifact;
it never alters the production engine or submits wagers.


## Historical outcome — evaluated October 10, 2026

The isolated research workflows executed successfully on the repo's **real
archived independent forecasts**, not only synthetic test fixtures.
Full results are attached as GitHub Actions research artifacts.

| Sport | 2025 evaluation games | Strength selected on 2024 | Baseline three-way log loss | Adjusted | Baseline Brier | Adjusted |
|---|---:|---:|---:|---:|---:|---:|
| NFL | 272 | 1.0 | 0.695499 | 0.687508 | 0.442958 | 0.441499 |
| CFB | 808 | 1.0 | 0.594281 | 0.588069 | 0.368445 | 0.368506 |

The NFL training-selected *refit* multipliers for the combined absolute
margins were **3: 2.20** (at the protective cap) and **7: 1.4063**.
CFB's refit absolute-margin multipliers were **3: 2.20** and
**7: 2.20**, both at the cap. These are **distribution likelihood
multipliers, not point-spread adjustments, expected profits or winning
probabilities**.

Week-paired conservative simultaneous 95% bootstrap intervals for
*baseline loss minus challenger loss*:

| Sport | Log-loss improvement 95% CI | Brier improvement 95% CI | Required both >0? |
|---|---|---|---|
| NFL | [0.001363, 0.014523] | [-0.001488, 0.004646] | **No** |
| CFB | [0.003694, 0.009116] | [-0.001164, 0.001246] | **No** |

The log-loss result supports an *exploratory scoring-distribution
hypothesis*, but the Brier intervals include zero and CFB's Brier
point estimate slightly worsened. Therefore **neither sport passed the
prespecified two-endpoint statistical screen**. This is **not** an
investment/betting-edge finding. No scores or live probabilities were
altered, no candidate was promoted, no bets were made. The 2025
data are retrospectively inspected, not untouched validation.
