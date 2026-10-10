# Walters-style opponent-aware scoring residual — research only

This research compares the **independent projected score** with a conservative
correction based on recent, historically observed offensive and defensive scoring
residuals. It is inspired by the discipline of opponent adjustment; it is **not**
Billy Walters' proprietary model, does not reproduce private coefficients, and
does not place bets.

For each team in a season, store results of **completed earlier weeks**:
- Offensive residual = actual points scored minus projected points scored.
- Defensive residual = actual opponent points minus projected opponent points.
- Shrunk component = sum of last N residuals / (number observed + 4).
- Home points correction = (home offense residual + away defense residual) / 2.
- Away points correction = (away offense residual + home defense residual) / 2.
- Total correction = sum of the two; margin correction = home minus away.
- Proposed score = independent projection + (weight × relevant correction).

The grid is frozen at lookbacks N = 3, 6, 10 and weights = .25, .50, .75, 1.00.
Zero adjustment is also considered. Selection uses **2024 games alone**, requiring
lower margin/total MAE **and** RMSE on that season. The winning tuning variant,
if any, is evaluated once on 2025; the 2025 outcomes do not set parameters.
Historical 2022–2023 development games establish available chronology, not
additional selected weights. NFL and CFB are tested separately.

Every game's features are frozen **before any outcomes in its own week** are
ingested, even if CSV rows arrive unsorted. Roster turnover is handled
conservatively by discarding the prior season's team-level residual state.
The tests reject evaluation-driven selection and target-week result leakage.
Scoring excludes bookmaker odds entirely.

## Major limitations

These archived predictions were not independently verified as immutable
pre-kickoff forecasts and the 2025 seasons were already viewed in earlier
research. The 2025 evaluation is **retrospective exploratory**, not a pristine
holdout. The corrections do not have independent injury, starter, temperature,
pace, or schedule-quality evidence. A paired week-bootstrap with simultaneous
Bonferroni bounds illustrates uncertainty only. Neither a lower MAE nor positive
interval proves that an executable market edge exists; no ROI or CLV is inferred.
The model must NOT apply these parameters to production predictions and must
NOT turn the diagnostics into suggested or logged wagers.

## Reproduce

Run \`python -m pytest -q tests/test_walters_matchup_residual.py\` and
\`python scripts/walters_matchup_residual.py --sport SPORT --source SOURCE --out REPORT\`.

For NFL use \`SPORT=nfl\`, \`SOURCE=reports/free_market_predictions.csv\`.
For CFB use \`SPORT=cfb\`, \`SOURCE=reports/backtest_predictions.csv\`.
The manual **Walters Matchup Residual Research** Actions workflow uploads only
a JSON artifact; it cannot modify production scoring or execution policies.

If a repeated prospective advantage later emerges, independently verify entry
lines, same-market settlement, genuine quote-origin timestamps, source
freshness, user-accessible prices, closing-line value and economic results.

## Archived research outcome — October 10, 2026

The dedicated GitHub Actions research job completed successfully in both leagues.
These figures compare exactly the same 2025 games, with correction strength
and lookback chosen only from 2024 tuning scores.

| League | Target | 2025 games | Baseline MAE | Selected MAE | Baseline RMSE | Selected RMSE | Tuning selection |
|---|---|---:|---:|---:|---:|---:|---|
| NFL | Margin | 272 | 10.3724 | 10.3469 | 13.1202 | 13.0652 | prior 10 games, weight 0.25 |
| NFL | Total | 272 | 10.6990 | 10.6990 | 13.4717 | 13.4717 | zero correction |
| CFB | Margin | 808 | 12.4032 | 12.3997 | 15.8173 | 15.8278 | prior 3 games, weight 0.50 |
| CFB | Total | 808 | 12.8620 | 12.8620 | 15.9250 | 15.9250 | zero correction |

The NFL margin candidate improved by only 0.0255 MAE points. Its paired
familywise confidence bounds did **not** establish a positive improvement.
The CFB margin candidate worsened RMSE by roughly 0.0106 points. Neither
total market selected a nonzero correction. No adjusted score is promoted.

This run tested team-level prior-week **scoring residuals**, not independently
sourced verified QB injuries, lineup values, weather or possessions.
Market superiority, executable odds and profitability remain **unproven**.
The historical data were previously inspected: these results are descriptive,
not prospective validation. All model and betting release gates remain blocked.
