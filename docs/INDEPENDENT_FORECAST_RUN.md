# Run our independent NFL predictions

In GitHub Actions, choose **NFL MODEL — Our Scores, Spreads, Totals and Player Props**, then **Run workflow**. The existing daily and Sunday schedules also publish the same bundle. Other Actions entries are research/validation helpers, not the model forecast command.

Every successful forecast run produces:

| File | Contents |
|---|---|
| `independent_forecasts.png` | One-row-per-game score, home spread and total board |
| `independent_forecasts.html` | Score board followed by named-player stat tables for every game |
| `independent_forecasts.json` | Unrounded numbers, raw distributions, source provenance and limitations |
| `independent_game_predictions.csv` | Own game scores, spread and total |
| `independent_player_predictions.csv` | Own player means, medians, p10 and p90 by metric |
| `independent_forecasts.md` | GitHub run-summary game and player tables |

Download these from the model run artifact. Forecasts are generated before sportsbook-price capture. The projected home spread is the negative expected home margin; projected total is the sum of the two expected scores. Player and game expectations come from the same coupled bootstrap draws. Publication verifies score/spread/total reconciliation and that player receiving and rushing production sum to team production, including explicitly unallocated production. It does not refit or overwrite original model forecasts.

Supported player stats: passing attempts, completions, yards, touchdowns and interceptions; carries, rushing yards and touchdowns; receptions, receiving yards and touchdowns. Unsupported kicking, longest-play, defensive and first-scorer markets are listed as unsupported, not guessed. Historical player roles are not confirmed starters/actives, and a zero simulated workload is not a settled DNP or void. Player names are display labels joined by GSIS ID; name-source failure leaves the ID visible and logs the failure.

Raw p10/p90 and win/tie fractions are conditional simulation outputs, not calibrated prediction intervals or verified final NFL/OT market probabilities. Overtime and unusual scoring remain unvalidated. The independent power-rating score baseline is also retained separately. Predictions do not authorize wagers, and greater accuracy than market forecasts is **UNPROVEN**.

Run the independent model locally:

```bash
python run_nfl_predictions.py 2026
python run_nfl_predictions.py 2026 --week 6
```

The existing full operational command now publishes the independent bundle first, before its market audit and portfolio controls:

```bash
python harbin_nfl_model.py 2026 --refresh
```

Outputs remain available if a later sportsbook audit fails. Forecast source failures block the affected game instead of inventing zeros. Scores, spread and total remain unrounded in JSON/CSV; only the human display is rounded.
