# Harbin NFL Analytics — Agent Rules

## Non-negotiable modeling rules
- Never use future information in a historical feature. A game's features must be knowable before kickoff.
- Use chronological or season/week walk-forward validation. Never use a random split for a time-series performance claim.
- Sportsbook prices do not enter the core fair-score projection. Market prices may be used only after the fair margin/total is produced, or as an explicit risk/diagnostic layer.
- Current injuries, weather, depth-chart news, and other current-only context must not be backfilled into historical training unless a timestamp-correct historical source exists.
- Preserve unrounded model values internally. Round only in presentation.
- Missing data must remain missing or trigger a documented fallback. Never invent a sportsbook quote, player status, weather value, or feature.
- NFL weights, thresholds, calibrators, release evidence, and context adjustments must be validated independently from the CFB model.
- Do not claim profitability from MAE alone, a short record, or an in-sample fit.
- A backtest must state which line stage it uses and must not call a proxy an official closing line.

## Model promotion rules
- Any learned residual layer must be eligible for weight 0 if it loses to the baseline on chronological tuning data.
- Keep an untouched chronological evaluation block after hyperparameter/weight selection.
- Report MAE/RMSE for margins/totals and Brier/log-loss/ECE for probabilities.
- Report market results by market, signal, and edge bucket with bets, win rate, ROI, units, max drawdown, and confidence interval.
- Prefer positive CLV over short-term win rate when judging whether a pricing edge is plausible.
- Stake sizing must be capped and risk-adjusted; no uncapped Kelly.
- Most games should be PASS when estimated EV is weak or data quality is poor.

## Engineering rules
- Run `python -m pytest -q` before merging.
- Keep the production workflow green.
- Do not commit secrets, virtualenvs, caches, or binary model artifacts.
- Data-source failures must appear in metadata/health output instead of being hidden.
- Keep the NFL operational shell structurally aligned with Harbin Sports Analytics CFB where league-specific differences do not require divergence.
