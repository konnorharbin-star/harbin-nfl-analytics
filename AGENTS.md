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

## v1.0 build freeze
- NFL Model v1.0 is complete. Default work after 2026-10-02 is maintenance and evidence accumulation, not continued model expansion.
- Do not create a new numbered modeling/research stage unless the user explicitly asks to reopen model development.
- Evidence refreshes, forward grading, release-state transitions, bug fixes, and broken-source repairs do not count as new model stages.
- Do not weaken historical, policy, CLV, entry-integrity, or forward-sample release gates to force PAPER -> SHADOW -> PRODUCTION.

## Owner operating requirements — 2026-10-08
- **Never place bets automatically.** The platform may research, rank, notify, display, and grade suggested bets, but cannot submit orders, make deposits, connect wagering accounts for execution, or auto-fund or auto-stake anything. Human placement is always outside this software.
- **Remain free to operate.** Use open-source libraries, public/free legally accessible data and standard included GitHub resources. Do not add a paid odds API, paid database, paid hosting, new subscription, or a service with metered charges. If a free tool has quotas, fail closed or reduce collection frequency rather than incurring costs.
- Model `stake_units` or `approved_units` fields, if retained for research, are internal estimates only and never instructions to a sportsbook. A change in model release state does not authorize order execution.
- Recommendations are informational, not evidence of proven profitability. No unvalidated raw EV signal may be promoted to a confirmed market edge without point-in-time and out-of-sample evidence.
- These operating requirements apply to every future sport, workflow, dashboard, integration and AI coding agent working in the project.

## Owner research direction — 2026-10-10: Walters-inspired process
- Make the documented Billy Walters handicapping workflow the organizing research method; never claim access to his unpublished proprietary model or original coefficients.
- Start each game with an independent team strength/power-rating and fair-score forecast. Quantify pregame roster/QB availability, matchup, travel, rest, home-field and weather only when point-in-time evidence exists. Missing inputs remain unknown.
- Fit sport-specific weights and formulas using past-only training; select regularization and candidate families on a separate chronological tuning block; evaluate after selections are frozen. Never choose parameters, filters or edge thresholds by inspecting the evaluation outcome. Historical inspected archives are diagnostics, not pristine proof.
- Challenge every candidate against a no-vig, contemporaneous sportsbook market reference and the existing independent baseline. Require verified executable entry prices, timestamp provenance, reasonable data coverage, multiple-testing controls and independent forward paper results before claiming an edge.
- Keep a mathematically valid zero-adjustment / NO BET option. A backtest cannot be optimized merely to manufacture a positive return.
- Log each future published qualified bet recommendation as a prospective immutable paper-bet receipt, regardless of outcome, and grade ROI, CLV, calibration and drawdown; never auto-execute.
- Continue the free-only and no-automatic-betting rules above. Score-calibration research does not bypass release gates or promote betting probabilities.
