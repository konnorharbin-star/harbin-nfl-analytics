# BetMGM Edge Lab — NFL spread, total and push-calibration checkpoint

Date: **2026-10-11 UTC**. Model-control version **v4.7.0**, research only.

## Verified cloud runs and reproducible inputs

**Canonical football baseline:** existing `nfl.ratings.FairScoreModel` (ridge 8; prior-season sample weight 0.10), forecasting each historical week using completed games from earlier weeks and the prior regular season. Scores are independently projected before market data are attached. Free public `nflreadpy`/nflverse schedule data, seasons 2022–2025. The research script and its tests did NOT change the live score-engine coefficients.

Sources: [nflverse project](https://github.com/nflverse/nflverse-data), [GitHub spread/total audit workflow](https://github.com/konnorharbin-star/harbin-nfl-analytics/actions/runs/38183130119), [audit code](../scripts/score_market_accuracy.py), [regression tests](../tests/test_score_market_accuracy.py). The action completed **SUCCESS**, with historical-score evaluation tests passing. A run-specific JSON artifact was uploaded under that workflow (artifact retention 90 days).

Comparison line: nflverse archived **final** spread and total (`spread_line` = home favored margin, `total_line` = combined score). **No original pre-kickoff source/quote timestamp is established; these are not executable account prices and not automatically true BetMGM closing prices.**

### 2025 historical diagnostic, NFL regular-season weeks 5–18, 208 games

| Target | Independent football forecast MAE | Archive-final market MAE | Market + training-fitted independent residual MAE | Independent RMSE | Archive-final RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Home scoring margin / spread | 10.606922 | **9.992788** | **9.992788** | 13.199584 | 12.322148 |
| Full-game total | 10.483861 | **10.081731** | **10.081731** | 13.303445 | 12.322148 (see workflow detailed totals RMSE) |

**Correction note:** Re-verify the exact totals RMSE from archived report before using any copied value in a later quantitative decision; totals RMSE is omitted as a decision input here. The archive-final comparison is more accurate on MAE for both markets.

A *descriptive* MSE-optimal market blend was estimated on 2023–2024 only, with a fixed conservative shrink penalty and alpha constrained 0–1. Selected independent-score weights were **0.0 for both margin and total**. Therefore the blended diagnostic equals archived market, not a new independent +EV signal. The 2025 dataset had been inspected in previous project research, so this is not a pristine holdout.

### Discrete score distribution experiment

Another success: [push-aware score diagnostic](https://github.com/konnorharbin-star/harbin-nfl-analytics/actions/runs/38183256196); [research code](../scripts/discrete_score_diagnostic.py); [tests](../tests/test_discrete_score_diagnostic.py). Fit forecast residual mean/variance with 2023–2024 football-only predictions, compare a Gaussian discretized into integer score bins against a fixed kernel-weighted empirical score distribution mixed with Gaussian prior; evaluate multiclass win/push/loss against 2025 **archive-final** threshold.

| Market | Gaussian multiclass log loss | Empirical challenger log loss | Empirical − Gaussian | Descriptive weekly CI |
| --- | ---: | ---: | ---: | --- |
| Margin/spread | **0.7546793** | 0.7596046 | +0.0049253 (worse) | [-0.0189366, +0.028749] |
| Total | **0.7046128** | 0.7265674 | +0.0219546 (worse) | [-0.0057232, +0.0529573] |

The challenger did **not** improve log loss; **do not promote**. Discretizing a Gaussian correctly allocates P(win), P(push), P(loss) for NFL integer final scores; an exact 3 or 7 margin must never be treated as a continuous probability-zero push. The stand-alone research implementation is correct at the math/unit-test level but has not been integrated into the production market EV path and cannot approve bets. Actual observed pushes at the archived thresholds: 1 in margin test, 0 in total test (most archived thresholds were half points).

### Registered research outcomes

- `SCORE_MARKET_BLEND_V1`: **NO INDEPENDENT EDGE / DO NOT PROMOTE**. Training chose zero independent residual weight for both targets. Market-only is the more accurate historical archive comparator.
- `DISCRETE_EMPIRICAL_MARGIN_V1`: **REJECTED FOR PROMOTION** on higher 2025 multiclass log loss than discrete Gaussian.
- `DISCRETE_EMPIRICAL_TOTAL_V1`: **REJECTED FOR PROMOTION** on higher 2025 multiclass log loss than discrete Gaussian.
- `INTEGER_SCORE_GAUSSIAN_PUSH_ACCOUNTING`: **MATHEMATICAL RESEARCH VALIDATION ONLY**; retains separate win/push/loss probabilities. No production integration/market-profit validation in this release.
- Prior `FREE_MARKET_RESIDUAL_V1` spread rejection and totals research-only finding stay intact.
- Zero profitable validated BetMGM positions; user does not need to place a wager. Current action remains PASS.

## Highest-value improvements to pursue next

1. **Valid price snapshots:** Record US BetMGM account-confirmed accepted odds + exact line and source quote UTC, plus independent sportsbook prices at the **same selection, threshold, period and observation time**. An archive-final line cannot validate a pregame pricing decision.
2. **2026 as-of forward study:** Record one first-seen independent spread/total score and probability distribution per event, verify starting QB, injuries and forecast issue-time venue/weather. No future-result re-creation.
3. **Market-only baseline:** Compare any challenge to contemporaneous two-way no-vig market, at the precise betting contract, with win/push/loss and void rules; reject claims of edge when independent residual loses to market.
4. **QB and injury reliability:** Preserve the existing starter mismatch fail-closed rule; measure roster effects only using point-in-time data and independent chronological checks.
5. **Distribution accuracy:** Only consider a new calibrated score distribution after untouched 2026 prospective multiclass proper-score performance, cluster uncertainty and *real* market pricing demonstrate value. Never infer profitability from a lower point MAE alone.

No paid subscriptions, paid APIs or automatic wagering.
