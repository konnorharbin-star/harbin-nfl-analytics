# BetMGM Edge Lab — usage × efficiency NFL player-prop research (v1)

**Research result: 2026-10-11 23:00:56 UTC.** No paper wager, BetMGM offered odds, player prop fair price, or estimated positive EV. The active/canonical forecasts are NOT upgraded.

## Verified real-data replay

Free-source workflow: [successful 2024→2025 evaluation](https://github.com/konnorharbin-star/harbin-nfl-analytics/actions/runs/38182620487). Research artifact: [download from that workflow](https://github.com/konnorharbin-star/harbin-nfl-analytics/actions/runs/38182620487/artifacts/11718388396) (GitHub artifacts expire). Code: [scripts/usage_prop_research.py](../scripts/usage_prop_research.py), time-cutoff tests: [tests/test_usage_prop_research.py](../tests/test_usage_prop_research.py), cloud workflow: [edge-lab-usage-props.yml](../.github/workflows/edge-lab-usage-props.yml).

- Training/development: all eligible completed 2024 regular-season games, selecting **ONLY** the volume lookback from {3,5,8} on historical MAE.
- Diagnostic: eligible completed 2025 regular-season games. This season was ALREADY examined by previous project work. It is not an untouched holdout.
- Same-cohort benchmark: existing frozen `nfl.prop_challenger` mean forecast and rolling-three baseline on identical matched player-game rows. Player cohort eligibility determined solely from previous games. Missing postgame player rows scored as zero, but actual inactive/void status is not independently verified.
- Free historical data: nflverse player_stats and schedules through existing `NFLDataClient`. All features use earlier **weeks of the current season**, no target-week player outcomes or current market lines. Historical upstream data can be revised retrospectively; data-vintage availability at original kickoff is not verified.
- Normalized selected player-rows SHA256: `5a47bddbab1d7db070d2c09ba47b5bc6deda3db317247b29f9e670c10298f838` (NOT the raw nflverse archive checksum).
- Script version: `usage_efficiency_prop_research_v1`; run source Git SHA `374aa92dca5f66987d971d82d3f2bd05c729a13f`; run report captured `2026-10-11T23:00:56.289562+00:00`.
- 2024 selected lookbacks: receiving yards 8, receptions 8, rushing yards 5, passing yards 3. Fixed pseudo-opportunity efficiency priors were pre-specified in research code, not learned from 2025.

| Market | 2025 eligible | Same-cohort scored | Existing MAE | Usage MAE | Usage − existing MAE | Descriptive week bootstrap 95% difference |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Receiving yards | 3,996 | 3,941 | 18.603330 | **18.113386** | -0.489944 | [-0.587988,-0.372824] |
| Receptions | 4,036 | 3,980 | 1.430887 | **1.422352** | -0.008535 | [-0.012744,-0.004367] |
| Rushing yards | 2,121 | 2,108 | 15.705318 | **15.546624** | -0.158694 | [-0.339497,+0.019205] |
| Passing yards | 521 | 518 | **81.359717** | 81.456624 | +0.096907 | [-0.719750,+1.052369] |

2025 usage RMSE: receiving yards 25.326577 (baseline 25.965133); receptions 1.879958 (baseline 1.889322); rushing yards 24.760367 (baseline 24.884804); passing yards 102.154078 (baseline 102.678948). Passing-yard MAE nevertheless worsened.

**Decision registry:**
- `USAGE_RECEIVING_YARDS_V1`: RETAIN AS RESEARCH CHALLENGER; potentially useful signal, NOT production or a +EV model.
- `USAGE_RECEPTIONS_V1`: RETAIN AS RESEARCH CHALLENGER; small historical improvement, NOT production or a +EV model.
- `USAGE_RUSHING_YARDS_V1`: RESEARCH ONLY / INCONCLUSIVE; uncertainty interval includes zero.
- `USAGE_PASSING_YARDS_V1`: DO NOT PROMOTE based on MAE, while RMSE descriptive improvement is insufficient.

Important: Week-bootstrap intervals are descriptive and use roughly fifteen target-week clusters, with correlated players sharing games. They are not causal confidence intervals, pristine evidence, market-relative probability calibration, nor evidence of winning bets.

## How professional methods informed the challenger

Sean Koerner's public player-usage emphasis motivated separate opportunity and conditional efficiency modeling. For receiving yards, estimate **expected targets × smoothed catch probability × smoothed yards per catch**. For receptions use targets × catch probability; rushing yards use carries × yards per carry; passing yards use attempts × yards per attempt. The pseudo-priors are explicitly our experimental assumptions, **not** claimed to be Sean Koerner's private coefficients.

Documented source: [NFL player stats dictionary](https://nflreadr.nflverse.com/reference/load_player_stats); [nflverse public data releases](https://github.com/nflverse/nflverse-data/releases); [public-method registry](./PROFESSIONAL_METHOD_INTEGRATION.md).

## Required prospective validation and future code changes

1. Freeze this score-only specification without using inspected 2025 results to reselect the window/priors. Keep existing canonical prop model unchanged.
2. Freeze **first-seen 2026 pregame** usage-based forecasts before kickoff for *eligible future events*, with actual expected participation and injury data captured as-of time. Separate target/route opportunity uncertainty from conditional catch/yardage distributions.
3. Build properly calibrated **market-threshold** probabilities, not just means. Preserve push and void rules, and benchmark against exact player/book/threshold two-sided no-vig implied probabilities.
4. Compare log loss, Brier, calibration, proper count/yardage loss and conservative EV against contemporaneous account-confirmed BetMGM U.S. prices. Control selection, multiple testing and game/week correlations; archive decisions including PASS.
5. Maintain a strict fail-closed PASS for any incomplete price/participation/probability or validation field. Never place wagers, never invent line/odds, and never call this diagnostic a discovered edge.

**Bottom line:** This real-data run shows limited player-stat forecast improvements, not a verified sportsbook mispricing. Historical 2025 is no longer eligible to serve as an untouched evaluation period.
