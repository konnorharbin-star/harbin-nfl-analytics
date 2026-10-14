# NFL receiving-yards edge candidate: workload × efficiency

Classification: **RESEARCH CANDIDATE**, not a validated profitable bet.

The old usage test imputed missing player-stat rows as zero. This new fixed
experiment joins only recorded outcomes to forecasts selected from prior weeks.
Missing rows remain unknown. Both comparison arms use the same players/games.
2024 alone selects target-volume lookback from 3, 5 or 8 games and fits
position/market empirical normalized residuals. No 2025 outcome selects a
window, coefficient or threshold. 2025 was previously inspected and is
diagnostic, not a pristine confirmatory holdout.

One synthetic primary threshold per player/game/market is chosen using the
baseline-only probability closest to 50%, with smaller-line tie breaking.
No actual outcome or challenger probability chooses this threshold. The
3,000-resample game bootstrap uses 97.5% intervals for two-market Bonferroni
correction. These intervals do not correct all earlier project experiments.

| Market | 2025 groups / games | Base Brier | Usage Brier | Usage-minus-base adjusted interval |
|---|---:|---:|---:|---:|
| Receiving yards | 3,296 / 224 | 0.19781760 | 0.19410400 | -0.00554038 to -0.00197843 |
| Receptions | 3,331 / 224 | 0.22861669 | 0.22803991 | -0.00163278 to +0.00044793 |

Receiving-yards MAE fell from 18.1757455 to 17.7612451 yards, improvement
0.4145004 yards; adjusted interval -0.5520036 to -0.2822073 yards. Selected
lookback is eight games for both families. Training matched groups: 3,270
receiving yards and 3,290 receptions. Missing target stat groups:
1,389 in 2024 and 1,349 in 2025 across the two markets, excluded rather than
treated as zero/DNP/void. Removing them changes the evaluation population.
The 15-week-cluster sensitivity interval for receiving-yard Brier change is
-0.00549147 to -0.00183084; the receptions interval is -0.00156717 to +0.00023665.
Thus the receiving-yard sign survives both game and week clustering.
Results apply conditional on a recorded source stat row, not to every player
or sportsbook participation/settlement rule.

**Receiving yards has a historical predictive signal worth testing.
Receptions does not clear the adjusted probability-score comparison. Neither
has demonstrated market-relative profit.**

## Exact current research target

SEA at DEN, Thursday October 15, 2026, 8:15 PM Eastern.
Jaxon Smith-Njigba receiving yards **over 89.5 at listed -115**.
Observed public publisher:
https://www.covers.com/sport/football/nfl/players/196870/jaxon-smith-njigba
Publisher displays October 13, 21:16 ET update; original sportsbook update and
account availability remain unknown. Capture time/hash and source classification
are retained in research/jsn_receiving_usage_watch_20261014.json.

Independent conditional workload forecast captured 2026-10-14T01:59:07.109802Z,
before the subsequent full-page price evaluation:

- Expected receiving yards: **104.3863**.
- P(over 89.5 | matched recorded stat row): **0.5847840**.
- Conditional integer-stat push mass at 89.5: zero; under probability 0.4152160.
- No-push break-even at -115: **0.5348837**.
- Raw conditional arithmetic proxy: approximately +9.33% per nominal unit.
  This is not calibrated or executable expected return.

The publisher's displayed update precedes this new forecast freeze, even
though our subsequent page capture follows it. It cannot be reclassified as a
verified model-before-book-origin price observation. Opposite sides from other
books/different lines are not combined into a same-contract no-vig probability.
No real wager or paper bet was recorded.

An actual-line case check was also performed on Covers' archived 95.5-yard
lines for Smith-Njigba in 2026 Weeks 4 and 5. The prior-week-only usage model
gave conditional over probabilities 63.2495% and 54.6099%; actual yards,
crossmatched to nflverse, were 76 and 102 (one under, one over). Both prices
were displayed as -110 but price-side/original publication provenance was
unverified. Two player-selected retrospective observations provide no market
validation or legitimate ROI estimate. Report:
reports/jsn_real_line_sanity_check_20261014.json.

Official Seattle source, updated October 13 at 16:10 site display:
https://www.seahawks.com/news/week-6-injury-report-seahawks-at-broncos
lists Smith-Njigba as estimated limited with an ankle injury. Official Denver:
https://www.denverbroncos.com/news/injury-report-cb-riley-moss-remains-limited-practice-participant-ahead-of-week-6-game-vs-seahawks
lists Surtain/Hufanga as estimated nonparticipants and Moss limited. These are
practice estimates, not final inactives. No unvalidated numerical injury or
weather adjustment was added to the forecast.

## Evidence frozen for the next test

540 unpriced receiving-stat forecast groups across 15 future NFL games within
10 days, with original values and 2024 residual catalogs, are retained in
research/receiving_usage_forward_first_seen_20261014.json. An initial development
scan included the rest of a partly eligible future week; 372 longer-horizon
groups were excluded by a game-level 10-day cutoff before publishing this
cohort. No actual outcomes determined this exclusion. This new research cohort
does not revise the earlier 698 primary anchors or coupled first-seen forecasts.

Source row SHA, quantitative report and source quality:
reports/receiving_usage_conditional_edge_test.json.
Code: scripts/nfl_receiving_usage_edge_test.py.

Next proof: exact historical/current offered lines, two-sided same-contract
market probabilities, verified player eligibility and settlement, independent
frozen forward outcomes and price-matched returns. A better synthetic-threshold
Brier score cannot certify a sportsbook advantage. Full unpriced forecast
outputs are retained so later price comparisons cannot refit the original model.

Current decision: **PASS for a real-money recommendation; retain the
Smith-Njigba contract as an explicit research watch.**
