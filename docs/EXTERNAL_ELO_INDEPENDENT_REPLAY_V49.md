# External NFL Model Independent Replay — v4.9.0

**Completed 2026-10-11 UTC — all scores historical RESEARCH ONLY; no bets.**

This is the follow-through on the user's request to **independently test apparently successful external models and only change our forecasting coefficients when actual evidence supports it**. No fabricated prices, paid data feeds, accounts or third-party private algorithms are involved.

## Source methods and reproducibility

[FiveThirtyEight's published NFL Elo implementation](https://github.com/fivethirtyeight/nfl-elo-game/blob/master/forecast.py) and [methodology](https://fivethirtyeight.com/methodology/how-our-nfl-predictions-work/) publicly specify K-factor 20, a 65-Elo-point historical home advantage, offseason 1/3 mean reversion and a score-margin-sensitive Elo update. This is an older public template, **not the later full quarterback-adjusted version**. Source license is MIT; the new repository code is written independently using the documented mathematical methods, not copied proprietary material.

We built [scripts/external_public_elo_replay.py](../scripts/external_public_elo_replay.py) with [seven safety/unit tests](../tests/test_external_public_elo_replay.py). Source: nflverse public historical schedules via `NFLDataClient`, 2014–2025. Ratings mature on 2014–2018 results; 2019–2022 completed results fit **only** a scoring-margin-per-Elo coefficient. Entire weeks are scored using their pre-week team ratings; no game updates enter the SAME week, with updates and trailing scoring history applied only after all that week's forecasts. Franchise relocations are normalized and neutral sites remove home Elo advantage.

FiveThirtyEight Elo itself does **not** forecast totals. We independently proposed a parsimonious, historically shrunk prior-8-game team offense/defense trailing average, league-average prior and fixed home scoring allowance. The 2019–2022 period determines a modest total intercept. This is our **own challenger**, not a model misattributed to FiveThirtyEight. All parameters are locked before 2023–2025 diagnostics.

Compare the exact same 2023–2025 NFL regular-season **weeks 5–18** event IDs with the canonical `nfl.priors.build_prior_walkforward` spread/total football model and nflverse archive-final sportsbook lines. Archived lines are NOT independently source-time-verified BetMGM offers or proven that a price was on screen before kickoff. All 2023–2025 games have already been inspected during prior project research, so they cannot be used to justify a model release.

### Actual results from GitHub Actions

[Real-data replay GitHub Actions run: SUCCESS](https://github.com/konnorharbin-star/harbin-nfl-analytics/actions/runs/38183991172). Regression unit tests **passed**, full free 2014–2025 data replay succeeded, and the run produced an archived fingerprinted JSON result.

| Year | Target | Public-method-inspired challenger MAE | Canonical independent MAE | Archive-final market MAE | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| 2023 | Margin | 9.975454 | 9.998622 | **9.487981** | slight challenger historical gain vs canonical, no market gain |
| 2023 | Total | 10.766366 | 10.866259 | **10.406250** | slight challenger historical gain vs canonical, no market gain |
| 2024 | Margin | 9.989469 | 9.984249 | **9.288462** | canonical marginally better, market best |
| 2024 | Total | 10.077979 | 10.127727 | **9.783654** | slight challenger historical gain vs canonical, market best |
| 2025 | Margin | 10.610963 | 10.606922 | **9.992788** | canonical marginally better, market best |
| 2025 | Total | 10.618955 | 10.483861 | **10.081731** | challenger worsened, market best |

208 matched games per year; **624 diagnostic game comparisons** total. Descriptive weekly bootstrap 95% intervals for challenger-minus-canonical MAE all straddled zero:
- 2023 margin [-0.307806,+0.211462], total [-0.395400,+0.218625]
- 2024 margin [-0.224324,+0.279198], total [-0.293253,+0.174824]
- 2025 margin [-0.272097,+0.287694], total [-0.108507,+0.442224]

**External_Elo_Margin_V1: RESEARCH-ONLY, NO PROMOTION.** Improved canonical MAE in 2023 by 0.023, then slightly worsened 2024 and 2025, did not beat archive-final market in any year. This is not verified incremental market skill.

**External_Trailing_Offense_Defense_Total_V1: RESEARCH-ONLY, NO PROMOTION.** Slight diagnostic gains in 2023 and 2024 but material deterioration in 2025; market remained more accurate all years. No demonstrated betting value.

## Claimed profitable external strategy verification

[ShamgarBN/nfl-bet-engine](https://github.com/ShamgarBN/nfl-bet-engine)'s README reports **+5.3% historical moneyline ROI**, but the referenced `logs/backtest_v6_full.csv` artifact was not available at that path in the publicly accessible default-branch repository during this 2026-10-11 inspection, and the entire `logs` directory was absent via GitHub contents API. This is **not proof its numbers are wrong**; it means the advertised report cannot be independently checked from that artifact as published. We have not rerun the source's full pipeline, verified original in-time betting prices, or validated account execution. **Do not promote its ROI claim, ATS thresholds, player-feature weights or teaser multipliers.** It remains REPLICATION_REQUIRED.

The same warning applies to any model with market-discord "edge tiers": selecting top decile after inspecting historical performance creates selection bias, and without original decision-time odds data an archive-final return is not achievable in real time.

## Preserved architecture and next research direction

Nothing in `nfl.ratings`, `nfl.probability_runtime`, `nfl.current`, `nfl.market` or account execution was changed. GitHub Actions runs are reproducible on a free runner with no paid APIs. Our model's existing pregame starter/injury mismatch protections and PASS states remain active. The research branch is the GitHub default branch, but modules are **research-only** and not inserted into the production score pipeline.

The sole credible next step to possibly prove an independent edge is **forward** source-time-verified 2026 spread/total predictions and actual two-sided U.S. BetMGM or other matched-book quote snapshots, versus a market-only benchmark with proper win/push/loss settlement and pricing. Untouched future games must be evaluated without changing the pre-registered parameters. New external models with reproducible true chronological validation can be tested as separate challengers; none reviewed yet qualifies for profit-model promotion.

**Policy decision: PASS, no verified profitable NFL spread/total BetMGM edge; version v4.9.0 denotes evidence and research capability, NOT a production forecasting coefficient upgrade.**
