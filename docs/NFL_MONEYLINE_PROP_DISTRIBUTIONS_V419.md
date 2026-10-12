# BetMGM Edge Lab v4.19.0 — NFL Moneyline + Full Player-Prop Distributions

**Status:** Research-only infrastructure/model baseline verified on 2026-10-12 UTC. No approved positive-EV bets, no executed bets, and no new claim of outperforming BetMGM.

## NFL two-way moneyline: deployed

- Code: [scripts/nfl_moneyline_elo_research.py](../scripts/nfl_moneyline_elo_research.py)
- Tests: [tests/test_nfl_moneyline_elo_research.py](../tests/test_nfl_moneyline_elo_research.py)
- Scheduled GitHub workflow: [.github/workflows/edge-lab-moneyline-elo.yml](../.github/workflows/edge-lab-moneyline-elo.yml)
- Cloud run **success**: https://github.com/konnorharbin-star/harbin-nfl-analytics/actions/runs/38204208915
- Original 2026 source records: [history/moneyline_elo_v1_2026.jsonl](../history/moneyline_elo_v1_2026.jsonl)

The formula is fixed, public-style outcome-only Elo (1500 initial, K=20, nonneutral +65 home Elo, 2/3 preseason retained). Forecast every matchup in a week **before** updating team rating with any scores from that same week; incomplete weeks cannot update any team's state. Fit tie rate only on 2014–2023 regular game outcomes, with predeclared half-count smoothing. Calculate P(home win), P(away win), P(game tied/refunded) separately for two-way NFL moneyline. This is a structural baseline, not a new verified market-beating model. Ties void standard two-way BetMGM NFL moneylines; three-way moneylines are a different market. BetMGM background: https://sports.betmgm.com/en/blog/nfl/what-happens-in-nfl-betting-for-a-tie-game-bm06/

**Genuinely new forward forecasts:** 15 first-seen original 2026 pregame winner-probability rows, all from approaching Week 5 Monday + 14 Week 6 games; user account executable BetMGM moneyline quotes = zero. Original Monday 2026_05_BUF_LA snapshot 2026-10-12T04:33:07.671814Z: LA win 0.5479706189862706, BUF win 0.4480033073940974, tie/push 0.004026073619631902. **This is a weak baseline, not an approved moneyline wager**; current opponent QB/injury/market forecasts not comprehensively integrated.

**Previously inspected 2024/2025 diagnostic; all non-tied games, archived no-vig moneyline common games:**

| Season | Common games | Elo Brier | Archived no-vig market Brier | Elo accuracy | Archived market accuracy |
|---|---:|---:|---:|---:|---:|
| 2024 | 272 | 0.2230131 | 0.2001616 | 0.6323529 | 0.7169118 |
| 2025 | 271 | 0.2302738 | 0.2121390 | 0.6088561 | 0.6531365 |

Lower Brier better. **The archived market baseline beat current Elo in both previously inspected years**. Retrospective market fields were not accepted/account-verified BetMGM quotes, and 2024/25 were already inspected; no independent edge proof or tradable historical EV.

## NFL player props: original complete-distribution representation deployed

- Code: [scripts/nfl_player_prop_full_distribution.py](../scripts/nfl_player_prop_full_distribution.py)
- Unit tests: [tests/test_nfl_player_prop_full_distribution.py](../tests/test_nfl_player_prop_full_distribution.py)
- Scheduled workflow: [.github/workflows/edge-lab-full-player-prop-distribution.yml](../.github/workflows/edge-lab-full-player-prop-distribution.yml)
- Cloud run **success**: https://github.com/konnorharbin-star/harbin-nfl-analytics/actions/runs/38204402094
- Immutable **2024-only** originally fitted residual vector catalog: [history/player_prop_2024_residual_catalog_v1.json](../history/player_prop_2024_residual_catalog_v1.json)
- Original 2026 player-game-market expected value + residual-catalog identifier: [history/player_prop_full_distributions_2026.jsonl](../history/player_prop_full_distributions_2026.jsonl)

Cloud ran 27 unit tests across distribution and original participation v2, passed. First run froze a single calibration catalog SHA256 **2f6040e722592c7771d8a75d8022926dbff32b667626a87cf771093664cf8978**, 17 position/market residual groups from 2024 stat-row-conditional population with 8,870 player-market rows, and **694 first-seen full-distribution sufficient-stat snapshots covering 15 future NFL games**. These are one distinct snapshot per player/game/stat, **not 694 offered sportsbook prices**. Source original 2026 legacy v1/v2 threshold journals unmodified. Subsequent runs must load the first calibration catalog and its hash, never silently refresh its fitted errors.

At any exact sportsbook-observed half-point line, frozen pregame mean and scale plus originally preserved sorted 2024 residual vector directly compute the v2 conditional empirical over probability. No nearest threshold and no interpolation of post-hoc grid probabilities. At *positive integer* stat threshold, separate P(over), P(under) and P(push) by adjacent empirical tail values. Whole-number **zero** is conservatively blocked because rushing-yard actuals can be negative; attempting P(under 0)=0 would be false. The function rejects later as-of than kickoff and any quote stated to originate earlier than the frozen distribution. Participation/void is not independently verified: these probabilities remain conditional on an observed official stat row, NOT unconditional tradable payoff probabilities.

**Illustration for future Monday Bills at Rams** from new original snapshot frozen 2026-10-12T04:36:20.760177Z, with direct empirical residual interpolation-free query:

- Kyren Williams 69.5 rushing yards: conditional P(over)=0.402367; P(under)=0.597633; original pregame point forecast=67.436364.
- James Cook 73.5 rushing yards: conditional P(over)=0.576923; P(under)=0.423077; original pregame point forecast=92.86.

These *lines* were listed in an Oct 11 2026 BetMGM authored article: https://sports.betmgm.com/en/blog/nfl/2-best-nfl-player-prop-bets-mnf-rams-vs-bills-week-5-bm16/ at -115 either side. **The article predates the model snapshot**, and its displayed 9:54 AM site time has unverified UTC timezone and NO original sportsbook market-offer update time or accepted U.S. BetMGM price. Article is entertainment/editorial, not a contemporaneous account quote. These sample exact-line probabilities are **NOT a legitimate old article price comparison, no EV, and NO recommended bets**. A legitimate value study requires model freeze BEFORE a future quote's original offer time and correct active/void status.

## Independent validation gates still not passed

1. Source original (model, prices, personnel, eligibility, feature vintage) must pre-date kickoff and be contemporaneous.
2. Compare model probabilities to precisely same-contract no-vig market benchmarks; don't label conditional distributions as unconditional probabilities.
3. Pre-register chronological future holdout, clustered uncertainty and multiple test controls. Archived favorite/market baseline wins should be acknowledged, not ignored.
4. Estimate contract-specific P(win), P(push), P(loss), execution-conditional EV only with verified eligible U.S. BetMGM prices. Zero offers confirmed to date.
5. Log prospective paper trades and closing-line value before any profitable model promotion. Registry EMPTY, status PASS.

**Next iteration:** properly estimate player active/snap opportunity and DNP/void probabilities, discover independent original sportsbook quote times and player identity crosswalks, and use market-aware but strictly pregame winner probabilities that might improve beyond the current demonstrably weaker Elo baseline. No paid APIs or automatic execution.
