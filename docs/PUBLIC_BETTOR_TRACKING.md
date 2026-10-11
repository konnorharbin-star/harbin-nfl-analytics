# BETMGM EDGE LAB — Public bettor/source registry
*Initialized 2026-10-11. Free public research only. Not a list of guaranteed winners.*

## Two different things to monitor

**Source-complete dated picks:** independent exact-contract published picks, auditable to source/version and publication timestamps; market and player-prop returns require exact line/price. The source of truth is [frozen 2026 ledger](../research/public_expert_picks_2026.csv); [research workflow](../.github/workflows/public-bettor-watch.yml) regrades game markets from free NFL results.

**Professional methods:** documented process insights without pretending private bet lists or proprietary ratings are public. Extract falsifiable hypotheses, then evaluate against our chronological market-only baseline; never fit arbitrary professional "quality" coefficients without holdout proof.

| Person | Watch role | Publicly verifiable basis as of 2026-10-11 | Limitations | Starting free source |
|---|---|---|---|---|
| Sean Koerner | NFL player-usage and prop selections | Action Network profile *reports* 1,271-1,023-20 NFL, +140.16 units; FantasyPros independently documents multiple fantasy accuracy championships | Action-tracked results are not bookmaker-fill audits; fantasy accuracy != wagering ROI | https://www.actionnetwork.com/article/author/sean-koerner/ |
| Stuckey | NFL and FBS sides/totals + market disagreement | Action Network published NFL recommendations and long history of analysis | No independently validated complete NFL account ROI in this registry | https://www.actionnetwork.com/nfl/baltimore-ravens-vs-atlanta-falcons-prediction-pick-odds-nfl-sunday-night-football-october-11 |
| Billy Ward | NFL sides, key spreads | Action Network publishes authored selections | Do not assign professional-level profitable ROI absent full audited sample | https://www.actionnetwork.com/content/ravens-vs-falcons-picks-props-odds-predictions-sunday-night-football-oct-11/embed |
| John Lanfranca | NFL totals, QB/situational analysis | Action Network publishes authored totals picks | Preserve bet-to-number threshold (e.g. under 43 vs 43.5) | https://www.actionnetwork.com/article/author/john-lanfranca |
| Sam Farley | NFL receiving props and QB/target usage | Action Network publishes authored prop selections | A single article may quote two odds (London over 80.5 at -111 in body vs -115 summary); capture both and flag | https://www.actionnetwork.com/content/ravens-vs-falcons-picks-props-odds-predictions-sunday-night-football-oct-11/embed |
| Brandon Anderson | NFL futures, player props and longshot portfolio benchmarking | Action Network has a published author profile and tracked units | Many longshots, returns depend critically on staking and quoted prices; self-reported platform accounting | https://www.actionnetwork.com/article/author/brandon-anderson |
| Billy Walters | PROCESS ONLY | Historical reporting describes independently built ratings and line-aware trading | No full public contemporary pick ledger or reproducible proprietary exact parameters | https://www.espn.com/espn/feature/story/_/id/12280555/how-billy-walters-became-sports-most-successful-controversial-bettor |
| Rufus Peabody / Captain Jack Andrews | PROCESS ONLY | Unabated publishes professional market-analysis, modeling, bankroll-risk practices | Do not attribute unpublished picks or fabricated record | https://unabated.com/about-us |
| Rob Pizzola | PROCESS / EVIDENCE-AUDIT CANDIDATE | Public discussions on market efficiency and line-shopping; an independent 2026 evidence review exists | Public records offer contradictory samples; do not represent lifetime audited profitability as proven | https://sportsvestment.com/research/pizzola/ |

## Frozen pregame rules

1. Require exact source article and observed UTC strictly before kickoff. Author, league/event, period, selection, participant, line, odds and sportsbook; missing known quote -> NO PRICE.
2. Store the observed published page version and subsequent revision corrections in new records. Public articles can change after capture. An editorial quoted price is NEVER a verified executable BetMGM U.S. quote.
3. Grade both sides at **their own thresholds**. Example BAL +3.5 -122 versus ATL -3 -118 are not complementary; a three-point ATL victory grades BAL +3.5 W and ATL -3 push.
4. Grade player props only from official box scores or equivalent game-level player stats; absent precise player statistics -> NEEDS_PLAYER_BOX.
5. Hypothetical $1 results at article prices measure the published picks only, not a bettor's personal account, modeled EV, execution quality or user bankroll. Report ROI only on settled eligible samples with explicit missingness.
6. Evaluate analysts' forecasts on enough **future** records; avoid survivor bias, overlapping correlated bets, cherry-picked winning streaks, selective article publication, and multiple-comparison false positives.
7. Only blend analyst-derived features with our own independent forecast after demonstrating incremental out-of-sample scoring improvement and conservative EV vs contemporaneous sportsbook prices.
8. GitHub workflow automatically regrades previously logged picks on Sunday, Monday, Tuesday (free nflverse-derived schedule mirror); **it does not independently discover new picks**. New records are added by cited ChatGPT research/GitHub commits before kickoff.

## Operational sources

- [Public frozen picks](../research/public_expert_picks_2026.csv)
- [Reproducible grader](../scripts/public_bettor_tracking.py)
- [Cloud tests](../tests/test_public_bettor_tracking.py)
- [First successful run](https://github.com/konnorharbin-star/harbin-nfl-analytics/actions/runs/38181552643)
- [Living project policy](https://docs.google.com/document/d/12vooOwxCDrxaSRRXFof2xIXtBYJyqj3pacZeQj8ptYE/edit)
- [Immutable-in-principle forward research ledger](https://docs.google.com/document/d/1hcb2MBOor5O8-48-6wyg43rJ39RKh8cWA5hxSs0dHJk/edit)
