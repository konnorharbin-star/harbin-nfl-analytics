# BetMGM Edge Lab v4.13.0 — Weekly NFL public market coverage + exact-line book comparison

As-of: 2026-10-12T03:03Z. Status: **RESEARCH ONLY / NO VERIFIED POSITIVE-EV BETMGM WAGERS**.

## Upgraded actual forward price coverage

The [scheduled all-upcoming NFL public quote collector](../scripts/weekly_public_nfl_odds.py), its [synthetic chronology and failure tests](../tests/test_weekly_public_nfl_odds.py), and [scheduled Action](../.github/workflows/edge-lab-weekly-public-nfl.yml) now choose **every regular-season NFL game with a published kickoff inside an eight-day forward horizon** (192 hours, max 32). They load season 2026 schedule via free nflverse, exclude started/completed/invalid kickoff games, fetch public Action Network scoreboard odds week by week, preserve each original API *collection-time* record, and append each to the same [historical evidence file](../history/public_aggregator_quotes_2026.jsonl) with a fingerprint. GitHub Actions is scheduled three times daily during Sep–Dec 2026, with a 2026-year schedule guard; cron is best-effort, not a guaranteed exactly timed or always-on process. Future NFL seasons require reviewing the selected season in the workflow.

**First live all-upcoming test** [GitHub run 38197977052, SUCCESS](https://github.com/konnorharbin-star/harbin-nfl-analytics/actions/runs/38197977052): **15 upcoming games with captured odds, 312 bookmaker-attributed two-way spread/total/moneyline market pairs**, zero API source failures. Of the 312 pairs, **ZERO had independent original bookmaker quote-update timestamps**; no U.S. BetMGM account price or wager was verified. The sample includes original Bills at Rams Week 5 and 14 Week 6 games, with all source observations pregame. The persisted journal now has 16 records, including the earlier separate Bills–Rams capture. Last observed source time for the Week 6 games was 2026-10-12T03:00:14.569622Z. GitHub default main branch is the source and original commit history is visible; these entries are capture-only, not claim of actual accepted trades.

## First same-line cross-book market screening

New [scripts/crossbook_price_watch.py](../scripts/crossbook_price_watch.py) and [tests/test_crossbook_price_watch.py](../tests/test_crossbook_price_watch.py) use the *latest* captured public API observation per NFL game to screen BetMGM-attributed prices against **at least three identifiable OTHER named sportsbooks** on **exactly the same full-game market and threshold**. Unknown book IDs are excluded. For each opposing side, reference book proportional two-sided no-vig conditional probability is estimated separately and the **least favorable** named reference is used for a conservative, descriptive *market-price proxy*. A fixed >5% hypothetical market-price advantage and <=2.5 percentage-point inter-reference dispersion are research watch conditions only. Reference probabilities are **not independent forecasts**; spreads/totals at integer thresholds have nonzero pushes not recoverable from two-way no-vig pricing; moneyline may have NFL tie settlement conventions. The proxy must never be displayed as actual expected return or approved wager.

**First actual snapshot audit** [GitHub 38198160371, SUCCESS](https://github.com/konnorharbin-star/harbin-nfl-analytics/actions/runs/38198160371):
- 15 events evaluated.
- 30 exact selection-and-line comparisons had at least 3 other identifiable books with narrow reference disagreement.
- **0 research-price-proxy observations passed the predetermined >=5% watch threshold**.
- **0 validated model-driven U.S. BetMGM opportunities**. Betting decision **PASS**.
- Multiple simultaneous comparisons increase false discovery risk; bookmaker source update times and account executable quotes were not verified. Those 30 are not 30 independent model bets.

Some other books offered slightly different numbers (e.g. Rams -3 vs -3.5, game totals 53.5 vs 54.5). Comparing them as identical is **invalid** because cover/push probabilities differ; the new screen does not do this.

## Additional freely readable player-prop lines (NOT a sportsbook model feed)

Free BetMGM editorial pages publicly list dated Week 5 game-specific thresholds, including Allen passing yards 243.5, Stafford passing yards 282.5, Williams rushing yards 69.5, Kincaid receptions 4.5:
- https://sports.betmgm.com/en/blog/nfl/josh-allen-props-odds-gaa/
- https://sports.betmgm.com/en/blog/nfl/matthew-stafford-props-odds-gaa/
- https://sports.betmgm.com/en/blog/nfl/kyren-williams-props-odds-gaa/
- https://sports.betmgm.com/en/blog/nfl/dalton-kincaid-props-odds-gaa/

Public third-party https://legionreport.com/bills-vs-rams-player-props/ reports automatically refreshed "best over/best under" prices across other books and BetMGM. The page states it refreshes approximately every 10 minutes and lists e.g. Dalton Kincaid receptions 4.5 best under BetMGM -120, best over DraftKings +101; it does **not** supply bookmaker-original offer timestamps, and the displayed "best" prices are NOT necessarily same-book two-way quotes. Do not combine them for BetMGM no-vig probabilities or assume U.S. BetMGM account availability. Player ID match and the archived frozen probability's own fixed synthetic threshold are both prerequisites for any legitimate paired analysis. We did not install a scraper or invent account-executable odds.

## Operating rules and remaining blockers

Free resources only. No wager execution. Legacy score model remains **non-primary**. The 4,004 2026 frozen player-prop research curves remain **unpriced shadow forecasts** and have not demonstrated a market-profitable edge. None of the previously inspected 2025 synthetic Brier gains on correlated artificial threshold grids establish forward BetMGM ROI.

The next required research is either: (a) a source providing a true U.S. BetMGM per-player, per-side, same-threshold **original bookmaker quote-update timestamp and region**, or (b) forward field-verified manually captured BetMGM account offers and an independently validated calibrated player-stat probability. Once prices are available, record selection and odds **before kickoff**, evaluate outcome/tie/void, compare true EV with uncertainty to a same-line market baseline, and assess paper CLV/profit after enough *independent* chronology.

**Final status:** source coverage was materially expanded and a conservative first retrospective-of-live-capture price discrepancy audit ran successfully with no qualifying market-proxy watches. **No verified profitable selection.**
