# Jason Cooper (@JasonCooperBets) method audit — research registry

Captured 2026-10-11 UTC. **Research only. No claim of verified profit from this project.**

## What is actually visible

A user-supplied screenshot of an October 6, 2026 pinned X post on @JasonCooperBets shows a Pikkit-branded NFL-filtered chart labelled **+112.85 units**. The graph's filter reads **August 1 to October 31, 2026**. On the date of the screenshot, October 31 had not yet occurred, so the filter end-date is not evidence of future outcomes. The screenshot does not show the underlying number of wagers, bet-size distribution, market breakdown, ROI, average odds, source sportsbook, pregame line quality, CLV, or individual entries. It is a **posted performance claim**, not a sample for statistical validation.

External public profile snippets for [@JasonCooperBets](https://x.com/JasonCooperBets) have described algorithmic selection in MLB home-run/moneyline and NFL. The actual code, coefficients, historical forecast timestamps, or account-level bet list were **not** verifiably accessible as of this review. No secret method has been recovered.

Pikkit [official tracking page](https://pikkit.com/bet-tracker) states that BookSync imports sportsbook bets, including BetMGM, and provides ROI, profits, wins and market/sport splits. The free core tracker and follow tools differ from paid Pro CLV features. Pikkit users control which betting information is made public, so viewing a shared filtered screenshot is not equivalent to auditing every bet on a linked account. No Pikkit connector or authenticated public account history was available to this agent; do not claim otherwise.

Source clarity: Independent [Betcast records for The Hammer Daily](https://betcastai.com/podcasts/the-hammer-daily) reflect **podcast-discussed selections**, including guest picks and multi-leg parlays. Do **not** treat that show's aggregated picks as JasonCooperBets' individually executed Pikkit bets or as a contradiction of his profile P&L without independently verifying account identity, wager attribution, and original prices.

## Research objective: learn if the profit transfers to a BetMGM follower

Questions to resolve, not speculation:
- What is his complete, fully linked public Pikkit betting record and the number of **synced** wagers with actual unit stakes, price/book/market and settlement?
- How much of the NFL profit comes from full-game spread/total versus **props, live bets, futures, parlays, or boosted odds**? What are the exact ROI denominators and per-market sample sizes?
- What odds did he obtain, how soon after posting did followers see the pick, and could *our* BetMGM U.S. account get the exact same contract and odds? Estimate follower lag/price deterioration with **source time**, never substitute his original ticket's price.
- Does his unit profit persist in a newer independently frozen time period? Market movement and exact-line CLV are diagnostics; they are NOT guaranteed profitability. Were profit spikes driven by concentrated high-payout bets?
- Can his **publicly documented algorithmic variables** be sourced from free football information and independently proven to improve proper out-of-sample calibrated win/push/loss vs a contemporaneous market-only baseline? Never copy unverified percentages.

## Development completed

A standalone [Python public bettor contract audit](../scripts/public_bettor_market_audit.py) and [synthetic safety regressions](../tests/test_public_bettor_market_audit.py) were committed. They reject social profile screenshots as individual bets, duplicate IDs, postkickoff or untimestamped picks, mismatched follower lines, unsupported parlays/props graded from final-score alone, and unobservable source prices. The script optionally grades NFL full-game moneyline/spread/total against separately supplied final NFL scores and prints **flat-one-unit hypothetical returns** at both the public bettor's quoted odds and a later, separately sourced follower quote. The two price results must never be represented as account-verified or Pikkit actual units. It refuses to infer a positive BetMGM edge or approve a wager.

**No Jason-specific individual picks were scraped or added to this ledger.** At this time `audit([])` must return `NO_INDIVIDUAL_PICKS_AVAILABLE` and `PASS`. The user-uploaded image is not published to GitHub; this document preserves only its displayed public metric and research caveats.

Cloud CI workflow creation was blocked in this session; the new tests are committed but **their runtime passing is NOT verified**. This code is research-only and does not modify `nfl.ratings`, real-time sportsbook collectors, historical diagnostic baselines, prediction coefficients, or betting gates.

## Forward collection standard

Once a genuine public Pikkit profile or individual pregame betting post is available, capture `bettor, post_url, posted_utc, captured_utc, event_id, league, kickoff_utc, bookmaker, region, exact market/period/participant/selection/line/odds`. The posted odds are unverified source quotations unless independently accepted at a bookmaker. Freeze source content **before kickoff**; a retrospectively discovered previous winner must be marked RETROSPECTIVE and excluded from forward ROI and the immutable forward-bet ledger.

For actual personal wagers, Pikkit BookSync can track results **outside this model**, with personal account-level consent. Do not request sportsbook credentials in ChatGPT, attempt account automation, or upload another person's private wagering records without authorization.

**Decision: Copy verification, discipline and hypothesis formation, NOT claimed profits or unverified selections. No validated positive EV and no bet recommendation.**
