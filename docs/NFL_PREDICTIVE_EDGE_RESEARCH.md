# NFL edge research: prediction correction and free two-book price scanner

## Findings
The earlier canonical NFL archive reports strong model overconfidence and negative historical archived price ROI. The market-relative raw probability deviations are **not** demonstrably profitable. Raising the raw probabilities, removing the regime blocks or calibrating directly against results of games already seen would only camouflage the problem.

This upgrade isolates **two distinct ways to find a possible edge**:

### 1. Can the football model improve the sportsbook market probability?
The `nfl.market_residual_challenger` audit calculates

`logit(p_candidate) = logit(p_no_vig_market) + alpha * (logit(p_football) - logit(p_no_vig_market))`.

- **2024** archived outcomes select the signed residual weight `alpha` using development log loss plus a fixed zero-centered penalty. This candidate can discover that the football model's disagreement with the market is historically *anti-informative*, but it does not assume this is a true predictive signal.
- **2025** archived outcomes evaluate a fixed 2024-selected alpha against both zero-alpha market only and the raw model. The 2025 games have been **repeatedly inspected** by previous work. Consequently they are a historical *diagnostic*, **not a pristine untouched holdout**.
- Every market separately publishes paired Brier, log loss and hypothetical archived ROI above a fixed +3% EV cutoff, with actual counts. The actual historical archived entry timestamps are unverified. Neither a better probability score nor an archive ROI can validate actual bet execution.
- **Nothing** writes a new probability into the canonical model. No staking, threshold adjustment, release gate or bettor-authorized recommendation results from this report. The experiment needs fresh, frozen forward 2026 outcomes for predictive promotion.

Published diagnostic: `docs/market_residual_challenger.json` after a successful NFL-model workflow.

### 2. Can free current sports books disagree enough to show a mathematically attractive two-price opportunity?
The `nfl.price_edge_scan` inspects **already-collected** public ESPN and Action Network two-way quotes, without fetching additional sources.

- Require verified game ID, provider event ID, named book, exact opposing sides, exact same market point/line, named distinct canonical sportsbooks, **source-origin per-side bookmaker update timestamps on BOTH complementary sides**, timezone-aware quote captures, no future/postkickoff observations and no stale source quotes older than 10 minutes. An identical collector download timestamp across books is never proof that these book odds were current. Quotes with only an ESPN capture time or Action Network collector fallback are excluded from evidence-qualified findings.
- Compare captures from two different books only if within **five minutes**. Do not build a fictional middle from mismatched spreads or totals, attribute anonymous nflverse schedule prices to a book, accept paid-provider rows, or pair identical sportsbook aliases.
- Only surface potential two-outcome arithmetic when `1/decimal_A + 1/decimal_B <= 0.995`, i.e. a **0.5 percentage-point theoretical buffer**. Reports show the quoted timestamps, point line, named books, inverse-decimal sum and theoretical payout.
- **These are not independently certified simultaneous executable prices.** Public quote capture can lag the actual sportsbook, bets can move instantly, stake limits/settlement rules can differ, and an apparent arbitrage may be unfillable. It never tells the user to place a wager.

Published same-run scout: `docs/price_edge_research.json`, including the number of zero-match scans and excluded reasons.

### Cross-book consensus disagreements (independent price-based challenger)

In addition to cross-book theoretical arbitrage, the scout calculates a **leave-one-book-out** no-vig consensus from at least **three other named sportsbooks** quoting the exact same market line within a five-minute capture synchronization window. It rejects consensus references that disagree with each other by more than **2.5 percentage points**, and surfaces hypothetical market-implied EV above **5%** at a candidate sportsbook.

This is a **candidate for manual investigation, not a proven value bet**. Removing each target sportsbook from its own reference guards against self-confirmation, but the other books may share correlated prices, the public captures may be delayed, bookmaker limits can vary, and a three-book median is not necessarily a sharp true probability. These data cannot alone certify an edge. Scan results include a separate `disagreement_watchlist`; an empty list means the price data did not support any candidate. Current NFL prediction probabilities are never used in this market-price reference.

## Operational constraints

No paid data, authentication, browser accounts, wagering automation, wallet, order endpoint, deposits, transfers, placing orders or betting subscription. GitHub Actions/Python and free public data only.

**The combined system may truthfully say NO EDGE.** It must not invent a winner, inflate EV or report proof of profitability when quotes or independent holdout results are missing.

Next work: accumulate **first-seen 2026** predictions with exact source timestamps, independently check final scores, evaluate model residuals and possible two-book market arbitrage out of sample, and add source-origin timestamps where the public data permits. A genuine positive, persistent edge can only be claimed with independent forward validation.


## Source-provenance correction — 2026-10-08

The first operational scanner found 10 numerical cross-book pairs and nine
consensus disagreements, several implying wildly implausible 10–40% two-way
returns. They were **not independently verified tradeable opportunities**:
those book prices shared a synthetic request-capture timestamp because a free
provider supplied no trustworthy source-origin quote update time.

The scanner now **rejects all such pairs**. For Action Network, only each
specific side quote's timezone-aware `last_update`, `updated_at` or
`timestamp` counts. It must exist for both sides of the book's two-way
snapshot and the two timestamps must agree within two minutes. An event-level
update, market-level timestamp, or collector-observed time is not sufficient.
ESPN records with no per-book quote-update time remain unverified.

No measured sportsbook profit, risk-free arbitrage, or edge claim should be
drawn from the original 10/9 observation counts. Historical reports or cached
screenshots showing those values are superseded by the strict source-time
verification requirement. Results may legitimately drop to **zero**, which is
preferable to misleading apparent arbitrage.
