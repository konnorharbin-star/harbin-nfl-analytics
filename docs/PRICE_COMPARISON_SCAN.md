# Exact-line market-first price scan v1

Run `python -m scripts.price_comparison_scan --sport SPORT --refresh-market`.
SPORT is nfl or cfb. This is a pricing diagnostic and review queue, never BET.
It changes no football prediction or release requirement. The existing three-times-daily
recommendation workflow refreshes bulk public data and permanently publishes scans.
The existing hourly line-capture workflow also scans its newly collected records
without extra odds requests. CFB model publications refresh the scan from their
current data. No continuous background work is claimed beyond these workflows.
The NFL bulk client makes at most two public endpoint attempts; CFB one request.
No paid odds client is called. HTTP/source failures remain visible. The scanner
may use an existing published observation only while its collector age is <=15 minutes.

Require two source same-book prices. Both original opposite spread lines or
equal over/under lines must be retained. The existing CFB normalized feed drops
those paired lines, so CFB spreads/totals remain unsupported; CFB moneylines and
all three NFL markets are supported for diagnostics. Compare the exact game, market and line only;
never mix different spreads or totals. Home/away spreads must use opposite signs;
missing prices never receive -110. Deduplicate bookmaker aliases, including duplicate
ESPN/Action Network DraftKings labels. Keep each book's latest observation and require
at least four unique source-book identities within two minutes. Exclude the quoted
book from the reference; take the median de-vigged probability from the other three
or more. Flag gaps of at least one percentage point between reference and quoted
break-even probability. Rank review items using a fixed stress haircut of the greater
of two percentage points or the full reference-probability range. This is **not a
confidence interval or calibrated model edge**. References condition on non-push
settlement; no true EV, ROI, fair-price, guaranteed-profit or bet instruction is inferred.

The free aggregator feed does not verify per-side book update timestamps, original
book availability, geographic access or settlement rules. Some labels are unresolved
numeric IDs or legacy brand names. Canonical keys only deduplicate; mappings do not
certify current sportsbook identities. No observed provider is certified a sharp
reference. Economic performance of this market-reference strategy is unproven.
Those blockers keep betting authorization false and stakes zero for every scan,
regardless of price disagreement. Source record dates never become sportsbook
update dates. Older prices and started games are excluded; the dashboard also hides
expired candidates at viewing time. Crossed/ambiguous or unsupported data stays blocked.

Immutable scan captures retain normalized source records and an append-only NO BET
scan decision in history/price_scan_v1. These research scans are separate from BET
recommendation receipts and do not count toward paper wager returns. Existing
recommendation recording and grading remain unchanged. Free bulk collection occurs
at the existing recommendation schedule, not continuous polling or wagering.

Concurrent collectors retain separate mutable dashboard summaries (_line, _model
and the recommendation-job default). The dashboard reads the newest dated summary,
while all immutable captures share the append-only history. This avoids overwriting
another collector's report or changing strict publication conflict handling.

## Source-coherence quarantine (October 10, 2026)

The October 10 NFL snapshot contained an unverified Action Network-labelled
FanDuel moneyline pair for PHI at JAX: **JAX +110 / PHI -130**. Seven other
same-game paired books showed Jacksonville about -375 to -400.
FanDuel's published Week 5 schedule also showed JAX as a substantial favorite,
not +110:
https://www.fanduel.com/research/nfl-week-5-schedule-odds-for-every-game

This is not a certified mispriced sportsbook offer. Source-to-book mapping,
game association, update timestamps and executable prices were not verified.
It is an instructive case of an aggregator anomaly being mistaken for a
potential huge edge, and must not appear at the top of a betting queue.

The scan now uses a prespecified **source-integrity**, not profitability,
threshold. On each exact game / market / handicap, find the paired no-vig
probability of each book. Compare to the median of all **other** books.
A quote more than **7.5 percentage points** away is quarantined unless all
of book identity, quote-origin timestamp and executable price were
independently verified. First-seen aggregator retrieval time alone does not
qualify. Quarantine retains the full quote, observed price and divergence for
manual review; it removes unsupported extremes from both candidate selection
and reference probabilities. If fewer than four books survive, issue no
candidate for that market/line.

This 7.5 pp is a deliberately conservative fixed quality-control boundary
for conspicuous quote errors, **not fitted to past bet outcomes or validated
as an EV threshold**. A genuine independently verified large price
difference may still appear in the diagnostic queue, but it never
authorizes bets. False positives and legitimate rare opportunities can both
be quarantined, so report the count and review the raw provenance instead
of silently treating them as no edge.

The change does not alter fair score, probabilities, active betting policy,
economic thresholds, historical wagers or autonomous execution. It only
affects diagnostic review-candidate rankings. No profit claim is made.
