# Exact-line market-first price scan v1

Run `python -m scripts.price_comparison_scan --sport SPORT --refresh-market`.
SPORT is nfl or cfb. This is a pricing diagnostic and review queue, never BET.
It changes no football prediction or release requirement. The existing three-times-daily
recommendation workflow refreshes bulk public data and permanently publishes scans.
The NFL bulk client makes at most two public endpoint attempts; CFB one request.
No paid odds client is called. HTTP/source failures remain visible. The scanner
may use an existing published observation only while its collector age is <=15 minutes.

Require two actual same-book prices. Compare the exact game, market and line only;
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
