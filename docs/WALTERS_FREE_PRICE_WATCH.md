# Walters free three-book pregame price-watch adapter (2026)

**Goal:** Add an actual $0 data source that can observe *paired* NFL and
college-football main spreads at independent identifiable sportsbook labels
**after** the football projections have already been frozen in Git. This
adds price-discovery capability, **not** a demonstrated betting advantage.

## Documented free source

TheRundown advertises a permanent free API-key plan (no credit card), a
five-minute-delayed pre-match feed, and three sportsbooks: DraftKings
(affiliate 19), BetMGM (22), FanDuel (23). Both NFL (sport 2) and college
football (sport 1) include main moneyline/spread/total in the free
subscription; we deliberately request **only main full-game spreads**.

Official docs and pricing:
- https://therundown.io/nfl-odds-api
- https://therundown.io/college-football-odds-api
- https://docs.therundown.io/reference/data-model
- https://docs.therundown.io/guides/getting-live-odds

The adapter in `scripts/therundown_free_watch.py` requires
`THERUNDOWN_API_KEY`. It never purchases, calls paid history, uses
historical odds/backfills, or accesses a betting account. The secret is sent
only through `X-TheRundown-Key`, never in URLs, Git or reports.

To activate from the user's own account, create a **free** TheRundown key
and add it as a GitHub Actions repository secret
`THERUNDOWN_API_KEY` in **both repos**. Without that secret the workflow
succeeds in an explicit `NO_KEY_NO_PRICE` state; a key is not presumed to
exist. The source is configurable only through the secret, not chat text.

## Why this is safer than the old aggregator labels

The free API gives canonical affiliate IDs and team IDs, per-price
`updated_at`, price IDs and line IDs (when available). For each
previously Git-published 2026 forecast:

- Require original immutable Git publication strictly before observation,
  original kickoff strictly after observation, and matching sport/date/teams.
- Identify one provider event from matching home/away team identities and
  kickoff ±5 minutes. NFL requires exact team abbreviations (known official
  aliases); CFB requires exact school-name or school-plus-mascot prefix,
  both teams. Any ambiguous or conflicting event is **excluded**.
- Require a single full-game market ID 2, period ID 0, exactly two team
  participants with IDs matching event teams, exactly one `is_main_line`
  selection per side and book, opposite spreads in half-point increments,
  real non-sentinel American prices for BOTH sides of the **same book**.
- Require both `updated_at` values to be no older than 15 minutes at
  collector observation time, never in the future, and no more than three
  minutes apart. Stale/off-board/missing/asynchronous sources are excluded.
- Reconstruct baseline and 3/7 challenger actual-line WIN/PUSH/LOSS, with
  one-unit expected-value **hypotheses** and independent no-vig bookmaker
  reference probabilities using the previously fixed frozen formula.
- Store one immutable **indicative** source observation under
  `history/walters_key_forward_v1/indicative_quotes/` keyed by the exact
  fixture/book/line/prices/provider timestamps. The JSON includes the
  provider endpoint URL (no credentials), raw response SHA-256, provider
  event ID and price timestamps, the paired spreads, and the two models'
  paper prices. Repeated unchanged provider ticks do not create new files.

The existing NFL and CFB production model workflows (the canonical Git
generated-state writers) invoke the watcher **after** the frozen forecast
step and commit its reports/records with all other generated evidence.
The PR research workflow tests everything without an API key. No new
competing writer or automation placing wagers is created.

## Critical provenance limitations

TheRundown is a third-party odds feed, **not the sportsbook itself**.
Its provider `updated_at` is an upstream price-change timestamp, not
independently audited bookmaker transaction or publisher evidence. Free
prices are five minutes delayed. The original web quote may have changed;
the user's jurisdiction/account may not support the book or price. This
adapter deliberately keeps `official_booksite_execution_verified=false`
and `wager_authorized=false`. No indicative record is converted into a
`book_quotes` executable-price attestation by default.

No positive model-only EV number proves profitability. The 2025 historical
3/7 key-number challenger did **not** pass both statistical scoring
uncertainty screens. Any retrospective/forward quote data still needs
robust same-book late price and settled outcomes, sufficient counts,
prospective predeclaration and uncertainty tests. Never force a bet.

## Reproduction

```bash
python -m pytest -q tests/test_therundown_free_watch.py
# With no secret: no external call, no made-up quote.
python -m scripts.therundown_free_watch --sport nfl
# Separate college-football repo:
python -m scripts.therundown_free_watch --sport cfb
```

Result: `reports/walters_free_three_book_watch.json`. If no credentials
are installed, `mode: NO_KEY_NO_PRICE`; quotes = 0, ROI/CLV = null,
and betting remains blocked. If configured, only active, timely **indicative**
paired price observations are reported.
