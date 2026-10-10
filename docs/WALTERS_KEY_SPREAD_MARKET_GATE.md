# Walters exact-spread market price bridge — October 2026

## What this advances

The published 2026 `walters_key_number_forward_v1` snapshots lock an
independent home-margin forecast, fitted historical Gaussian residual, and
Walters-inspired excess exact-final-margin probabilities at **±3 and ±7**.
The next operation is a **separate** price comparison after freezing the
forecast, never setting the score from a sportsbook line.

A spread bet at -3 is a three-way event: WIN, PUSH, LOSS. The original
continuous-Gaussian market comparison had treated push probability as zero.
This research bridge computes the **exact** integer-margin probability
for arbitrary real -3, -3.5, +7, etc., with opposite handicap and actual
book prices verified independently. The expected return on one hypothetical
unit at an offered American price is:

`unshrunk_EV = P(win)*(decimal_odds - 1) - P(loss)`

A push returns the unit. The correct break-even probability is on
the **conditional resolved event** `P(win)/(P(win)+P(loss))`.
It is not `P(win)` alone when pushes exist. Same-book two-sided no-vig
probabilities are a reference, not independent probabilities in the model.

This is exploratory modeling, **not a BET or staking signal**; the
2025 independent historical challenger did not pass both prespecified
statistical endpoints. No prospective winning strategy or bookmaker
execution advantage has been confirmed, and no wagers are made.

## Historical-first chronology and data safety

Each first-seen model file must already be committed before kickoff
and remain identical to its Git addition commit.
**Both league model workflows now check out full Git history** with
`fetch-depth: 0`. A shallow checkout can prove only the HEAD and will
falsely reject the first publication after later commits.

The consumer is `scripts/walters_key_market_gate.py`.
Its optional input is one immutable JSON per independently recorded *paired
spread quote* under
`history/walters_key_forward_v1/book_quotes/<quote_id>.json`.
Do **not** populate it using guessed odds or automatically promote a
secondary aggregator quote to an executable book price.

Required JSON fields:
- `spec: "walters_key_spread_quote_v1"`, `quote_id` (must match
  safe filename), `game_id`, `sport`, `home_team`,
  `away_team`, `kickoff` exactly matching the first-seen forecast.
- `book` and direct HTTPS `source_url`, `source_evidence_note`
  documenting a genuine independent bookmaker/source review.
- `home_spread`, `away_spread` (must be exact opposite half-point or
  integer handicaps), `home_american_odds`, `away_american_odds`.
  Both prices must be the **same bookmaker**, same fixture and line.
- `source_quote_at`, `captured_at` (explicit timezone-aware times).
  The sportsbook quote must have been updated no more than 15 minutes before
  capture, and the quote must be observed **after the independent forecast
  was first committed** but before kickoff. The quote file's original
  Git commit must also be after capture but before kickoff.
- Five attestations: `book_identity_verified`,
  `source_quote_time_verified`, `executable_price_verified`,
  `book_access_verified`, `settlement_rules_verified`.

These flags are a review checklist, **not automatic independent proof**.
They may only be set true if a real bookmaker source was separately checked.
The executable odds and original source time must exist in reality.
Fail missing/contradictory line, odds, identity, date, first Git publication,
or source provenance closed; never default spread odds to -110.

On every qualifying complete quote, the bridge reports both forecast
models' separate home/away WIN/PUSH/LOSS probabilities, conditional
model and de-vigged reference probabilities, and correctly settled model
EV at the actual line and paired American prices. It explicitly labels
all values **ATTESTED_REVIEW_ONLY_NO_ECONOMIC_EDGE_PROVEN**.
No inputs are scraped by this module. A quote from an unverified
third-party aggregator is NOT converted into an attested bookmaker observation.

Run:
```bash
python -m pytest -q tests/test_walters_key_market_gate.py
python -m scripts.walters_key_market_gate \\
  --forecasts history/walters_key_forward_v1/forecasts \\
  --quotes history/walters_key_forward_v1/book_quotes \\
  --out reports/walters_key_market_gate.json
```

The existing league **model workflows** own the report as generated
research state. The read-only forward-shadow workflow can also upload the
new report on demand. No independent writer races the main model pipeline.

### Release decision

A complete quote is not a proven profitable betting edge. A betting edge
would additionally need a forward economic test against source-verified
executable prices and independent closing reference, with enough clean
samples and uncertainty assessment to reject market-quality or selection
artefacts. Current conclusion: **NO VALIDATED ECONOMIC EDGE**.
