# Stage 42 — Free ESPN archived NFL historical evidence

Stage 42 removes the historical-entry dependency on a paid provider by using ESPN
Core's explicit archived sportsbook opening and closing snapshots for completed NFL
games.

The discovery path first tested ESPN's movement-history endpoint. That endpoint returns
HTTP 200 but no movement rows for the sampled completed games. The same Core odds
objects, however, expose provider-labeled open, close, and current market stages with
complete prices and lines.

## What ESPN archives

For completed games, ESPN Core can expose sportsbook-specific:

- opening home/away moneylines;
- opening spread lines and side prices;
- opening total line and over/under prices;
- closing home/away moneylines;
- closing spread lines and side prices;
- closing total line and over/under prices.

The sampled 2025 Week 1 event exposed both ESPN BET and an ESPN Bet live-odds feed.
Canonical sportsbook identity keeps those aliases from being treated as independent
books.

## Historical provenance contract

The archived opening and closing stages are not timestamped market captures.

Stage 42 therefore does not put them into the point-in-time market-history table and
does not invent capture timestamps. Historical verified rows explicitly carry:

- entry_price_stage = espn_archived_open;
- closing_price_stage = espn_archived_close;
- historical_provenance = provider_labeled_open_close;
- entry_price_verified = true;
- entry_quote_verified = true;
- entry_timestamp_verified = false.

That is sufficient for the historical opening-entry evidence contract because ESPN
identifies the market stages and exact prices. It is not sufficient for forward
execution evidence, which continues to require actual pre-kickoff capture timestamps.

## Leak-free backtest

The football side is unchanged.

For each historical week:

1. rebuild the fair-score projection without sportsbook inputs;
2. fit the probability distribution only on earlier games;
3. read ESPN's archived opening two-way market;
4. compare the already-built football probability with the opening price;
5. select one best side per game/market;
6. grade at the exact archived opening price;
7. compare the selected opening side with the same sportsbook's archived closing
   market for CLV.

No historical sportsbook price enters football features or score fitting.

## Evidence and policy gates

The free ESPN rows write the same canonical evidence files as the optional timestamped
provider path:

- reports/verified_market_bets.csv
- reports/verified_market_backtest.json
- reports/evidence_report.json
- reports/production_policy.json

The existing hard standards are unchanged. Historical evidence still needs at least
1,000 verified bets, positive lower 95% ROI confidence bound, positive average CLV,
at least 90% closing-CLV coverage, and positive evidence in at least two markets and
two seasons.

Nested chronological policy calibration still uses development, tune, and untouched
evaluation splits. Stage 42 does not force a market or deployment mode to pass.

## Automation

Free ESPN Historical Market Backtest validates fixtures on pull requests. After merge
or manual dispatch it builds the 2023–2025 archive, commits the canonical verified
evidence through the shared generated-state reconciler, and triggers the canonical NFL
model workflow to recalculate release state.

The optional The Odds API historical workflow remains available as a timestamped
historical source but is no longer required simply to obtain verified archived
opening/closing prices.

## Forward boundary

Stage 42 is historical only. Current and forward evidence remains based on observed
timestamped ESPN/Action Network line captures, independent grading, and the existing
300-bet forward release gate.
