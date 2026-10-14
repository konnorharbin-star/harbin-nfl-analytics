# Test receiving yards against actual sportsbook lines

The existing candidate improves a synthetic-threshold forecast diagnostic. It
has not demonstrated a profitable sportsbook edge. This benchmark addresses the
missing **actual price comparison**, without changing the forecasting model.

New manual GitHub Action: **NFL Receiving Yards — Real Price Benchmark**.
Use an **EndzoneAPI free account with no paid plan or funded wallet**. The provider
[documents 500 free credits/month, no card](https://endzoneapi.com/pricing), and
[historical prop lines, both odds and recorded outcomes](https://endzoneapi.com/docs/get-prop-history).
These are provider claims pending independent source crosschecks, not endorsements
of its odds provenance or assertions that its data proves an edge.

The only setup dependency is a free API key: create one at
https://endzoneapi.com, then add `ENDZONE_API_KEY` under repository **Settings →
Secrets and variables → Actions → New repository secret**. Do not paste the key
in an issue, committed file or chat. Then manually run the new Action.

One run requests at most 20 players × 10 credits = 200 credits. Players are
selected from **2024 regular-season WR/TE target volume**, with GSIS ID tie breaks;
2025 outcomes do not choose players. Requests specify DraftKings, receiving yards,
the 2025 season and at most 200 rows/player. The workflow is never scheduled and
does not run during the regular NFL forecast workflow. Repeated manual runs consume
additional free allowance. It cannot purchase credits, upgrade, fund a wallet,
retry exhausted requests or place wagers. Completed responses are checkpointed
before the next call so source failures leave usable partial evidence.

For an existing export, no API call is needed:

```bash
python -m scripts.nfl_receiving_market_benchmark --history PATH_TO_EXPORT.json
```

For a local one-time free import with the key already set in the environment:

```bash
python -m scripts.nfl_receiving_market_benchmark --fetch-free-history --players 20
```

The eight-game window remains fixed at its previously selected 2024 value. The
conditional empirical error distributions fit 2024 only; forecasts for each 2025
game use prior weeks. Evaluation years already inspected remain **diagnostic**.
Import requires a declared single bookmaker, exact GSIS/game identities, real
half-point line, finite over/under prices and a recorded outcome matching nflverse.
Missing rows are not zero/DNP/void. Duplicate contracts are removed entirely;
integer lines with unresolved push treatment are excluded. No synthetic or default
price is substituted. Provider-reported historical close remains a proxy because
book-origin times, executable entry and settlement rules are not verified.

Reports compare usage/baseline/market-proxy Brier scores, log loss and fixed
ten-bin ECE. Game-cluster probability-score intervals use a four-family correction
and require at least 20 distinct matched games. One positive raw-EV candidate per
game is selected before using its outcome, with deterministic tie breaks. Its
nominal fixed-odds return is a **diagnostic proxy**, not actual ROI; CLV remains
unknown. This test cannot establish a verified price-matched forward record.

Artifacts: `receiving_actual_market_history.json` and
`receiving_real_price_benchmark.json`. Without a key/export, the report explicitly
says `SOURCE_BLOCKED`. Validated edges and promotion remain zero/false even if a
small historical subset looks profitable. Actual results are pending data access;
test fixtures are not sportsbook evidence. Existing release gates and immutable
receiving/game forecast cohorts are unchanged.
