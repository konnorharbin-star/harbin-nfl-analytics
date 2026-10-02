# Stage 37 — ESPN multi-book breadth recovery

Stage 37 fixes a loss of sportsbook breadth in the free current-market collector.

The ESPN scoreboard commonly supplies one complete provider object. The ESPN Core odds
endpoint can also expose provider objects, but the previous client queried Core only
when a scoreboard market type was missing and then collapsed all normalized rows to one
row per game/market.

That behavior made verified current market breadth appear single-book even when Core
could expose additional providers.

## Collection contract

For each target game the ESPN client now:

1. parses all scoreboard odds objects;
2. always attempts ESPN Core odds as optional enrichment;
3. preserves one normalized quote per game × market × canonical sportsbook;
4. deduplicates provider aliases such as `Draft Kings` and `DraftKings`;
5. retains the complete scoreboard quote if the Core endpoint is unavailable.

No Core failure can remove otherwise valid scoreboard markets.

## Breadth accounting

The downstream market-intelligence layer already line-shops all supplied market rows
and counts distinct canonical books. Stage 37 therefore does not change edge formulas
or fair probabilities; it restores the source rows required for those existing rules.

The ESPN diagnostic now reports:

- normalized row count;
- canonical books observed;
- book count;
- games with two or more books;
- multi-book coverage;
- maximum books per game;
- maximum rows per game/market.

## Acceptance rule

Stage 37 only improves the multi-book release blocker if live ESPN data actually exposes
two or more distinct canonical sportsbooks. If the live diagnostic remains single-book,
the release gate stays blocked and another verified source is still required.

## Model boundary

Sportsbook prices remain downstream of the independent fair-score and probability
models. This stage does not change model projections, probability calibration, betting
thresholds, staking, or release-state requirements.
