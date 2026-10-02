# Stage 41 — NCAA-parity free NFL multi-book source

Stage 41 ports the NCAA model's free Action Network current-market enrichment layer to
the NFL model.

The independent fair-score and probability models are unchanged. Action Network is
queried only after the football projection already exists.

## Source hierarchy

Current verified markets now aggregate:

1. ESPN public NFL scoreboard/Core odds;
2. Action Network public NFL scoreboard;
3. optional The Odds API when `THE_ODDS_API_KEY` is configured;
4. nflverse schedule market fields only as explicitly research-only fallback rows.

Action Network is enabled by default and can be disabled with
`HARBIN_ACTION_NETWORK=0`.

## Fail-closed Action Network contract

The adapter accepts only rows that can be mapped unambiguously to a target NFL game.

For each sportsbook and market:

- moneyline requires both home and away American prices;
- spread requires both side prices plus complementary spread lines;
- total requires both over and under prices at the same total;
- provider update timestamps are used when present;
- when the public NFL response omits provider update time, the UTC collector observation
  time at response receipt becomes the auditable quote timestamp;
- malformed or ambiguous data is skipped rather than synthesized.

Rows remain independent by sportsbook. Prices from two books are never combined into a
synthetic two-way quote. Collector observation time is not represented as a provider
last-update time; it records when this system actually observed that public price.

## Sportsbook identity

Known Action Network book IDs are mapped to sportsbook names before canonical breadth
accounting.

This matters because ESPN currently exposes DraftKings. Action Network DraftKings is
therefore canonicalized to the same `draftkings` identity and cannot falsely turn a
single-book game into a multi-book game.

Breadth only increases when a genuinely distinct book such as FanDuel, BetMGM, or
Caesars is observed.

## Live validation

The existing `Current NFL Market Comparison` workflow now runs on relevant pull
requests and:

1. executes the Action Network fixture tests;
2. runs the canonical current-market aggregation against the live slate;
3. writes source breadth, distinct-book count, and multi-book coverage into the
   diagnostic artifact.

The live response decides whether this stage reduces the production multi-book blocker.

## Model boundary

Sportsbook data remains downstream of the independent score model. This stage does not
change projections, probability calibration, betting thresholds, policy selection,
Kelly sizing, evidence minimums, or the release-state rules.
