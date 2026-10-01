# Stage 4 — NFL Context Layer

The NFL context layer now follows the same operating philosophy as the CFB model: availability, personnel, environment, rest, and travel are attached **after** the independent football projection and are subject to point-in-time data contracts.

## Implemented sources

### Injuries

`nfl/injuries.py` consumes the nflverse injury feed and supports the modern fields such as `report_status`, `practice_status`, `report_primary_injury`, and `date_modified`.

For a target week/as-of timestamp:

- injury rows from later weeks are excluded;
- injury updates reported after the as-of time are excluded;
- the latest admissible status for each player supersedes older reports;
- availability severity is summarized by team and position;
- quarterback risk is broken out separately.

The injury aggregation is a confidence/risk measure only. It does not assign arbitrary “points” to an injured player.

### Depth charts and rosters

`nfl/personnel.py` supports both modern timestamped nflverse depth charts and older weekly depth-chart schemas.

Modern depth records are filtered by their load timestamp and the most recent admissible team snapshot is retained. Weekly roster state is restricted to the target week or earlier. Player identity matches primarily by stable IDs with names as fallback.

The output tracks top-depth-player injury risk, offensive-line risk, skill-position risk, defensive risk, roster status coverage, and active quarterback counts.

### Rest and travel

Rest days use nflverse schedule fields when available. Travel uses deterministic NFL home-location coordinates to estimate straight-line away-team travel and time-zone change for standard home games. Neutral/international games may use the current venue location when it can be resolved; missing venue geography remains missing rather than being invented.

### Stadium / roof / weather

Indoor/closed-roof games have zero weather risk and do not make a forecast request.

Outdoor current games use the free Open-Meteo forecast API. Forecasts are selected by the hour nearest the scheduled kickoff. Temperature, precipitation probability, sustained wind, and gusts are preserved individually in the canonical output. A bounded weather-risk diagnostic is computed for monitoring/confidence only.

Venue geocoding for neutral sites is cached locally.

## Current-only boundary

By default `build_current_context()` refuses to attach current context to a historical season. Historical context research must explicitly opt in and supply point-in-time-correct source frames.

This prevents the common leakage error of using today's injury/depth/weather state inside an old backtest.

## Score-adjustment boundary

The Stage 4 output explicitly records:

```text
score_adjustment_enabled = false
```

That is intentional. The current context layer can:

- lower operational readiness;
- expose missing/stale data;
- inform portfolio confidence later;
- populate the forward decision ledger;
- support future point-in-time research.

It cannot move the canonical fair margin or total until a specific NFL context adjustment is trained chronologically and improves untouched NFL holdout data.

## Canonical context groups

Release monitoring uses four groups:

1. starting-QB state;
2. injuries/personnel;
3. rest/travel;
4. stadium/roof/weather.

Broad context coverage is the average of those four group coverages, while the hard release gate also checks QB coverage separately.

## Failure behavior

Context sources fail closed:

- unavailable injury/depth/roster feeds are recorded in source-health metadata;
- missing weather/venue data leaves the environmental component incomplete;
- unavailable rest/travel fields remain missing;
- no missing value is silently replaced with a favorable assumption;
- a context-source failure cannot enable a real stake.
