# Stage 12 — Neutral Venue and Weather Resilience

Stage 12 improves the current-only environmental context layer without changing the independent fair-score model.

## Why this exists

The 2026 Week 4 slate exposed two operational gaps: the neutral-site IND–WAS game at Tottenham Hotspur Stadium could not be geocoded by the generic place-name endpoint, and another outdoor forecast hit a transient TLS timeout.

## Changes

- recurring NFL international/neutral venues have deterministic venue coordinates;
- unknown neutral venues still use the free geocoder and remain missing if unresolved;
- Open-Meteo HTTP calls use bounded retries for transient URL/TLS failures and retryable HTTP status codes;
- forecast requests are scoped to the kickoff UTC date rather than requesting a full 16-day hourly payload;
- current weather/venue context remains a risk/confidence layer only; it does not adjust the fair score.

## Fail-closed rules

- no weather value is fabricated;
- a failed forecast remains a visible weather error after retries are exhausted;
- indoor games continue to suppress weather fetching;
- unknown venue coordinates remain missing;
- all context remains downstream of football projections until NFL point-in-time historical validation earns a score adjustment.

## Acceptance target

The current Week 4 live-source run should resolve Tottenham Hotspur Stadium without generic geocoding and should reduce transient Open-Meteo failures when the source is reachable. Release and staking gates remain unchanged.
