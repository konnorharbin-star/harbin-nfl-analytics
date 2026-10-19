# Public game input contract

The public web intake now checks structured current game observations in
addition to collecting the mandatory situational searches and public prop
prices. It requests ESPN's free NFL scoreboard once per Eastern game date and
archives the raw response through the existing hashed snapshot store.

An observation is accepted only when one event has the exact home team, away
team, and timezone-aware kickoff in both the event and competition records.
The event must still be scheduled, and collection and consumption must be
before kickoff. The source URL is fixed to the approved endpoint and game
date. A six-hour maximum capture age is an operational freshness policy,
not a learned predictive parameter. Freshness is checked again when the
intake finishes. Duplicate matches, postponed/live/final events, wrong dates,
missing hashes, and stale or future captures fail visibly.

Accepted fields are limited to literal provider assertions: teams, kickoff,
venue name/ID, reported indoor flag, and neutral-site flag when supplied with
the right types. The indoor flag does not confirm a retractable roof's game-day
position. Absent values remain absent. Collection time is not presented as a
publisher timestamp. Current captures cannot backfill earlier forecasts.

Each game's `structured_game_observation` preserves provenance and rejection
reasons. Its `input_data_contract` distinguishes `PARTIAL_VERIFIED_INPUTS`
from `ESSENTIAL_GAME_BINDING_BLOCKED`, lists accepted fields, and explicitly
lists missing starting-QB, player-availability, and weather/roof inputs.
The JSON and Markdown reports expose these checks in every existing default
web-intake entrypoint. A provider assertion passing these checks is not a
guarantee of external factual accuracy or complete situational coverage.

This change completes event/venue source binding, not automatic extraction
of every situational fact. Article prose stays unverified. The existing
forecast still runs before supplemental collection; no old or current
forecast is retroactively context-adjusted. `ready_for_context_adjusted_prediction`
remains false until the remaining input adapters and a tested consumption
path exist. This contract does not change coefficients, qualify bets, or
demonstrate an accuracy or profitability gain.

Windows installations now explicitly include the IANA timezone database
needed by the existing and new Eastern-date checks.
