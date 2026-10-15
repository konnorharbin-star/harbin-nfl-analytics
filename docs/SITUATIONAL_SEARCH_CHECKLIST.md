# Mandatory situational search on every NFL run

Owner reaffirmed this requirement October 14, 2026. The previous four broad
queries did not establish explicit coverage of individual situational factors.

Every forecast game now receives 15 dedicated factor queries: availability and
injuries; quarterbacks; weather; venue/roof/surface/altitude; rest; travel; kickoff
and body clock; schedule spots; matchups; coaching; player workload; recent form
and variance; motivation; officiating; and market sentiment. Game/prop price and
historical-price searches are additional, for 17 queries per game.

All run through the existing free public search route, publisher-domain allowlist,
dated snapshots and source hashes. This is mandatory in both root forecast
commands and the main GitHub NFL MODEL workflow, independent of the optional
experimental-diagnostics switch. Four concurrent search requests per game bound
I/O concurrency; article reads remain capped at 32 per run and are allocated
round-robin across games rather than letting the first game consume the budget.

Each game's `situational_factors` maps all required factor names to an attempted
search, observation time, source hash and trusted leads, including page-read
status/hash/time. Search errors, no trusted results, unread leads and read but
unverified pages are distinct statuses. The report and GitHub summary display
coverage for every game. A completed checklist means attempts were recorded;
it does not mean all searches succeeded or facts were established.

Public article text is untrusted evidence. This collector does not verify its
publication time, game-specific facts or numeric effect; verified values and
model adjustments remain null. Missing information is not assumed neutral.
Motivation/body-clock/officiating anecdotes are not assigned invented points,
and market sentiment is not an independent football input. Existing structured
injury/weather evidence collection remains active downstream.

Outputs: `reports/nfl_web_model_intake.json`, its `.md` coverage summary,
and original compressed web snapshots. Neither forecasts nor historical
predictive fingerprints are changed by this search expansion. This implements
the sourcing requirement; it does not claim improved accuracy or a proven edge.
