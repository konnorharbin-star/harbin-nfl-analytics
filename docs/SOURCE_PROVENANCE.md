# Source provenance and prospective personnel research

Three-times-daily recommendation workflows also collect the public ESPN injury feed.
Original item dates, player/team identities, source statuses, request/observation times,
source URL and raw-response digest are retained. Compact structured payloads are content
addressed and immutable; separate capture records associate them with upcoming games.
The response-generation timestamp never establishes injury-item freshness. Each item
requires an aware date not after observation, a matched source team ID, a unique player
ID from the original source link, and a nonempty status. Items older than 72 hours are
stale. Games already started are excluded; missing records never imply a healthy team.

These are dated secondary reports, not independently verified official team availability
or starting lineups. Model scores and probabilities remain unchanged. Current evidence
cannot be backfilled into older training/evaluation seasons. HTTP, schema or size failures
are archived and shown on the dashboard. Only one bounded free request per sport/run is
made, with a 12 MB response cap; API keys, paid feeds and wagering are not introduced.

The live CFB ESPN injury feed currently contains obsolete 2020/2022 items despite a
2026 response timestamp and season label. Those items are rejected. CFB source coverage
remains blocked until a reliable current source is acquired; no substitute status is invented.
NFL current dated records are useful research inputs, but do not certify a starter.

NFL line collection now retains parsed quote-origin timestamps in immutable sidecar
records next to the legacy CSV. The CSV is locked and truly appended; its old bytes are
preserved, and repeated moneyline observations do not create duplicates. Collector time
is never converted to book update time. Origin validation requires origin <= capture <=
archive time < kickoff. Missing/naive/future origins fail closed. Legacy rows without
sidecars remain origin-unverified. Exact original request URLs are currently unavailable
in this normalized market interface and remain null; source provider/event IDs survive.
These observations alone do not certify an executable entry, closing price or edge.

Sources:
- https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries
- https://site.api.espn.com/apis/site/v2/sports/football/college-football/injuries
