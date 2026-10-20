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

## Player status and weather observations

The default forecast export includes a one-to-one GSIS/ESPN player-ID
crosswalk from `nflreadpy.load_players`, with collection time and a hash of
the normalized mapping. Ambiguous IDs are omitted; names alone never establish
identity. Game-specific ESPN summaries must independently match the verified
event ID, home/away teams, kickoff and scheduled pregame status. Only recognized
injury statuses with a source update no later than capture and no older than
seven days are retained. Conflicting latest statuses, wrong teams, unknown
IDs, undated rows and future updates remain unknown. The injury snapshot
must be fresh within six hours when consumed. These are verified reported
statuses, not official final actives or confirmed QBs. Not being listed never
means healthy or active.

NWS lookup uses the actual event venue name and the existing approximate
stadium-coordinate registry. Home venues additionally require a matching
home team and an explicitly non-neutral event; shared LA/NY venues allow
either tenant. Registered neutral venues require a neutral event. Unknown
venues and locations outside NWS coverage remain unavailable; no nominal-home-only
coordinate fallback exists. One hourly period must contain kickoff. Forecast
issue and capture must be pregame, not future and fresh within six hours.
Temperature, wind and precipitation units are checked; missing precipitation
stays missing. An outdoor forecast does not establish roof position or
on-field weather. Requests are shared by source URL within a run, hashed and
archived through the existing snapshot store. Structured requests reject
redirects to a different resource, even to another approved publisher.

## Publication and remaining gaps

The intake writes `outputs/independent_input_quality.json` alongside the
forecast by default, or beside a custom predictions path. It contains no
bookmaker prices. The current dashboard validates its exact original forecast
byte hash, generation/capture chronology, game coverage, teams and kickoff,
then rechecks source age and player identity. Missing, stale, malformed or
mismatched reports are shown as unavailable and replace any older quality
artifact. Original forecast JSON/CSV/image values and first-seen history are
not rewritten. Supplemental evidence and its hash are published separately
and archived with the workflow artifacts.

Article prose stays unverified. The forecast still precedes supplemental
collection; no forecast is retroactively context-adjusted.
`ready_for_context_adjusted_prediction` stays false because tested feature
consumption and predictive weights require separate evidence. Source checks
do not qualify bets or demonstrate improved accuracy or profitability.

## Explicit official personnel and roof applicability

Two targeted searches now precede general research queries: official game-day
inactive lists and explicit starting-QB announcements, restricted to NFL.com
and the two clubs' domains. They share the existing article request budget.
Official-page redirects must stay within the same publisher. Dates support
one unambiguous JSON-LD NewsArticle/Article as well as publisher meta tags;
future or contradictory issue/modification times are rejected. Only main
article blocks are parsed, not navigation, related headlines or RSS snippets.
All pages retain raw snapshot hashes and capture times.

Announcements require correct teams/week, recent publication and pregame
capture. Club pages can confirm starters only for their own team. Starter
confirmation requires a unique model QB explicitly named in a definite
statement such as "will start"; conditional wording, depth-chart order, pass
leaders and backup inactives are insufficient. Prose names are linked by exact
team plus a unique normalized name from the player directory's explicit name
variants. The composite binding method and model player ID are preserved;
ambiguous or unmatched names stay unresolved.

Game-day inactive lists require same-day publication and explicit team sections,
or the matching club's own report. League omnibus reports bind same-day team
lists through the independently verified schedule/week. Explicitly reported
actives are separate from inactive lists. Omitted players never become active.
Emergency-QB annotations are retained; listed-inactive status does not guarantee
zero emergency participation. Conflicting designations remain unknown. The
dashboard shows starter/list coverage and explicit player designations separately.
Official evidence is checked again at publication, including capture order.

The intake captures normalized nflverse schedule rows once, preserving their
hash and collection time. A row must match game ID, teams, kickoff, ESPN event
ID and actual venue. `outdoors` means an operating roof position is not
applicable; `dome` means a fixed roof, with outdoor weather waived only when
the event source also reports an indoor venue. `open`/`closed` schedule values
never by themselves confirm a retractable roof's game-day position. Missing,
stale, changed or mismatched rows stay unknown. These are source classification
assertions, not on-field measurements. Each resolved blocker clears separately;
predictive calibration and betting gates remain independent of source completeness.

Windows installations now explicitly include the IANA timezone database
needed by the existing and new Eastern-date checks.
