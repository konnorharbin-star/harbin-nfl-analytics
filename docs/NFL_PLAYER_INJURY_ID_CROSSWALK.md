# Source-attributed NFL player injury and QB research crosswalk

**Research only:** No verified starter, no player availability assumption, no numeric scoring adjustment or sportsbook bet follows from this document.

The original free football-first model emits props under GSIS-style player IDs. ESPN's public injury snapshot describes named athletes, sometimes with an ESPN athlete ID. Reporting an injury about a person with a matching *name* alone is not enough for a deterministic individual-player model update.

The web-context companion now:
1. Captures `athlete.id` from public ESPN injury feed as `espn_athlete_id`, without upgrading reported status to official final inactive status.
2. Obtains free `nflreadpy.load_players()` crosswalk with `gsis_id`, `espn_id`, name and position. This is **not** an independently verified live depth chart or final starting lineup.
3. For a given game, joins IDs only if the original research player ID agrees with a **unique** GSIS↔ESPN ID map and the injury is explicitly in that player's original game **team**. Duplicate/missing ID maps and duplicate ESPN injury entries remain ambiguous. No fuzzy name matches.
4. Lists every original player in forecast distributions and their explicit status: ESPN reported injury (not final), source unavailable, no ESPN crosswalk, unmatched in snapshot (NOT confirmed healthy), ambiguous identity, etc. Scores and player-prop probabilities remain untouched. Leading QB is ranked by *historical simulated passing attempts only*, not declared the game-day starter.
5. Records original forecast capture, later supplement capture, kickoff, source URLs and eligibility checks. No future-issued or postkickoff reports can backfill an original frozen forecast.

The score/prop research and injury/news/NWS context JSON stay in the same read-only GitHub Actions artifact. The same tool still works if the public player-directory endpoint fails: it logs the failure and emits a clearly unavailable player-source review, never guessed status.

For actual pregame adjustments: review official team/NFL inactives or the full independently read text of ESPN/CBS/NFL reports with datestamps, ascertain role uncertainty, and estimate effect coefficients on development games with independent chronological confirmation. Source trust doesn't establish coefficient sign/magnitude or expected value.

Free references: https://nflreadpy.nflverse.com/api/load_functions/ (load_players ID mappings), https://nflreadr.nflverse.com/reference/load_players.html (GSIS primary and ESPN ID), and public ESPN injury feed source https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries. Do not invent a current game's officially confirmed starting roster.
