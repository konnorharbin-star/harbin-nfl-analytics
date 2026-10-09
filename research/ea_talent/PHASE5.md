# Phase 5 — acquisition / roster crosswalk status

As of 2026-10-08, EA publicly exposes Madden NFL 27 and College Football 27 **browser-visible** player tables:
- https://www.ea.com/games/madden-nfl/ratings
- https://www.ea.com/games/ea-sports-college-football/ratings

Public browsing is **not** permission for mass extraction/redistribution, and a searchable page is not a documented free bulk CSV download or public API. No actual EA player exports or point-in-time historical archives have been imported by this PR. Do not claim an accuracy improvement or flip the live feature flag.

## Roster matching adapter

Create an EA-approved/exportable CSV with `ea_player_id,player_name,team,position,ovr`, retaining source provenance. Create a separate first-party roster export with `model_player_id,player_name,team,position` from your existing point-in-time roster dataset. Confirm team naming aliases and positions before matching. Run:

```bash
cd research/ea_talent
python ea_crosswalk_review.py ea_export.csv existing_roster.csv crosswalk_review.csv
```

Every suggested match stays `UNREVIEWED`, including perfect names. Verify identifiers, duplicates, current team, transfers and position before creating the effective-dated `ea_import.py` crosswalk. Do not treat browser result ordering or player names as a permanent player ID.

## Feasibility

The immediate next data dependency is a permission-compliant CSV/source feed from EA or a source whose license permits this use. Until available, tests run against local fixtures only. Phase 4's evidence gate must continue to report BLOCKED, with zero contribution to wager decisions. Official CFB team ratings are a separate optional *team-level* research prior and must not be misrepresented as player ratings.
