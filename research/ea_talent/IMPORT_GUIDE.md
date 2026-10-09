# Phase 2: EA ratings archive and identity crosswalk

The importer is research-only. **No EA data has been imported yet**; it requires a lawfully obtained, permitted CSV export. No authenticated scraping, paid feeds, or invented historic ratings. Obtain permission/verify EA data usage terms before copying or redistributing rating datasets.

Raw CSV (example headers): `ea_player_id,player_name,team,position,ovr,awr,spd,str,agi,cod,inj`. EA player identifiers must be durable IDs, never just player names.

Crosswalk CSV: `ea_player_id,model_player_id,team,effective_from,effective_to`. Dates are timezone-aware ISO strings and end is optional/exclusive. Link the model_player_id to the existing NFL/CFB roster feed using verified identity and roster affiliation. **Do not** auto-match by name alone; transfers, duplicate names and traded players require reviewed mappings.

Metadata JSON requires: `source_url` (HTTPS), `snapshot_at` (the demonstrated publication/observation timestamp), `obtained_at` (when actually collected), `source_sha256` (hash of raw file), and `license_note`. Metadata format provides traceability; it does not establish truth of supplied timestamps. If earlier ratings snapshots cannot be verified, they are ineligible for historical backtests.

Call:
```python
from ea_import import normalize

summary = normalize(
    "raw_export.csv",
    "verified_crosswalk.csv",
    "provenance.json",
    "data/ea_ratings/nfl_2026-10-08.csv",
    as_of="2026-10-08T12:00:00-05:00",
    min_coverage=0.95,
)
print(summary)
```
Run from `research/ea_talent` or add that directory to `PYTHONPATH`. Coverage below the specified threshold fails; missing players are reported and **never inferred**. The output is immutable; duplicate, future, invalid, incomplete or ambiguous records fail. Normalize and validate before using `team_features`.

Limitations before promotion: the adapter currently uses rating-based top-N groups, **not confirmed starting lineups**, and must not price injuries or adjust scores. In particular, the `available` field should be left unset until a separately verified point-in-time active roster is available. Future work: verified player rosters/starting assignments, coverage reporting per position, backtests.
