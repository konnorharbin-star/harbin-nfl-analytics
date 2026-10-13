# v4.28.0 — Prospective NFL prop research journal rerun integrity

**Software correctness upgrade; research-only, no BetMGM value or model calibration claim.**

## Verified pre-existing failure

The v4.26 exact public prop quote join processed ALL accumulated original quote candidates against its current wall-clock time. Once the first registered player's game kicked off, the next invocation raised "quote observation cannot be joined after kickoff", even if the identical candidate had already been validly joined, graded by the same frozen CDF and protected in an immutable journal before kickoff. The v4.27 manual source intake had the parallel flaw: every original raw source entry was required to be within 10 minutes of the current run, even after prior successful archiving. These paths would make forward multiweek records un-rerunnable or tempt an operator to delete or change historical evidence.

## Correction

- At first eligible real pregame join, v4.26 journal now stores the SHA256 of the EXACT original candidate source JSON. Before EVERY later run, the complete historical journal is loaded and checked for duplicate IDs and original hashes. Each still-present source record must have the same byte-canonical JSON SHA; if unchanged, **reuse the first registered result without changing its first-seen date, bookmaker line, odds, or model forecast**, including if the game's kickoff is now past. Any new candidate still passes all old as-of cutoff, exact event, GSIS player, two-sided quote and catalog hash checks before it can be saved.
- v4.27 manual intake independently stores the SHA256 of the original raw human-transcribed source observation inside its hash-protected candidate. Later ingestion verifies both historical hashes, matches old raw quote_id to the immutable first raw SHA, and only prepares fresh IDs with the current <=10 minute human observation and pregame cutoffs. Repeated past original is allowed; changed old source or late new original remains rejected. Missing original raw lineage requires explicit audit, not blind migration.
- No forward source inputs or original quotes had been registered before this migration: original source input research/nfl_prop_manual_evidence_2026.json and downstream candidate research/nfl_prop_market_quote_candidates_2026.json both remain records: []; original exact research journal had no real quotes. Existing 698 frozen original primary synthetic props and 4,886 related threshold forecasts remain untouched.
- Both exact quote and manual-source reports now distinguish historical source rows verified from genuinely new prospective eligible candidates. Updated regression fixtures exercise replay AFTER kickoff, unchanged original byte-content, tampering of original odds, attempted postkickoff new quote ID and normal empty-source PASS. First original can only be made before kickoff; no re-originating model after outcomes.
- No scraping, odds provider subscription, bookmaker account execution, bet or EV estimation introduced.

## Core invariants

1. Old valid record: passed strict ex ante registration; original source hash intact; replay succeeds months later with same original source and join.
2. New record: the CURRENT time must precede kickoff, human source must be <=10 minutes old, and model forecast must have predated source collection.
3. Changed old record: raise exception even if it shares the old quote_id; never silently overwrite or re-date.
4. Not-yet-joined past source: reject; missing bookmaker original quote time stays null; empty source produces zero comparisons.
5. First-party BetMGM offer, stat-row/participation/void settlement, calibrated market advantage, executable betting EV still UNVERIFIED; trading decision PASS.

### Verification

Run focused tests: python -m pytest -q tests/test_nfl_prop_exact_public_market_shadow.py tests/test_nfl_prop_manual_evidence_intake.py

GitHub PR and CI details are in the live Google Doc methodology after verification. Software status remains RESEARCH_ONLY, profitable registry EMPTY.
