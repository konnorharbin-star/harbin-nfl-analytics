# BetMGM Edge Lab v4.29.0 — NFL forward player-stat grade journal audit

**Release: research-only grading-integrity correction. No change to model forecasts, bookmaker prices, coefficient fitting or betting recommendations.**

### Verified weakness

Prior forward paired NFL prop grader ranked provider-stat grade revisions by largest revision number without requiring a cryptographically linked chronological correction history. The scorecard therefore could trust a latest row whose p_original/p_snap had been altered, whose binary OVER disagreed with its numeric stat, or which claimed grading before 72 hours after kickoff. A mismatch could make a challenger appear to outperform the originally frozen CDF.

### Fixed rules

For every formerly frozen exact player/game/stat/half-point line, v4.29 stores a SHA-256 on the first matured official-source grade. If a provider later returns a changed stat, append—not overwrite—a correction with consecutive revision integer, prior source value, prior grade SHA and new grade SHA. Validate the entire appended grade journal against the original immutable 2026 CDF SHA for that exact line before forward scorecard computation, including stable event/player/team/kickoff, original forecast UTC, and EXACT original versus snap CDF probabilities. Any missing original, duplicate revision, broken prior SHA, changed pregame prediction, nonfinite stat, impossible reception count, stat-vs-binary OVER contradiction, before-72-hour grading, missing both-team source population thresholds, or false sportsbook settlement claim fails closed before writing the new grade or scoring model accuracy. Missing player official-stat rows remain UNKNOWN, not UNDER/DNP/VOID.

The grader first commits prospective PREKICKOFF probabilities before querying any target-game outcome. It separately appends the 72-hour matured grade journal after the provider check. The full journal validator enforces correction integrity after writing, before Brier scores, clustered confidence intervals, and preregistered primary-anchor scores. Game-level bootstrap and four-market familywise intervals are unchanged.

SHA chains provide software tamper detection, **not authentication that a provider was honest**, and are weaker than external independent append-only time certification if an attacker could edit all prior evidence. The original provider's status, actual sportsbook account offers, active participation, no-action/void rules, official settlement and value remain separate unverified questions.

### Source coverage and release

Prior to this fix, no 2026 official 72-hour matured grade source had been recorded in history/player_prop_paired_cdf_line_grades_v1_2026.jsonl (file absent at code inspection). Thus implementation does not migrate or discard any genuine graded records. 698 frozen primary single-line prop anchors and 4,886 originally frozen correlated synthetic thresholds are preserved.

Tests in tests/test_nfl_prop_grade_journal_integrity.py cover changed 2024/2026 CDF identity, fake correct-looking SHA, impossible binary outcome, too-early grade, duplicate revision, broken prior SHA, provider correction lineage and missing-row NO GRADE; earlier cohort source/anchor tests retained. No paid APIs, no automatic bets, no U.S. BetMGM verified EV.

**Status: RESEARCH_ONLY / PASS, profitable registry EMPTY.**
