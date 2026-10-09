# Pregame player-availability verification — October 2026

This change hardens the existing `nfl/qb_current.py` selection and betting veto. It does **not** estimate the value of a QB or alter the independent fair score.

A normalized QB injury with `freshness_status=STALE/UNKNOWN` may not be used as definitive proof the player is out. It is also not treated as proof he is healthy: QB certainty is capped and `expected_qb_decision_ready=False`. Roster records must be current (`roster_freshness_status=FRESH`) and current depth-chart evidence must be fresh to certify a projected starter as bet-decision-ready. Unverified/future roster/depth data prevents confident decisions. The existing sportsbook PASS/zero-stake controls remain authoritative.

Fields `expected_qb_{roster,depth,injury}_freshness` are attached to each game's home/away context to aid audits. The `expected_qb_id` remains a **projected** starter, never a verified official game-day starter unless independent pregame lineup evidence is added. No predictions are retroactively changed after a final score. Regression tests cover stale injury-out reports, stale roster/depth and current fresh evidence.

A true starter-confirmation service must have source-captured timestamps, player IDs, team IDs, and actual pregame availability status. EA ratings do not confirm starters or injury statuses.
