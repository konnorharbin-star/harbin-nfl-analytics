# NFL Step 6 — NFL-native context integrity and matchup forensics

**Research-only. No live bets are re-enabled, score correction is authorized,
market-only gates are overridden, or stakes are changed.**

The existing NFL model already has a historically audited fixed QB model,
a current expected-starter resolver, live injury/depth/roster ingestion, and
historically audited schedule/rest/travel and non-QB personnel regressions.
Historical context coefficients for schedule and non-QB personnel **failed
their fixed multi-season predictive gates** and therefore stay disabled.
This phase hardens source semantics and makes the resulting live context
observable and prospectively auditable; it does **not** simply add a
handicap boost for a quarterback or an injured offensive lineman.

## Source integrity defects corrected

### Injury designation overrides practice

Previously the injury severity parser checked `"active" in status`
*before* checking unavailable designations. Consequently
**Inactive / Full Participation** incorrectly produced severity 0.
**Out / Full Participation** could also become healthy. These are
serious discrepancies when selecting expected QBs and measuring starter
injury risk.

The parser now interprets official **Inactive, Out, Suspended, Injured
Reserve** as severe unavailability, **Doubtful** as high uncertainty,
**Questionable** as uncertainty, and only then considers Probable,
explicit active/healthy, or practice-only DNP/limited/full reports.
For example, Questionable + Full Participation remains Questionable,
not fully cleared. Numeric risk semantics are conservative diagnostics,
not validated point-spread values.

### League freshness is NOT team-specific availability

Previously a timestamp-fresh injury report belonging to any single
team could mark an entirely **missing opponent injury report** as
fresh, because absent teams inherited the league-level status.
The context builder now treats a team without an admissible injury
report as **UNKNOWN**, preserving an explicit per-team source coverage
field. A fresh league feed does not prove an unreported club has a
clean injury report. This state remains a betting freshness veto.

The current public nflverse Week 5 injury feed has injury rows
but lacks usable original report timestamps. That source cannot be
called timely, regardless of the time the API download completes.
This upgrade **does not invent timestamps** or treat `refresh=True`
as a verified injury-report publication time.

## Game-level forensic register

The downstream context layer now writes an auditable **one row per
game** rather than three unrelated lists for moneyline/spread/total.
It checks that all market candidates for one game share the exact
same source context values (otherwise its research readiness is blocked).

The register includes:

- Expected home/away QB IDs, last-starter changes, depth-based starter
  confidence and explicit non-ready reasons
- Per-team injury, depth-chart and roster freshness and the injury
  feed's original timestamp source state
- Starter/OL/skill/defense **availability-risk** scores from current
  depth + injury joins, with evidence-suppressed flags when personnel
  inputs are stale
- Observed rest days, travel distance, timezone shift, neutral-site
  and weather/venue evidence coverage
- Matched **own OL availability minus opposing defense availability**
  as a *descriptive paired injury-risk difference*, NOT a pressure,
  sack-rate or matchup-win probability estimate
- Observable stacked conditions (QB starter change alongside OL
  injury stress; away short rest alongside long travel), clearly
  labeled as **untested research hypotheses**
- Named missing-source blockers and separate descriptive context
  flags; weather availability does not create a false wind adjustment

The predeclared descriptive flags use OL risk >= 0.20, skill/defense
risk >= 0.25, <= 6 rest days, >= 1500 away-travel miles and >= 2
timezone hours. These are *not* discovered profitable cutoffs or
eligible trading rules. A clean game row is labeled
`OBSERVED_RESEARCH_ONLY`, **never** `BET` or
`production_authorized`. A feed/identity/time discrepancy yields
`BLOCKED_CONTEXT_PROVENANCE`, with explicit reasons.

## Freeze first context before kickoff

`history/context_forward_snapshots_v1.csv` freezes the first
available, source-time-consistent **2026 pre-kickoff** context for
each game, even if the injury and roster readiness is blocked.
Newly downloaded or revised depth charts and injury reports cannot
rewrite its first observation. Historical games are not reconstructed
with today's player state. Unknown/future source times cannot be
stored as legitimate pregame state.

This snapshot allows subsequent independent forward analysis of
whether a source was actually available before kickoff, whether a
QB replacement was anticipated, whether injury/OL source coverage was
present and whether a matchup hypothesis deserves validation.
A current state cannot retroactively be assigned to earlier seasons.

## Model operations and output files

Each successful NFL operational run adds:
- `outputs/context_intelligence_report.json` and
  `docs/context_intelligence_report.json` — coverage, readiness,
  underlying source status, named blockers and descriptive stress counts
- `outputs/context_risk_register.csv` and
  `docs/context_risk_register.csv` — one game per row, source and risk
  evidence, no speculative betting probabilities
- `history/context_forward_snapshots_v1.csv` — append-only
  first admissible pre-kickoff game context, keyed by game

The generated `outputs/README.md` links the live reports; the
canonical model JSON includes `meta.context_intelligence`. The new
diagnostics run **after** football fair score, sportsbook market
comparison, QB/context vetos, portfolio allocation, Step 1 edge
triage and Step 2 timing research. The context audit cannot feed
back into or overwrite any previous calculation.

## Promotion gate

Do **not** promote an injury adjustment, OL coefficient, travel
factor or opponent-mismatch ranking based on these live flags.
An NFL-only **timestamped forward** sample with appropriate source
coverage, fixed hypotheses, independent future outcomes and a
new out-of-sample baseline comparison is required. The earlier
negative historical schedule/personnel tests stay binding. Any new
research candidate must separately establish positive incremental
value before another release/reliability review.
