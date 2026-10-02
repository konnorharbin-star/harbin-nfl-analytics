# Stage 45 — Fixed non-QB personnel context validation

Stage 45 tests the remaining major Phase 2 football-context family that has not yet
received a leak-free historical predictive gate: non-quarterback personnel availability.

The current operational context layer already reports offensive-line, skill-position,
defensive, and starter injury risk. Those values remain diagnostic only. Stage 45 asks
whether the same information, reconstructed before each historical game, can improve
the independent fair-score margin or total.

## Scope

Quarterback information is deliberately excluded. QB state already has its own fixed-spec
Stage 15 validation and Stage 16 forward shadow.

The candidate uses only:

- non-QB starter injury risk;
- offensive-line starter injury risk;
- skill-position starter injury risk;
- defensive starter injury risk.

Sportsbook prices are never read by this research path.

## Point-in-time reconstruction

For each completed regular-season target game:

1. determine the scheduled kickoff in UTC;
2. discard injury updates reported after kickoff;
3. retain only depth-chart state admissible by timestamp or target week;
4. join injury severity to the pregame depth chart;
5. calculate team availability risk;
6. build the canonical fair-score baseline from games completed before the target week;
7. calculate margin and total residuals.

Games without a usable kickoff or both teams' depth state are excluded rather than
filled with future/current information.

Legacy weekly depth charts are labeled as weekly temporal evidence. Modern snapshots
with source timestamps are labeled timestamped. Season-only depth rows are tracked
separately and cannot be silently represented as timestamped.

## Source gate

The audit reconstructs 2022–2025 regular-season Weeks 5–18. Every season must recover
at least 90% of eligible games before predictive model selection is allowed.

If any development season misses the coverage gate, the result is
`SOURCE_INSUFFICIENT` and both margin and total remain disabled even if partial-data
metrics appear favorable.

## Fixed predictive gate

Margin and total are evaluated independently.

Predeclared feature families are:

- starters;
- position groups;
- all non-QB personnel signals.

The same feature family and ridge alpha must be used in every rolling 2023, 2024, and
2025 development fold. A nonzero specification survives only if:

- MAE improves in every fold;
- RMSE improves in every fold;
- aggregate MAE improves;
- aggregate RMSE improves.

Baseline / zero adjustment remains an admissible result and wins automatically when no
fixed specification clears all conditions.

## Evidence boundary

2022–2025 is development evidence. A surviving specification is at most eligible to be
frozen for a separate 2026 SHADOW ledger.

This audit always keeps:

- `canonical_score_adjustment_enabled=false`;
- `promotion_eligible=false`.

Historical results alone cannot alter the canonical fair score.


## Live-source result

The real nflverse audit completed successfully. Historical personnel state was
reconstructed for **831 of 831** eligible regular-season games in Weeks 5-18:

- 2022: 207 / 207 games (100%); legacy weekly depth charts;
- 2023: 208 / 208 games (100%); legacy weekly depth charts;
- 2024: 208 / 208 games (100%); legacy weekly depth charts;
- 2025: 208 / 208 games (100%); timestamped modern depth charts.

The source gate therefore passed. The candidate failed for predictive reasons.

### Margin

No fixed personnel specification cleared all three rolling folds.

The best tested specification was:

- feature family: `starters`;
- ridge alpha: `0.1`;
- positive folds: 2 / 3.

It improved both MAE and RMSE in 2023 and 2024, but failed in 2025:

- 2023 MAE: 9.9986 baseline -> 9.9730 adjusted;
- 2023 RMSE: 12.8739 -> 12.8293;
- 2024 MAE: 9.9842 -> 9.8685;
- 2024 RMSE: 12.9657 -> 12.7910;
- 2025 MAE: 10.6069 -> 10.6497;
- 2025 RMSE: 13.1996 -> 13.2302.

Even though aggregate best-tested error was slightly lower, the predeclared every-fold
gate rejects the candidate. Margin remains `disabled`.

### Total

No fixed personnel specification cleared all three rolling folds.

The best tested specification was:

- feature family: `groups`;
- ridge alpha: `100.0`;
- positive folds: 1 / 3.

It passed 2023, failed both metrics in 2024, and failed MAE in 2025. Total therefore
remains `disabled`.

## Decision

Stage 45 creates **no new 2026 personnel shadow**.

The canonical fair score remains unchanged. Non-QB personnel availability continues to
be published as operational context/risk information only.
