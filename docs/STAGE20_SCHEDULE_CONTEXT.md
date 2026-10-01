# Stage 20 — Historical schedule and venue context

Stage 20 tests a timestamp-safe context layer after multiple PBP residual and structural score alternatives failed their rolling gates. It deliberately avoids current injuries, live weather, and sportsbook information.

## Candidate signals

Every target game uses only information available before kickoff:

- home-minus-away rest days;
- short-rest asymmetry and combined short-rest load;
- deterministic away-team travel distance for non-neutral games;
- deterministic home/away timezone shift at the game date;
- neutral-site indicator;
- combined travel burden where the deterministic non-neutral route is available;
- combined rest deviation from a standard seven-day cadence.

Neutral-site travel is not guessed from the nominal home team. Those games receive an explicit neutral indicator and a travel-availability flag instead.

## Canonical baseline

The comparison baseline is the current independent fair-score model, including the already validated prior-season weight of `0.10`. Sportsbook prices never enter either the baseline or the context candidate.

## Chronological gate

Development rows are reconstructed for 2022–2025 regular-season Weeks 5–18. Expanding test folds are 2023, 2024, and 2025; each fold trains only on earlier seasons.

Margin and total use one predeclared feature set each. Ridge penalties are limited to `1.0`, `10.0`, and `100.0`. A nonzero candidate survives only when the same feature set and ridge improve both MAE and RMSE in every development fold and in aggregate. Baseline / zero adjustment remains admissible and wins automatically if no candidate clears the gate.

## Live-source result

The real nflverse audit completed successfully and verified the required schedule fields, including `home_rest`, `away_rest`, and `location`. The walk-forward dataset contained 831 rows across 2022–2025 and 624 evaluation games across the 2023–2025 folds.

Neither target cleared the fixed gate:

- **Margin:** selected adjustment `disabled`; best tested ridge `100.0`; positive folds `0/3`; baseline MAE `10.1966` versus best-tested `10.2702`; baseline RMSE `13.0138` versus best-tested `13.0624`.
- **Total:** selected adjustment `disabled`; best tested ridge `100.0`; positive folds `0/3`; baseline MAE `10.4926` versus best-tested `10.5185`; baseline RMSE `13.4427` versus best-tested `13.5101`.

The candidate therefore failed for predictive reasons, not because of missing source data or an engineering failure. No schedule-context shadow candidate is created and the canonical fair score remains unchanged.

## Evidence boundary

The 2022–2025 seasons are development evidence only. `canonical_score_adjustment_enabled=false` and `promotion_eligible=false` are hard-coded. A surviving historical candidate could justify a newly frozen prospective 2026 shadow ledger, but historical results alone cannot change the canonical fair score.

The live-source audit also fails closed if nflverse schedules do not provide the required rest and location fields. Missing deterministic team geography is represented as unavailable rather than fabricated.
