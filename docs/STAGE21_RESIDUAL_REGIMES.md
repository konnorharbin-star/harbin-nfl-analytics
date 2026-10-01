# Stage 21 — Canonical residual-regime diagnostic

Stage 21 stops adding score corrections long enough to diagnose where the promoted fair-score baseline is systematically wrong. It is descriptive research only: no coefficient is fit, no correction is applied, and no sportsbook input is used.

## Fixed pregame regimes

The diagnostic classifies completed 2023–2025 games using only values available before kickoff:

- projected home-margin magnitude: `0–3`, `3–7`, `7+` points;
- projected total: under `42`, `42–48`, over `48`;
- season phase: Weeks `5–8`, `9–13`, `14–18`;
- rest asymmetry: home edge of at least two days, away edge of at least two days, or balanced;
- neutral-site versus standard home setting.

The baseline is the canonical direct-score model with the validated `0.10` prior-season weight. Schedule/rest fields use the same source contract already validated by Stage 20.

## Persistence rule

For each regime and for margin/total separately, the audit reports games, mean residual (`actual - baseline`), MAE, RMSE, and season-level results. A regime is marked as persistent only when all of these conditions hold:

1. every 2023–2025 season has at least 24 games in the regime;
2. mean residual has the same nonzero sign in all three seasons;
3. an aggregate 95% week-block-bootstrap interval for mean residual excludes zero.

Positive mean residual means the baseline underpredicts the target; negative means it overpredicts. The bootstrap resamples season/week blocks rather than isolated games.

## Evidence boundary

This stage does not select or promote a model. `canonical_score_adjustment_enabled=false` and `promotion_eligible=false` are hard-coded. Any persistent regime is a hypothesis for a later, separately specified football model; it is not itself evidence that adding a correction will improve out-of-time predictions.

The 2026 season is excluded. Existing frozen 2026 recent-form and QB-total shadow specifications remain untouched and are not retuned from this diagnostic.
