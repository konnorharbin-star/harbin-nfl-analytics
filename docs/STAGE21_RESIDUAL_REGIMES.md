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

## Live-source result

The real-source diagnostic completed successfully on 831 walk-forward rows across 2022–2025, with 624 evaluation games across 2023–2025. Two predeclared regimes met the persistence rule:

- **Margin — away rest edge:** 100 games, with 29/39/32 games by season. Mean margin residual was `-3.0469` points overall and `-3.4503`, `-2.1800`, `-3.7380` in 2023/2024/2025. The week-block-bootstrap 95% interval was `[-6.0013, -0.6000]`. In these games, the baseline systematically overpredicted home margin.
- **Total — projected total under 42:** 148 games, with 72/45/31 games by season. Mean total residual was `+2.3396` points overall and `+1.4808`, `+4.6784`, `+0.9392` in 2023/2024/2025. The 95% interval was `[+0.5006, +4.2804]`. In these games, the baseline systematically underpredicted total points.

No other predeclared regime met all three persistence requirements. Neutral-site samples were far below the minimum-per-season requirement and are not treated as stable evidence.

These findings are diagnostic hypotheses, not score corrections. In particular, conditioning residuals on the model's own projected-total band can expose calibration or mean-reversion structure and does not by itself establish that a post-hoc total adjustment will improve future forecasts. The Stage 20 multivariable schedule-context candidate also failed its predictive gate, so the away-rest pattern is not sufficient evidence for a direct rest adjustment.

## Evidence boundary

This stage does not select or promote a model. `canonical_score_adjustment_enabled=false` and `promotion_eligible=false` are hard-coded. Any persistent regime is a hypothesis for a later, separately specified football model; it is not itself evidence that adding a correction will improve out-of-time predictions.

The 2026 season is excluded. Existing frozen 2026 recent-form and QB-total shadow specifications remain untouched and are not retuned from this diagnostic.
