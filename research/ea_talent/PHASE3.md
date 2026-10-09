# Phase 3: lineup impact & shadow comparison

Both modules are research only. Neither imports or modifies the production model. The lineup CSV needs `player_id,team,role,observed_at`, where role is **starter**, **reserve**, or **out**, corroborated at or before forecast cutoff. Matching is against already validated model-player IDs in a dated EA snapshot. This does **not** guess starters using ratings or treat Madden OVR differences as scoring-point differences. Absence of an authenticated active lineup means availability impact is not identifiable.

Example from `research/ea_talent`:
```python
from ea_lineup_shadow import lineup_units, shadow_metrics

units = lineup_units(
    "ratings.csv",
    "verified_lineups.csv",
    prediction_at="2026-09-01T12:00:00+00:00",
    kickoff_at="2026-09-01T20:00:00+00:00",
)
print(units)
print(shadow_metrics("frozen_predictions.csv"))
```

Prediction comparison contract: `game_id,season,kickoff_at,forecast_at,base_margin,candidate_margin,actual_margin`. Each challenger forecast must be **generated and archived pregame** without retraining on outcomes from the evaluation period. Metrics are MAE and RMSE (pooled + by season). Better MAE on one sample does not authorize use in betting; maintain an untouched chronological holdout, evaluate probabilistic calibration and independently verified executable odds where available, test uncertainty, and retain a zero-weight release gate until validated.

Known limitations: injury gap uses the largest unavailable player's talent minus the highest confirmed reserve talent, not a causal impact estimate, and the current unit rollups do not yet include scheme, snaps, lineup substitutions, full 11-on-11 starter validation, or exposure-adjusted depth. The evaluator checks forecast cutoff but cannot independently prove forecasts were genuinely archived pregame. Treat external immutable logs as the authority.
