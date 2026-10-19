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

Known limitations: only the QB unit calculates a confirmed QB1-out-to-named-starter rating gap, not a causal impact estimate. Other units return no injury replacement gap, and these research rollups do not yet model scheme, snaps, full starter coverage, or exposure-adjusted depth. The evaluator checks forecast cutoff but cannot independently prove forecasts were genuinely archived pregame. Treat external immutable logs as the authority.

## Phase 6 — independently sourced pregame replacement impact

The original `lineup_units` routine incorrectly compared a ruled-out QB1 with
the *highest-rated listed reserve*, which can be QB3 rather than the QB2 who
was actually named as the replacement. This is corrected. It requires the
pregame depth rank and the explicitly named starting replacement.

Required separate lineup CSV columns:
`player_id,team,role,depth_rank,observed_at,obtained_at,source_url,verified`.
Values of `role`: `starter`, `reserve`, `out`, `uncertain`. The
`verified=true` field records manual review but does NOT prove the source's
content or its acquisition time; preserve original provider records separately.

The QB difference is **pregame QB1 player talent minus verified QB2/current
starter talent**, in EA rating units. If QB1 is available, the reported
unavailable-to-replacement gap is **0**, not the normal QB1-vs-QB2 ability gap.
If QB1 is questionable, not identified, or not confirmed out, return
`None` and `qb_ready=false`. If the backup actually starting is not
identified, the gap is unknown. A verified uncertainty row never becomes
`available` merely because a Madden game durability attribute is high.

The previous `available` default of true for omitted rating CSV statuses is
also removed. Without an independently verified status, availability-sensitive
`team_features()` are `None`; raw talent ratings remain descriptive only.

No conversion of EA rating differences into football points has been learned.
No live model's score, betting odds, signals, or wager risk limits change.
Manual/legal data sourcing, player ID verification, independent archived
lineups and chronological paired outcomes remain necessary before promotion.
