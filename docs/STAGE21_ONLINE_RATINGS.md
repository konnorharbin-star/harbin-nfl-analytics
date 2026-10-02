# Stage 21 — NCAA-style online opponent-adjusted ratings

Stage 21 tests the largest remaining modeling-architecture difference between the CFB and NFL repositories. The CFB baseline maintains persistent opponent-adjusted offense/defense state and updates it after each game. The NFL canonical baseline instead refits a batch ridge model each week.

This stage ports the **architecture**, not the college coefficients.

## Candidate state model

Each NFL team carries persistent offense and defense state. Before a game:

- expected home points = league PPG + home offense - away defense + home field
- expected away points = league PPG + away offense - home defense

After the full weekly slate has been predicted, team state is updated from scoring residuals. The league scoring level is updated from that completed week's average team points. At a season boundary, team offense/defense is regressed toward zero and league PPG is regressed toward an NFL scoring prior.

The research grid is NFL-specific:

- update alpha: 0.04, 0.08, 0.12
- offseason carry: 0.40, 0.60, 0.80
- home-field points: 1.00, 1.75, 2.50

The league-week update rate is fixed at 0.08 for this architecture test.

## Leakage boundary

The weekly boundary is stricter than a naive sequential game loop. Every game in Week W is projected from the state at the end of Week W-1. No Week W result updates state until **all** Week W predictions have been frozen.

Sportsbook spreads, totals, prices, implied probabilities, CLV, graded bets, and stake outputs are not accepted by this module.

## Rolling gate

The 2021 and 2022 regular seasons are warm-up history. Development evaluation covers 2023, 2024, and 2025, Weeks 5–18.

Margin and total may select different fixed configurations. A side survives only if one unchanged configuration improves both MAE and RMSE in every development season and in the aggregate sample. Otherwise the canonical direct-score baseline wins automatically.

## Evidence boundary

Stage 21 is development research only. The evaluator hard-codes:

- `canonical_score_change_enabled=false`
- `promotion_eligible=false`

Any historical survivor may only justify a separately frozen 2026 prospective shadow experiment. Historical selection alone cannot alter canonical projections, probabilities, market decisions, or stake sizing.
