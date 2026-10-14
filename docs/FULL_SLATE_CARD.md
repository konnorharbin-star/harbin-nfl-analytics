# Full-slate NFL discovery card

Each scheduled or manually dispatched **Football-first NFL coupled totals and props research** run now forecasts the upcoming week, refreshes free public game prices across the next 192 hours, and evaluates every supported, priced contract against the original forecast distribution. The JSON and Markdown cards are uploaded in the run artifact; the per-game card appears in the GitHub run summary.

Every target game appears, including explicit missing-price, missing-model, stale-price, or replay-blocked statuses. Moneyline, spread and total comparisons use exact offered thresholds and both sides. One best research candidate appears per game; the complete slate is ranked by raw conditional EV proxy. This ranking does not establish a calibrated market edge. The current coupled model remains unvalidated, so qualified bets remain empty and every research candidate retains PASS plus concrete blockers.

Exact prop contracts can be supplied with `--prop-contracts path.json`. Supply an array of objects containing game_id, team, player_id (GSIS), market, side, line, american_odds, book, observed_utc and source_url; optional original-book-origin metadata is retained. Supported player metrics are passing_yards, passing_attempts, passing_completions, passing_touchdowns, passing_interceptions, receiving_yards, receptions, receiving_touchdowns, rushing_yards, carries and rushing_touchdowns. The corresponding player distribution must exist in the original forecast. The default free collector supplies game markets only; prop-price coverage is reported as NONE or PARTIAL, never full coverage. Unpriced props do not become bets.

Run locally after generating the forecast and replay:

```bash
python -m scripts.weekly_public_nfl_odds --season 2026 --hours 192 --max-games 64 --refresh-schedules
python -m scripts.nfl_full_slate_card
```

Outputs: `reports/nfl_full_slate_card.json` and `.md`. JSON records exact source prices, raw win/loss/push probabilities, model and price capture times, original forecast hash, conditional EV arithmetic, qualification blockers and coverage errors. A 15-minute observation freshness limit is operational integrity, not proof of bookmaker-origin freshness. No synthetic -110, cross-line blending, forecast rewriting, price-derived core predictions or automatic execution is introduced. Before pricing, deterministic replay must reproduce the original distribution exactly. Unknown OT, activation, settlement and calibrated probability uncertainty remain explicit. A larger positive raw number cannot override release gates.

This adapter currently covers NFL coupled research runs. It does not upgrade the separate CFB model or older model entry points, and does not claim validated edge discovery until independent market validation succeeds.
