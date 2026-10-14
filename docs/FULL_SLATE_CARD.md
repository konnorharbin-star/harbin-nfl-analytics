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

## Prospective validation of selected contracts

Each coupled run now saves the **first positive best raw-EV game contract per game** into the separate `history/full_slate_selected_v1_2026` research cohort. Later scans never replace that selection or count it again. Records retain the original exact contract, same-book opposite price when supplied by the collector, unrounded raw probabilities, forecast hash, model implementation fingerprint, actual checked-out code SHA, run ID and pre-kickoff archive time. A changed predictive implementation requires a new cohort. This is a research selection, not a paper or real bet.

Before each scan, the new grader reads free nflverse finals and evaluates original contracts only after 72 hours. It checks schedule/team/kickoff identity, scores the selected threshold, and compares model binary non-push Brier score against the same-book proportional no-vig non-push proxy. Pushes/ties are excluded from binary comparisons. Missing opposite prices do not become -110 and missing/malformed final scores stay unknown. Score corrections append revisions; unchanged grades are idempotent; summaries use currently valid source rows, not stale formerly valid grades. Multiple books and repeat scans never add independent game counts.

Economic ROI, CLV, bookmaker settlement and validation remain unknown. Source scoreboard grading is not sportsbook settlement, and no-vig normalization is a market proxy rather than calibrated truth. This observational cohort does not establish profitability or override release gates. The current report shows paired Brier comparisons without confidence intervals; a positive mean alone cannot promote a strategy. Automated exact prop outcome validation is outside this game-market cohort.

Both history writers use a shared concurrency group and check out current `main` when the queued job starts, rather than replaying stale history from the triggering commit. The recorded source code SHA is the actual checkout, so code provenance remains accurate. Other concurrent edits fail visibly through the existing rebase safeguards.
