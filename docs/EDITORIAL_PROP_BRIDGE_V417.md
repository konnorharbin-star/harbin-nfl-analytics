# BetMGM Edge Lab v4.17.0 — Exact-Line Editorial Player-Prop Research Bridge

Validated 2026-10-12 UTC. Research-only quote-to-frozen-model matching, NOT a profitable betting model.

## Shipped source

- scripts/first_party_prop_quote_bridge.py
- tests/test_first_party_prop_quote_bridge.py
- .github/workflows/edge-lab-editorial-prop-bridge.yml
- Successful cloud run: https://github.com/konnorharbin-star/harbin-nfl-analytics/actions/runs/38202249367

Cloud CI confirmed twelve passing unit regressions and 4,872 originally frozen v2 2026 synthetic player-stat threshold records read without modifying the first-seen journal. The real-history cloud smoke was deliberately run with no claim of executable price observations: 0 verified BetMGM quotes and 0 wagers.

The bridge matches exact event, GSIS player_id, player market, FULL_GAME, and exact half-point threshold. It requires original model capture before a known publication time to compute even a non-EV model/price difference. It checks paired over and under American odds and URL domain, disallows whole-number push thresholds, rejects duplicate contracts and invalid price/chronology. Source-asserted player ID crosswalk validity is NOT independently authenticated here. Editorial timestamps are NOT the original live sportsbook update time.

Any probability difference is a research-only unvalidated conditional nonpush no-vig reference difference; NEVER an estimated true EV or bet recommendation. All outputs PASS.

## Real editorial observation — no valid model comparison

Source https://sports.betmgm.com/en/blog/nfl/2-best-nfl-player-prop-bets-mnf-rams-vs-bills-week-5-bm16/
The webpage displayed Oct 11, 2026, 9:54 AM, no independently verified timezone or original bookmaker update time. Kyren Williams rushing O69.5 -115 and U69.5 -115; James Cook rushing O73.5 -115 and U73.5 -115. Frozen v2 rushing threshold grid: 19.5,34.5,49.5,64.5,79.5,99.5,124.5. Both offered editorial thresholds are NOT frozen model contracts. NO interpolation or made-up player GSIS ID. These editorial publication prices are NOT verified executable U.S. BetMGM account offers and their published author views are not official sportsbook price confirmations. Decision PASS for both, 0 independently verified model EV conclusions.

## Scientific controls unchanged

- v2 fit on 2024 stat-row population, with player participation/void unresolved and possible missing-player source rows.
- Previously examined 2025 diagnostic NOT a pristine independent holdout.
- Week5 Monday / Week6 original v1-v2 frozen cohort awaits future postgame grading and clustered validation.
- No current origin-timestamped U.S. BetMGM executable offers, prospective paper ROI or independently proven profitable player-prop edge.
- Approved profitable betting model registry EMPTY; model probability parameters UNCHANGED. No automatic wagering.

## Next objective

Collect exact offered player-prop lines from timestamped free public sources, verify player identities and eligibility and freeze independent probability at the actually offered threshold BEFORE kickoff, with original book offer time and as-of feature controls. Then compare against matched contemporaneous market baseline and independently evaluate prospectively. Do not treat dated editorial publication as a live accepted quote.
