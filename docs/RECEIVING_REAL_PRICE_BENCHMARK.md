# Public web research and actual-price testing

The owner's current instruction supersedes the former EndzoneAPI setup.
**No API key, paid odds subscription, credit allowance or new account is required.**
The former authenticated fetch client is removed.

Every **NFL MODEL — Our Scores, Spreads, Totals and Player Props** run now:
1. Generates independent football forecasts from free public nflverse data.
2. Searches public Bing RSS for each game's football/roster/injury context,
   conditions/weather/travel, game/prop markets and historical-price leads.
3. Reads up to 16 pages from approved established publishers, saving source URL,
   capture time, raw-content hash and compressed original snapshots.
4. Discovers player pages on the public Covers NFL prop board and reads up to
   60 model-matched players, rotating across games and prioritizing receiving
   workload within each game.
5. Imports supported paired receiving/reception/passing/rushing prices into
   the full-slate research card and reports missing coverage for every game.

Standalone no-key Action: **NFL Web Research — Public Pages and Prop Prices**.
It replaces the former key-based receiving benchmark Action. Run locally:

```bash
python -m scripts.nfl_web_model_intake
```

It reads our named independent publication; it never borrows a publisher's
projected score or sets coefficients from a search snippet. Existing NWS weather,
ESPN injury and Google News research also continues. Public structured downloads
remain useful for historical football statistics; these require no provider key.
This implementation is NFL-specific.

A prop needs an exact displayed home/away matchup and Eastern kickoff matching
one model game, a unique model-player name match, and both odds at the same line
from the same bookmaker row. Missing counterpart prices, ambiguous identities,
past games and mismatched lines are excluded. Name/game matching is not an
independent source GSIS attestation or active-player verification. Search queries
use the game's Eastern calendar date; capture clocks are UTC.

Search/article text remains unverified evidence until structured facts are
checked. Reading an established publisher does not verify every assertion.
Search/page failures remain visible. The source page's capture time is not a
bookmaker-origin update time, and public prices do not certify account access,
locale, eligibility or settlement. No automated bets or automatic edge promotion.

Reports:
- `reports/nfl_web_model_intake.json`: per-game searches, article evidence,
  paired prices and missing-source coverage.
- `reports/nfl_web_prop_contracts.json`: exact informational prop contracts.
- `reports/web_source_snapshots/*.gz`: compressed raw public pages/search RSS,
  named by original content SHA-256.

The existing game-market first-seen validator keeps the best **game-market**
candidate, even when a prop leads the informational card. Original cohorts and
forecast implementation fingerprints remain intact. Retrieved props remain
research candidates under the same release gates.

The fixed receiving-yard historical benchmark still accepts an offline public-web
history export with `source_kind: PUBLIC_WEB_RESEARCH_HISTORY` and approved
`source_urls`. Its `responses` hold the existing normalized single-book player
history schema (`player_id`, `market: player_reception_yds`, `bookmaker:
draftkings`, `games` with `game_id`, `line`, `actual`, `over_odds`,
`under_odds`). Missing historical sides or prices must stay missing.

```bash
python -m scripts.nfl_receiving_market_benchmark --history WEB_HISTORY.json
```

It fits conditional residuals on 2024 with the previously selected eight-game
window and replays prior-week-only 2025 forecasts. That inspected year remains
diagnostic. Actual-line Brier/log loss/ECE and nominal-return proxies do not
establish executable profitability or a verified forward record. Broad historical
two-sided web price evidence remains incomplete; no false default -110 price or
synthetic threshold is substituted. Proven-edge count remains zero.
