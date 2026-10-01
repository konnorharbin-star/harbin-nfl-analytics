# Stage 10 — Current market source diagnostics and research fallback

## Live-source finding

The GitHub-hosted operational runner receives HTTP 403 from the ESPN NFL scoreboard endpoint. The denial persisted across the existing client profile, a standard browser User-Agent, and a browser-style profile with JSON Accept headers and ESPN Referer. Season-pinning the request did not change the result.

The same live audit inspected the nflverse 2026 Week 4 schedule frame. All 16 upcoming games carried non-null values for:

- `home_moneyline` / `away_moneyline`;
- `spread_line` plus home/away spread prices;
- `total_line` plus over/under prices.

The inspected schedule schema did not expose sportsbook identity or a quote-update timestamp for those fields.

## Fallback contract

`nfl/schedule_market.py` normalizes those nflverse schedule fields as `nflverse_schedule_snapshot` rows only when a verified live source does not already cover the same game/market.

They are **research-only**:

- they can support model-vs-market probability/edge diagnostics;
- they do not count as a verified sportsbook;
- they do not count toward verified ML/spread/total release coverage;
- their policy/allocation signal is forced to `PASS`;
- execution validation rejects them;
- they are excluded from `history/market_snapshots.csv`;
- they cannot create CLV or promotion-quality forward betting evidence.

The model therefore gains useful current market context during an ESPN outage without relabeling an unattributed schedule field as an executable quote.

## Source priority

1. ESPN public current source, when reachable.
2. Optional The Odds API multi-book source, when configured.
3. nflverse schedule market fields for missing game/market pairs, explicitly research-only.

A verified source always takes priority for a game/market pair. The fallback never overwrites or manufactures sportsbook breadth.

## Release behavior

`market_coverage.moneyline/spread/total` continues to count only verified rows. Research fallback coverage is reported separately as `research_moneyline`, `research_spread`, `research_total`, and `research_market_coverage`.

Consequently, full research comparison coverage can coexist with a failed verified-market release gate. This is intentional.
