# NFL pregame web-context research (bookmaker-neutral, no-cost)

This companion searches **current** public NFL context for every upcoming NFL game processed by the independent score/prop research scan. It is a **separate, timestamped JSON research artifact**, not part of fitting the football-only prediction.

### Inputs by evidentiary role

- **Injury roster snapshot** — public ESPN NFL injuries endpoint, a *captured feed observation*. Athlete, team, reported position and status are recorded as displayed. The collector does not claim an ESPN injury snapshot is a final NFL official weekly injury report, verify starter designation, or infer that missing athletes are fully healthy. See `nfl/injuries.py` for existing canonical timestamp/freshness handling; web research does not bypass it.
- **Searchable news** — Google News RSS search by full team name and injury / starting QB / inactive terms, seven-day publication window. Store timestamp, headline, RSS URL and publisher named in feed (max six results per team). **Search results are only leads** and do NOT prove the article text or an injury/starting QB statement. Google does not guarantee stable RSS API access. No free text news headline is parsed into a betting adjustment.
- **Weather** — NWS `/points/lat,lon` -> `forecastHourly`, for approximate US home-stadium coordinates already versioned in `nfl/weather.py`. Store hourly forecast covering scheduled kickoff, original generated/updated time, wind speed, temperature, precipitation %, and source URL. An issued forecast with timestamp AFTER the scan cannot be used. Indoor roofs are labeled N/A and neutral/international venue coordinates require separate identification. Weather coefficients are NULL, not fabricated.

### How it runs

```sh
python -m scripts.nfl_coupled_drive_player_research_run --season 2026 --out reports/coupled_drive_player_shadow.json
python -m scripts.nfl_web_context_research_run --forecast-scan reports/coupled_drive_player_shadow.json --out reports/nfl_web_context_research.json
```

The existing GitHub Actions workflow `.github/workflows/nfl-coupled-shadow-research.yml` performs these sequentially on its scheduled/manual run. Both artifacts are uploaded together; original independent predictions are never retroactively overwritten by an injury/news search. News or weather endpoints can be unavailable; missing data is labeled `SOURCE_UNAVAILABLE` rather than silently substituting an expected outcome. No paid API keys, bookmaker, external account, bet execution, or text-generation service is necessary.

### How to use the results appropriately

1. Review recent quarterback/lineup **leads** and obtain independent official report/game-day inactive confirmation before marking a starter as verified.
2. Use issued weather as a *dated context observation*, not a trained coefficient. Indoor/roof status must be verified if uncertain.
3. Compare evidence snapshots with earlier snapshots before updating source conditions; preserve source issue/capture times and do not reuse later game news in earlier historical fit.
4. Apply only independently estimated, chronological-holdout-validated contextual coefficients to a separate model challenger. No automatically computed point or prop modifiers on the canonical score path.
5. Publish missingness and market PASS; a news citation alone does not prove predictive advantage.

### Risks and limitations

Public ESPN endpoints and Google News RSS are not guaranteed interfaces. The code has bounded HTTPS requests, network timeouts, fail-closed source reporting, no arbitrary user URL fetching, and synthetic tests. Source publication times may not exist in the injury feed, and queries may miss news or surface false reports. Government forecasts have issue time, but stadium coordinates are approximate; non-U.S. games and neutral venues must be resolved separately. Current research implementation cannot establish that a reported starter is the actual starter and cannot authenticate a same-game prop wager or closing-line value.
