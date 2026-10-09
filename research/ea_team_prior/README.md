# EA 27 preseason team talent pilot — FREE / research only

This is a distinct, independently sourced **team-level rating prior**, not the player-level EA import in existing draft PRs #152 / #48. The EA player-rating source/identity historical availability constraints remain unresolved. This project does **not** add EA team rating scores directly into production projections, adjust point spreads, or increase betting exposure.

## Official publications
- NFL / Madden NFL 27: https://www.ea.com/games/madden-nfl/madden-nfl-27/news/madden-27-team-ratings — published July 31, 2026.
- CFB / College Football 27: https://www.ea.com/games/ea-sports-college-football/college-football-27/news/college-football-27-team-ratings — published June 23, 2026.

`team_ratings_2026.csv` has preseason offense, defense and overall EA team ratings transcribed from the official source (NFL: 32, CFB: 138). `source_metadata.json` records source and publication date, *not* a verified immutable first-seen content hash. The pages may have changed since their dated publication: independence of the original values cannot be fully established retrospectively.

This dataset applies only to the **2026 season**, to forecasts observed and games kicked off *strictly after* the publication day. It cannot be backfilled into historical NFL 2022–25 or CFB 2023–25 backtests.

## Matchup feature
Given the home/away team's EA offense and defense ratings:
- Home offensive mismatch: home offense minus away defense.
- Away offensive mismatch: away offense minus home defense.
- EA matchup differential: home mismatch minus away mismatch.
- EA total pressure: home mismatch plus away mismatch.

These values are EA *rating units*, NOT football points. No coefficient converts them to margins or totals in production. For example, Madden's published Dallas 87 OFF / 80 DEF against Tampa Bay 83 OFF / 79 DEF gives an EA differential of **+5 rating units toward Dallas**. The actual Buccaneers–Cowboys result does not become a retroactively adjusted model forecast.

## Run locally (no paid services)
```bash
python research/ea_team_prior/audit.py --sport nfl \
  --ratings research/ea_team_prior/team_ratings_2026.csv \
  --metadata research/ea_team_prior/source_metadata.json \
  --games reports/probability_forward_graded.csv \
  --output reports/ea_team_prior_audit.json \
  --rows-output reports/ea_team_prior_residual_rows.csv
```

Replace `--sport cfb --games reports/live_graded_predictions.csv` for college football. The workflow `EA Team Talent Shadow Audit` does this automatically after tests and uploads the report and per-game data as **30-day research artifacts**. Its permissions are read-only, so it cannot publish model modifications or generated-state commits.

## Research interpretation
- Residual is **projected minus actual**. A correlation with the EA prior is descriptive only.
- The archives contain timestamps but a standalone CSV is not independently immutable proof of pregame commitment.
- The CFB ledger can be a selected sample of suggested-bet games; it is **not a random all-games validation population**.
- Never call a team rating a confirmed player availability measure. Ratings must not be used to identify starters or infer injuries.
- Current EA data lack properly archived 2023–25 versions. To estimate a score weight, first preregister a challenger and freeze it before upcoming matchups; compare on an untouched future set with sufficient sample size, uncertainty, and paired scoring metrics.
- Any publication or sportsbook-edge recommendation still requires existing quality/release gates.

Next phase: inspect residual strata, source coverage, QB/OL availability *at prediction time*, and design a genuinely prospective score challenger only if the evidence supports it.
