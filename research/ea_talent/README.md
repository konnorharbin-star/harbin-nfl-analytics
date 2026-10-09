# EA player talent research adapter (OFF by default)

This directory supports an **optional and unpriced** Madden NFL 27 / College Football 27 talent challenger. Nothing here is imported by the live scoring, probabilities, actionability, or wager pipeline. No new bet is recommended by this feature.

## Data contract

Archive a legally obtained/exportable ratings snapshot as a CSV, e.g. `data/ea_ratings/2026-08-01.csv`. Do not silently scrape or overwrite older ratings. Columns required: `player_id,team,position,ovr,snapshot_at`; optional `awr,spd,str,agi,cod,inj,available`. All rows must share the *actual publication/observation instant* `snapshot_at`, formatted as offset-aware ISO 8601, **not the time it was later downloaded**. Ratings downloaded now are not evidence of what was available weeks ago.

`available` is research input (1/0) and must be independently verified as of the prediction cutoff. When historical availability cannot be reconstructed, do not backtest injury impact. Player identifiers and team labels must be crosswalked against the model's roster IDs before usage. Position sets and top-N group sizes are deliberately minimal research defaults and are not trusted starter/depth models.

```python
from ea_player_talent import load_snapshot, team_features, matchup_features
rows = load_snapshot("data/ea_ratings/2026-08-01.csv",
                     prediction_at="2026-09-01T12:00:00-04:00",
                     kickoff_at="2026-09-01T20:00:00-04:00")
scores = team_features(rows)
home_minus_away = matchup_features(scores, "HOME_TEAM", "AWAY_TEAM")
```

Run `python -m unittest discover -s research/ea_talent -p 'test_*.py'` from the repository root with PYTHONPATH including `research/ea_talent` or run from that directory.

## Promotion gate

1. Acquire immutable, legal, *time-stamped* source snapshots and verify player joins/lineup coverage.
2. Train position adjustment / points conversion **only** on preceding games, and compare with existing baselines on untouched chronological seasons/weeks.
3. Check score MAE/RMSE, probability Brier/log loss/calibration, injury subgroups, coverage and bet-level CLV on verified archived prices.
4. Promote only with independent forward/holdout support. Until then score adjustment = 0, production recommendations unchanged.

EA game ratings are subjective third-party estimates, not official NFL/NCAA measurements. Current EA pages: https://www.ea.com/games/madden-nfl/ratings and https://www.ea.com/games/ea-sports-college-football/ratings
