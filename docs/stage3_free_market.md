# Stage 3 Free NFL Market Archive

The NFL model now follows the same free-data philosophy as the NCAA model for historical sportsbook research.

## Historical source

The primary free archive is nflverse/nfldata. The regular schedule dataset already carries archived market fields for NFL games, including:

- away/home moneyline;
- spread line;
- away/home spread price;
- total line;
- over/under price.

The separate public `initial_lines.csv` file is used when it contains an opening spread or total for a game. No API key is required.

The Odds API remains optional. It can still be used for richer multi-book and timestamped historical observations, but the model no longer depends on a paid historical plan in order to run an NFL backtest.

## NCAA-style archive behavior

The backtest reconstructs the football projection first. Sportsbook data is attached only afterward.

For each game/market:

1. use a free archived opening line when one is available;
2. otherwise use the nflverse archive-final line as an explicit fallback;
3. use the archived executable price when available;
4. use `-110` only for missing spread/total prices, matching the CFB backtest convention;
5. do not invent a moneyline if none exists;
6. select at most one modeled side per game/market;
7. grade the selected side at the archived price.

The nflverse schedule field `spread_line` is a favorite-margin representation: positive means the home team is favored and negative means the away team is favored. The adapter converts that field into the bettable home handicap by reversing its sign.

## Probability boundary

The score distribution is trained only on earlier chronological football-projection residuals. A target week cannot contribute to its own probability calibration.

That gives the research path:

```text
pregame football history
-> independent fair score
-> chronological residual distribution
-> free archive line/price
-> model probability + EV
-> historical grading
```

Sportsbook prices never enter the fair-score fit.

## CLV proxy

As with the NCAA archive, free-data line movement is labeled a proxy rather than an official timestamped closing-line claim.

When a distinct opening observation exists, the report can compute opening-to-archive-final movement. When no distinct opening exists, the archive-final line is the execution fallback and the CLV proxy remains missing instead of being manufactured.

For spread/total markets, the proxy is reported in side-aware line points. For moneylines it is reported only if distinct opening moneyline prices are actually present.

## Validation / holdout

The free archive backtest can tune probability-edge and EV thresholds on one validation season and then freeze the selected rule before evaluating a later holdout season.

A positive archive holdout is research evidence only. It does not bypass the platform's release gates, forward/shadow requirements, market-breadth requirements, or execution controls.

## Run it

```bash
python run_free_market_backtest.py \
  --start-season 2022 \
  --end-season 2025 \
  --validation-season 2024 \
  --holdout-season 2025 \
  --refresh
```

Outputs:

- `reports/free_market_predictions.csv`
- `reports/free_market_bets.csv`
- `reports/free_market_backtest.json`

The same command is available as a manual GitHub Actions workflow so historical research can be reproduced without a local Python environment.
