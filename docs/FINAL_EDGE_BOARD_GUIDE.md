# Final betting edge decision board — production-grade separation of evidence and action

The NFL and CFB pipelines now write `outputs/final_edge_board.csv` and `outputs/final_edge_summary.json` on every successful model refresh. There is also a **free, read-only** `Final Edge Decision Board` GitHub Actions run for verification, testing and downloadable artifacts.

The aim is to identify *defensible* market opportunities, **not** to invent or guarantee profitable picks. The final board reads existing model probabilities and validated edge assessments: it does **not** alter forecasts, train coefficients, mark bets as placed or increase exposure.

## Decision definitions

- **BET_READY** — global model release eligible; timestamp-valid, currently fresh (at most 120 min), validated quote from an identified sportsbook; all market, holdout, context, probability and conservative-price criteria satisfied.
- **RESEARCH_CORE** — strong historical edge evidence exists but some essential requirement (e.g. independent provider timestamp or global release gate) is missing. **Not a betting recommendation.**
- **RECHECK_PRICE** — supported research idea with an anonymous, stale or otherwise unverified bookmaker price. Must re-shop and independently verify a current quote; do not assume the model's saved number can be bet.
- **NO_BET** — not supported or otherwise ineligible. A large raw model-vs-market EV is not enough.

Exactly one highest-ranked candidate per game appears on this summary, avoiding double-counting correlated spreads, totals and moneylines as independent opportunities. Ranking is deterministic by decision tier, then conservative price cushion and spread-line cushion; it **does not** optimize an in-sample composite score for past winnings.

## NFL

Source: `outputs/actionable_betting_board.csv`. A row needs a supported research edge tier, positive held-out/conservative EV, valid source/book/price, fresh QB/injury context, reliable regime and win probabilities, plus an approved model betting action. The authoritative `reports/release_gate.json` must be production eligible, otherwise it is not `BET_READY`.

The present NFL model's observed positive raw expected values have **not** demonstrated robust incremental forward value versus the sportsbook benchmark. Its release is blocked. The board should show `NO_BET` instead of trying to force an NFL selection.

## College football

Source: `outputs/edge_priority.csv`. A research core requires `ROBUST_CORE`, a supported subgroup, verified chronological historical holdout, `CONFIRMED` price evidence, positive conservative price EV and at least a half-point of remaining spread-line cushion. The conservative price lower bound uses historical Wilson evidence; it is **not** a calibrated game-specific win probability or proof of future profits.

An anonymized book such as `ActionNetwork book 69` is **not a sportsbook the user can necessarily access**, and a collector-observed timestamp is not independent evidence of the book's quote time. Those records remain `RECHECK_PRICE`/research-only. A future fully verified bookmaker quote must explicitly supply `market_execution_verified=true` and `market_quote_timestamp_verified=true`. The global release gate must also pass.

Historical subgroup support alone is not evidence that the 2026 live strategy wins; the existing forward edge validation file has zero clean graded selections. Do not claim live betting success.

## Running it

NFL:
```bash
python scripts/final_edge_board.py --sport nfl \
  --source outputs/actionable_betting_board.csv \
  --gate reports/release_gate.json \
  --out-csv outputs/final_edge_board.csv \
  --out-json outputs/final_edge_summary.json
```

CFB: use `--sport cfb --source outputs/edge_priority.csv`, with the same gate/output arguments.

The decision board runs **after** each model refresh and its publication validation, then joins the normal model output commit. It does not create extra GitHub writes, so it cannot race the grade/line-capture workflows as an independent writer.

## Acceptance

Tests enforce global no-bet release authority, quote freshness, rejected anonymized books, provider timestamp verification, conservative price margins, no post-kickoff picks, and game-level de-duplication. Do not weaken these conditions to make recommendations appear.

A bet remains a **candidate**, not an automatically executed wager. The user must verify the exact price, time and availability directly before staking. Even statistically supported opportunities can lose.
