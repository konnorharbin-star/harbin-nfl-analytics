# NFL first-seen independent-score forward journal — version 4.10.0

Date: October 11, 2026 (UTC), model **research-only**, betting action **PASS**.

The objective is to make profitability research *prospective* rather than repeatedly discovering attractive historical scores after outcomes are known. This is a **data integrity upgrade**, not new predictive accuracy or evidence of a profitable BetMGM NFL edge.

## Working implementation

- [scripts/freeze_nfl_score_forward.py](../scripts/freeze_nfl_score_forward.py): verifies a current projection has the required canonical football margin and total, joins NFL schedule identity and exact timezone-aware kickoff, rejects already-started games and missing kickoff data, and adds only events without any existing first-seen record.
- [tests/test_freeze_nfl_score_forward.py](../tests/test_freeze_nfl_score_forward.py): synthetic regression suite checks no post-kickoff additions, no overwriting frozen event scores, signed margin arithmetic, missing/duplicate event rejection, impossibly large projected margins, schedule mismatch, after-final blocks, source SHA requirements and no fabricated EV.
- [GitHub workflow](../.github/workflows/edge-lab-score-forward.yml): automatically checks source and model, freezes current independent score outputs and commits newly eligible events to repository history. Scheduled free CI twice daily during Sep–Dec; actual scheduler timing can be delayed. Future years must have their season/version reviewed, not silently treated as 2026 validation.
- [Permanent history](../history/score_forward_2026.jsonl): append-only-in-code event-first-seen research records; a new run **does not overwrite an existing game**.
- [Successful first cloud run](https://github.com/konnorharbin-star/harbin-nfl-analytics/actions/runs/38185758447), created with source SHA `ea97bb0025c7889224913c3f368b91b5f0966462`, all workflow steps including tests and git persistence verified **SUCCESS**.

### Actually frozen at 2026-10-11T23:50:45.661474Z

| Event | Date/time UTC | Frozen home margin | Frozen full-game total | Mean-score inference | Known limitations |
| --- | --- | ---: | ---: | --- | --- |
| BAL at ATL | 2026-10-12T00:20Z | -0.8445508518 (away BAL favored slightly) | 47.7702948251 | ATL 23.46, BAL 24.31 | Previous-week BAL QB proxy **L.Jackson**, but confirmed starter is **Tyler Huntley** according to team injury reporting. **UNRELIABLE / not usable for BetMGM selection**. The historical journal preserves the original wrong-context input for accountability instead of silently repairing it. |
| BUF at LA | 2026-10-13T00:15Z | +0.4618004526 (home LA marginal favorite) | 50.0138287981 | LA 25.24, BUF 24.78 | QB proxies J.Allen/M.Stafford are NOT independently verified official active starters for this snapshot, and injuries, weather, odds and lineup confirmations are not certified. No positive EV claim. |

The frozen record is a first-seen record **within this new research journal**, not a claim it predates every earlier archived prediction in the project's Google Drive forward ledger. In fact, an earlier dated BUF–LA game-level PASS observation in Google Drive already documented essentially this score. That record remains intact. We do not relabel an older score as newly discovered edge.

No exact accepted U.S. BetMGM spread/total/odds or two-sided quote source time has been verified by this workflow. It computes **no betting probabilities, CLV or EV**. The workflow does not place real or paper bets and cannot validate Jason Cooper's Pikkit claim or reproduce a private algorithm.

## Why this improves the research

Profitability must be shown on forecasts that existed **before the game**. The next test will cross-reference a truly contemporaneous, same-contract sportsbook line (not an archive-final 2025 line) to an independent probability distribution and compare expected win/push/loss to subsequent *real* settlement. Model-vs-line score gaps are not themselves positive EV.

For every market, preserve source time, quote-capture time, original selection, price, market period, line, bookmaker region, a selection strategy pre-registered before observing outcomes, QB roster state, weather issue-time, and null estimates when missing. Don't compare different spread thresholds as the same no-vig contract. Delayed/scraped editorial bookmaker prices are NEVER automatically account-executable.

**Decision: hold core ratings and probabilities unchanged; maintain first-seen research journal, retain QB blocker and stake=0; no verified positive-EV BetMGM candidate.**
