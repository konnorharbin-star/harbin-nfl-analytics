# NCAA-to-NFL Architecture Parity Matrix

This document records the intended relationship between `harbin-sports-analytics` (CFB) and `harbin-nfl-analytics` (NFL).

The goal is **platform parity, not coefficient copying**. The NFL repository should expose the same major operating contracts as the NCAA repository while using NFL-specific data, features, calibration, thresholds, context behavior, validation results, and release evidence.

## Parity contract

| Capability | NCAA reference | NFL implementation | Parity status |
| --- | --- | --- | --- |
| Data contracts and cache boundary | `harbin/contracts.py`, `harbin/data.py` | `nfl/contracts.py`, `nfl/data.py` | Equivalent contract; NFL-native schemas |
| Pregame leakage protection | chronological/as-of helpers | `pregame_history`, PBP/week cutoffs, timestamp checks | Equivalent |
| Independent fair-score model | CFB football model | `nfl/ratings.py` + validated NFL prior | Equivalent role; NFL-native model |
| Advanced football features | `harbin/advanced.py` | `nfl/advanced.py`, dataset/residual research modules | Equivalent role; different features where league data differs |
| Chronological walk-forward evaluation | CFB walk-forward stack | NFL walk-forward, nested residual, prior, probability and QB audits | Equivalent methodology |
| Probability calibration | CFB calibration/win-probability layers | `nfl/probability.py`, NFL win-probability research | Equivalent role; NFL calibration only |
| Quarterback treatment | context/advanced CFB state | dedicated NFL QB state/dataset/validation subsystem | NFL is intentionally more explicit |
| Historical market research | archive backtest/runtime | free nflverse archive + optional timestamped provider | Equivalent evidence contract |
| Current market source hierarchy | free primary + optional enrichment | ESPN primary + optional The Odds API | Equivalent |
| Line shopping | professional market layer | canonical distinct-book aggregation in `nfl/pro_market.py`/`market_intel.py` | Equivalent |
| No-vig probability / EV | CFB market math | `nfl/market.py` | Equivalent |
| Policy signals | PASS/LEAN/BET/STRONG | `nfl/policy.py` | Equivalent structure; NFL thresholds independent |
| Kelly sizing | capped fractional Kelly | `nfl/policy.py` | Equivalent structure; NFL settings independent |
| Execution validation | quote provenance/freshness/pregame checks | `nfl/execution_market.py` | Equivalent |
| Portfolio concentration caps | slate/game/team/market/book/time caps | `nfl/portfolio.py` | Equivalent |
| Drawdown/adverse-run throttle | bankroll risk state | `nfl/portfolio.py` | Equivalent |
| Injuries/personnel context | CFB context stack | `nfl/injuries.py`, `personnel.py`, `context.py` | Equivalent role; NFL-native sources |
| Rest/travel/weather/venue | CFB context stack | `nfl/weather.py`, `context.py` | Equivalent role |
| Context score adjustment policy | validate before promotion | NFL context is confidence/risk-only until NFL point-in-time holdout passes | Equivalent fail-closed rule |
| Forward market ledger | line-history capture | `nfl/line_history.py`, hourly workflow | Equivalent |
| Forward decision ledger | cap-constrained decision history | `nfl/decision_ledger.py` | Equivalent |
| Independent grading | live/shadow grading | `nfl/grading.py`, `grading_audit.py` | Equivalent |
| CLV evidence | pre-kickoff close comparison | forward snapshots + normalized historical CLV diagnostics | Equivalent role |
| Backtest uncertainty | block bootstrap diagnostics | `nfl/backtest_runtime.py` | Equivalent |
| Feature audit | training/live/leakage audit | `nfl/feature_audit.py` | Equivalent |
| Backtest audit | quote/provenance audit | `nfl/backtest_audit.py` | Equivalent |
| Portfolio audit | stake/gate/cap audit | `nfl/portfolio_audit.py` | Equivalent |
| Consolidated audit snapshot | system-level audit | `nfl/audit_snapshot.py` | Equivalent |
| Monitoring / drift | live-readiness monitoring | `nfl/monitoring.py` | Equivalent |
| Health report | operational health | `nfl/health.py` | Equivalent |
| Model card | machine-readable model contract | `nfl/model_card.py` + generated Markdown | Equivalent |
| Hard release states | RESEARCH -> PAPER -> SHADOW -> PRODUCTION | `nfl/release_gate.py` | Equivalent |
| Canonical operational pipeline | one model/ops path | `nfl/pipeline.py`, `harbin_nfl_model.py` | Equivalent |
| Human weekly board | CFB HTML/PNG picks board | `nfl/render.py` | Equivalent presentation pattern |
| Public docs bundle | latest state/audit/quant/run report | `nfl/publication.py` | Equivalent |
| Publication reconciliation | fail on stale/contradictory public state | `validate_publication.py`, canonical workflow check | Equivalent |
| Audit trend history | append-only run snapshots | `history/audit_snapshots_v1.jsonl` | Equivalent |
| CI / scheduled operations | test, model, line capture, grading, backtest | NFL CI/audit/model/line-capture/grading/backtest workflows | Equivalent or stricter |

## Intentional NFL-specific differences

These differences are required rather than parity gaps:

- NFL team count, schedule structure, roster rules, depth charts, injury reporting and venue data differ from college football.
- NFL quarterback state is a first-class subsystem because a single starter change can alter team identity materially; no NCAA coefficient or manual point value is copied.
- NFL fair-score weights, prior-season regression, residual features, probability dispersion and signal thresholds are selected from NFL evidence only.
- NFL historical odds availability differs from CFB. Archive-final fallbacks remain research observations and cannot be promoted as verified opening entries.
- Current market breadth depends on the actual sportsbooks returned by the configured NFL sources. Duplicate feeds of the same book count once.
- Historical and live NFL evidence must earn NFL release states independently. CFB ROI, CLV, calibration or sample size never transfers to NFL.

## Promotion rule

Structural parity is considered complete when the NFL platform can run the same operating sequence as the NCAA platform:

```text
pregame football data
-> independent fair score
-> calibrated probability distribution
-> current/historical market comparison
-> policy signal
-> capped Kelly proposal
-> execution validation
-> portfolio caps
-> forward ledgers
-> independent grading
-> audits / monitoring / health
-> hard release gate
-> reconciled machine + human publication
```

That sequence is now implemented.

What remains is **NFL evidence accumulation and NFL model improvement**, not additional copying of NCAA coefficients or conclusions. The NFL model should stay in RESEARCH/PAPER/SHADOW whenever its own release evidence says so, even if the NCAA model is further along operationally or empirically.
