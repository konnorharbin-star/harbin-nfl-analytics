# BetMGM Edge Lab — Active-Market Scope v4.18.0

Decision date: 2026-10-11 America/Chicago / 2026-10-12 UTC
Classification: **methodology/scope update only; no predictive model coefficient change, no profitable betting strategy approved**.

## Active research scope

**NFL only, two independently validated workstreams:**

1. **Pregame full-game two-way MONEYLINE:** each team's win probability with potential tie/push mass explicitly identified. Price exact offered U.S. BetMGM moneyline sides; do not present merely the predicted winner as a bet. Alternative three-way moneyline only after separate contract specification and settlement support.
2. **NFL PLAYER PROPS:** starting with receptions, receiving yards, rushing yards and passing yards, over/under. Price exact game/player GSIS ID/stat/period/line/side, with eligibility, participation and void status as distinct dimensions.

NCAA football is deferred until NFL workstreams show independent forward reliability. **Spreads and totals are out of active bet-discovery/model-optimization scope**. Historical observations, rejected experiments, old workflows and original predictions stay intact; spreads and totals MAY remain explanatory team-strength features, historical reference prices and frozen control cohorts. Do not delete them or use such inputs with future information.

## Moneyline: planned independent probability engine

- Data: nflverse schedules (game ID, scores, team, rest, venue, historic home and away moneylines) and as-of team/player statistical features. The archived moneyline field is useful for historical benchmark/outcome testing, NOT a guaranteed time-stamped executable U.S. BetMGM wager or opening price.
- Independent prior: Elo/power ratings plus calibrated QB/starting personnel, situational and opponent-adjusted EPA, strength of schedule, injuries, rest, travel and home/neutral-site effects. Each feature must exist at prediction timestamp. No postgame injury/status/weather leakage; omit/mark unknown. Start with simpler market-free Elo/logistic and test nested temporal extensions rather than assuming complexity improves scores.
- Train chronologically; fit any parameters only on past games. Validate across unseen future seasons/weeks with calibration/reliability bins, Brier score, log loss and grouped confidence intervals. Compare against both favorite-always accuracy and market no-vig probability baseline using identical available-before-kickoff decision times. Also report moneyline game win accuracy, but never optimize win accuracy instead of expected return.
- Freeze P(home win), P(away win), P(tie/push) as independent pregame forecast alongside hash, model version and feature timestamps. Require sum to one and proper settlement of tied NFL games; BetMGM explicitly says a two-way NFL ML ties and voids with principal returned (https://sports.betmgm.com/en/blog/nfl/what-happens-in-nfl-betting-for-a-tie-game-bm06/).
- On a separately evidenced BetMGM account-offered exact price, EV of $1 stake = P(win) * b - P(loss), where b is net profit per dollar. P(push) returns stake (net 0). Fractional Kelly/staking NOT activated without independent calibrated robust profit and account confirmation; no automatic bet.
- Only consider model-selection after a truly future chronological evaluation including transaction quote quality, no-vig market calibration baseline and pre-registered multi-test adjustment. Comparisons of different moments' lines are not clean profitable-edge tests.

## Player props: planned exact-contract probability engine

- Preserve all original pregame 2026 v1-v2 frozen grid shadow forecasts for their registered forward Brier comparison; no rewrite, refit or interpolation to claim a frozen model contract.
- Prioritize frozen **full pregame player-stat distribution** (CDF/PMF, identity, model SHA, timestamps, participation/probable workload, schedule, position, injuries, projected snaps/carries/routes/targets). A full distribution frozen before offered quotes can be queried for the bookmaker's actual half-point line later without revising the pregame distribution or pretending a new line was previously pre-registered. Include P(over), P(under), P(push) and DNP/void conditional treatment.
- The current code is only an **editorial exact-line diagnostic** and is not an account-executable BetMGM accepted quote or market-probability edge model. It does not independently verify GSIS crosswalks.
- Moneyline and player props get separate model calibrations, market baselines, ROI/paper ledgers, failure gates and release decisions. A player prop can depend on game script but must not be double-counted as independent evidence when correlated with moneyline forecasts.

## Required research output

For every future game, rank research quality only, with separate MONEYLINE and PLAYER_PROP dashboards:
- event/kickoff/selection/exact line/period/participant (props only);
- source URL and timestamps for model forecast, odds offer, quote capture, kickoff;
- book and jurisdiction, whether BetMGM U.S. offer was actually accepted or merely editorial;
- P(win), P(push), P(loss), DNP/void treatment (props) with calibration and uncertainty;
- exact contractual price, implied and bookmaker-side no-vig reference where mathematically valid;
- estimated EV, conservative EV and uncertainty only when qualified by usable prices;
- **status PASS** when source, identity, chronology, independent market performance or participation is missing.

No claim of profitability without a pre-registered, prospective, truly tradable/replicable cohort that independently improves against a market-only baseline after vig. No paid APIs, subscriptions or execution.

## Retired/deferred work

- ATS surprise hypothesis REJECTED; totals RESEARCH ONLY, now PAUSED; old rejected results permanently preserved.
- Synthetic player prop research Brier cohort remains under automated forward validation, not wager selection.
- No forced plays; betting registry remains **EMPTY**.
