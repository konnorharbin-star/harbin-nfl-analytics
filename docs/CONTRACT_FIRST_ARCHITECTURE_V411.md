# BetMGM Edge Lab v4.11.0 — Contract-first research architecture

**Date 2026-10-11, research status only. No new validated U.S. BetMGM +EV strategy; no wagering.**

## Why the old score-first primary selection method is retired

The 2025 football-only canonical independent model spread MAE=10.606922, total MAE=10.483861 on 208 NFL weeks 5–18 games, worse than NFLVerse archive-final market margin MAE=9.992788, total MAE=10.081731. The market-blend regression selected zero independent football contribution for spread and total on historical 2023–24 development. External Elo and trailing score challengers failed consistent archive-market benchmarking. The old projected score remains a secondary football descriptive benchmark and an immutable first-seen forward research artifact; a score-minus-market mismatch MUST NOT itself generate a bet.

Existing `harbin_nfl_model.py`, `nfl.market`, historical reports and NFL projection functions remain preserved for regression and comparability. **They have NOT been rewritten or deactivated.** For BetMGM Edge Lab research requests, the **authoritative selection and eligibility contract** is the new market-first path below. This policy change does not assert a production-approved trading application.

## Primary new research components

- [scripts/contract_first_scanner.py](../scripts/contract_first_scanner.py) — independent exact-contract, two-sided bookmaker source check, no-vig **conditional-on-nonpush** market baseline, precise American price, third-outcome P(win/push/loss), full-match contract provenance, quote/capture time and freshness checks, per-market blockers, provisional-only EV diagnostics, and no-wager decision.
- [tests/test_contract_first_scanner.py](../tests/test_contract_first_scanner.py) — synthetic test scenarios; zero-snapshot PASS; no inferred push from no-vig; integer spread/total push math; half-point no-push rule; no joining different lines, different player IDs or books; refusing stale/postkickoff prices; price-only market baseline and selected-side model identity; models with huge hypothetical positive EV still cannot approve any bet. **Cloud CI SUCCESS:** [38196342164](https://github.com/konnorharbin-star/harbin-nfl-analytics/actions/runs/38196342164). Initial CI failed due two test assertions expecting first blocker position/phrase; the regression assertions were corrected, and cloud rerun passed. Failures are preserved in run history.
- [scripts/frozen_prop_contract_adapter.py](../scripts/frozen_prop_contract_adapter.py) — joins the existing [immutable first-seen 2026 player-prop statistical forecasts](../history/player_prop_probability_forward.csv) to a specifically observed two-way quote only on exact `game_id`, market, `participant=player_id`, period, `line`, selection and first-seen model availability **at or before quote time**. On synthetic half-point player props the frozen distribution has P(push)=0. No unverified book prices are synthesized and no bettor account access is performed.
- [tests/test_frozen_prop_contract_adapter.py](../tests/test_frozen_prop_contract_adapter.py) — positive and negative synthetic exact-player/time/line fixtures. **Cloud CI SUCCESS:** [38196452635](https://github.com/konnorharbin-star/harbin-nfl-analytics/actions/runs/38196452635). Includes the primary scanner tests and historical professional-method price audit tests.
- [docs/player_prop_probability_shadow.json](player_prop_probability_shadow.json) — original 2024 development and previously inspected 2025 player-stat synthetic-grid diagnostics. Source comparison: 2025 receiving-yards synthetic Brier 0.0826694 vs rolling baseline 0.0867300; receptions 0.1023001 vs 0.1056367; rushing yards 0.0757981 vs 0.0790409; passing yards same as baseline 0.1817817. **Synthetic threshold counts are correlated and are NOT bookmaker bets or independent opportunities.** Even positive Brier differences do not establish an actionable market error.
- Current contract scanner supports `GAME_SPREAD`, `GAME_TOTAL`, `GAME_MONEYLINE`, `PLAYER_RECEPTIONS`, `PLAYER_RECEIVING_YARDS`, `PLAYER_RUSHING_YARDS`, `PLAYER_ANYTIME_TD`, when provided as exact two-way contracts recognized by `professional_method_audit`. Frozen probability adapter currently connects only receptions, receiving yards and rushing yards from the archived 2026 prop file. `PLAYER_PASSING_YARDS` remains unsupported by the existing exact-contract audit's official market enumeration; the adapter deliberately skips passing yards rather than invent an unsupported instrument. Extending that market needs a separate validated source-and-settlement contract.

## Sources required for an actual scan

**One immutable quote snapshot per exact instrument**, with:
`event_id, kickoff_utc, market, period, participant, line, selection, american_odds, bookmaker, region, snapshot_id, quote_utc, captured_utc, source_type, source_url`. The two opposing sides need *the same* market/period/participant/threshold/book/region/source time. Spread lines are HOME_SIGNED when the selection is HOME or AWAY. With no quote-source update timestamp, the scanner must refuse to represent an offer as live; with a quote older than 15 minutes, it flags staleness. Account execution is always marked unverified and the event must remain in the future. No free feed currently certifies live U.S. BetMGM account-accepted prices, so the system cannot output a verified executable bet.

**Independent statistical probabilities** may be paired only with exact market and line plus timestamp before quote and kickoff. They must sum P(win)+P(push)+P(loss)=1.0 and obey impossible-push rules. The new scanner calculates the *research diagnostic*:
`EV per $1 = P(win) × (decimal price - 1) − P(loss)`.
A conservative diagnostic requires an independently justified lower bound on win probability, with remaining probability allocated to loss and push fixed. A raw deviance or huge hypothetical model EV is **not** a positive-EV approval. Explicit `APPROVED_MODEL_VERSIONS` is currently **EMPTY** until independently validated forward market and account-price evidence warrants a policy version upgrade.

## Operating commands

Market-first exact price audit: `python -m scripts.contract_first_scanner --input <verified_snapshot.json> --output reports/contract_first.json`.

Join frozen independent player probability shadows: `python -m scripts.frozen_prop_contract_adapter --quotes <verified_snapshot.json> --frozen history/player_prop_probability_forward.csv --output reports/frozen_prop_contracts.json`.

Both commands require a real input snapshot file; no source file is invented or downloaded from an unavailable provider. They print PASS when evidence is insufficient. The scanner has **no account execution or portfolio staking capability**.

## What remains before we could identify positive-EV trades

1. Secure legal, free, origin-time-verified U.S. sportsbook two-sided prop price data — BetMGM first, and synchronous sharp-market reference with same line. Verify actual account availability manually before any proposed execution.
2. Improve player participation/starter/QB state as-of, role/team adjustments, and derived player-stat distributions. Existing synthetic threshold tests provide **research prioritization**, not evidence the market is wrong.
3. Register independent chronological and untapped prospective 2026 validation on market-aligned exact contracts, bankroll-independent ROI/CLV with vig/push/void adjustments and clustering/multiple-testing uncertainty.
4. Model versions only become APPROVED through signed/persisted research policy after independent replay; no automated bets and no compulsory selections.

**Change decision**: Contract-first replaces score-gap betting recommendations *within this project's research policy*. Legacy score prediction code and archived output remain unchanged as a benchmark. Probabilistic research curves are connected but NOT promoted for betting. All unverified candidates PASS.
