# NFL v4.26.0 candidate — Prospective player CDF ↔ exact public prop market join

**RESEARCH ONLY. Zero actual BetMGM U.S. account quotes, accepted wagers, or proven positive EV.**

## Why this exists

Original 2026 pregame [full empirical player CDFs](../scripts/nfl_player_prop_full_distribution.py) can be evaluated at any integer or half-point prop line using immutable 2024 residual vectors. The earlier seven-line study and 698 registered single-line anchors are deliberately synthetic and cannot be read as offered market lines. This separate addition takes **future, source-observed, exact two-sided market quotes** and joins them by NFL event ID, GSIS player ID, team, statistic, full-game period and exact numeric line to an original CDF that was frozen BEFORE the actual research capture.

A paired-price snapshot is not a U.S. BetMGM account price. Its original bookmaker update time is usually **unknown**, and even a source-provided original time cannot be treated as independently authenticated by this module. The source-side no-vig conditional nonpush probability is descriptive. The CDF conditions on an **explicit official player stat row**, whereas the sportsbook settles by rules that depend on participation, voids, and sometimes different statistics. These targets are **not interchangeable**. The code intentionally leaves comparable model-vs-book delta, unconditional P(win/push/loss/void), estimated EV, and conservative EV as NULL for all records.

## The data contract

Populate [research/nfl_prop_market_quote_candidates_2026.json](../research/nfl_prop_market_quote_candidates_2026.json) with an array of original public observations **while the event has not kicked off**; never backdate an event or recreate a missed original after the game. Keep a first-party source link or a public quote-provider link and its actual collection timestamp. If source publication/update UTC is missing, pass source_quote_origin_utc as null; do NOT replace it with the capture UTC.

Each original observation requires:

- quote_id, game_id, player_id (**GSIS ID, not ambiguous name**), team, market (receiving_yards, receptions, rushing_yards, passing_yards), kickoff_utc, period=FULL_GAME, bookmaker, book_region, snapshot_id, source_url, source_observed_utc, source_quote_origin_utc (nullable), line (nonnegative integer or half point), sides.
- sides must contain exactly one OVER and one UNDER with identical line and integral American prices, both from the SAME provider/book snapshot. EACH side must repeat the identical snapshot_id, bookmaker, source_url, source_observed_utc and source_quote_origin_utc from the parent record; a mixed-source pair fails closed. Input-source attribution still requires independent audit before any acceptance claim.
- Optional source_origin_timestamp_claimed_verified describes a third party's claim; it does NOT mean independent book certification. It cannot be true if original quote time is null.
- Explicit account_price_accepted or actual_bet_placed TRUE is rejected in this capture-only pipeline. An actual account ticket must be tracked under a separate manual, authenticated flow and never inferred.

## Frozen methodology

First verify the exact original game/player/market/team/kickoff, the first-seen forecast precedes quote capture, and the research process itself executes before kickoff. Verify the immutable SHA of the 2024 residual-vector catalog and that forecast metadata references it. Evaluate the exact book threshold directly with pre-fitted 2024 normalized residuals; do not interpolate the fixed seven-line grid or refit on 2025 or 2026 outcomes. Track conditional P(over), P(under), P(push). Compute no-vig over using **the same-book** two-way implied probabilities, conditional on no push. Never compare this directly to an unconditional sportsbook winning probability without independently validated participation/void settlement, or claim it proves an edge. Quotes with unknown original time are marked CAPTURE_ONLY; those >15m old or predating the frozen forecast are separately marked as research-only provenance warnings. Every row is hashed and append-only with ID immutability.

Use:

    python -m pytest -q tests/test_nfl_prop_exact_public_market_shadow.py
    python -m scripts.nfl_prop_exact_public_market_shadow

GitHub CI tests the fixture guards and runs the CLI on the deliberately EMPTY quote file. No live prop feed, odds API, web scraper, bookmaker login, or paid source has been introduced. Populate only actually observed pregame quote records. Do not re-use after-kickoff articles as original observations.

## Predeclared next evaluation

Future source-valid and accepted-stat player cohorts require an independently verified offer & settlement-rule process. A paired market Brier/log-loss comparison must control for participation/void outcomes, same threshold, original quote timing, and game-cluster uncertainty; at least 20 distinct graded games per market and familywise correction are prerequisites, not proof of profitability. Do not compute or assert a profit, model skill, or ranking from the empty capture registry. Preserve existing 698 original anchors and the forward research ledger.

Status: **PASS / RESEARCH_ONLY**. Exact-contract joining software is a prerequisite for genuine market-relative validation, NOT evidence that profitable BetMGM pricing exists.
