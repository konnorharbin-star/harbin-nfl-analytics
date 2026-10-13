# BetMGM Edge Lab v4.30.0 — Prospective player-prop calendar readiness

This is **read-only forward research scheduling**, not a model-weight, probability, outcome-grade or betting-edge upgrade.

The canonical first-seen 2026 NFL full distribution, paired 7-line reference forecasts and one-line-per-player-game-market research anchors were originated before their registered games' kickoffs. The 2026 prospective grading policy requires (a) the ORIGINAL kickoff + at least 72 hours, (b) a completed official schedule, (c) full independently checked both-team official player-stat source coverage and (d) an explicit official matched player stat row; unknown player rows are NOT settled as UNDER, DNP or VOID.

## Problem

The model has an archive of 698 primary single-line anchors and 4,886 original same-player seven-threshold sensitivity forecasts, but its successful source-grade framework cannot determine actual predictive accuracy until enough games mature. Without an accurate forward timetable, it is easy to demand premature grading or read an empty grade journal as evidence that the model performed badly or well.

## Implementation

New scripts/nfl_prop_forward_maturity_readiness.py takes *only* frozen primary-anchor and original seven-line paired-CDF JSONL archives. Its no-outcome calendar runs the existing registered primary-anchor integrity validator: each market must have a unique original player/game/stat anchor with original SHA, exactly seven frozen half-point research thresholds, and ex-ante base-CDF-nearest-to-0.50 line selection. For each game, calculate earliest allowed temporal grading boundary = kickoff UTC + 72 hours, then classify it as upcoming, after kickoff but before 72 hours, or at least 72 hours since kickoff with OFFICIAL STAT SOURCE UNVERIFIED. Time alone NEVER qualifies the player for grading and does not imply official game completion. Count one prospective anchor per player/game/stat, not seven statistically independent bets; group confidence analysis requires 20 distinct games per market and multiplicity controls.

Report counts by market and game, next 72-hour boundary and earliest first allowed boundary. It explicitly keeps all real official source eligibility grades and betting settlement probabilities UNKNOWN, book prices and true EV NULL, decision PASS. No public download, use of target-week outcomes, roster inference or sportsbook link is included. The input forecasts remain byte-for-byte unchanged; new report is in reports/nfl_prop_forward_maturity_readiness_2026.json and GitHub Actions artifacts.

## Verification and release

Source-first code and regression fixtures live in scripts/nfl_prop_forward_maturity_readiness.py and tests/test_nfl_prop_forward_maturity_readiness.py, with .github/workflows/nfl-prop-forward-maturity-readiness.yml running isolated testing and the actual first-seen population. The nightly schedule is safe and read-only; no paid API, betting account login or automatic bet is available. The generated report itself is NOT a betting ledger decision: it is a calendar-only checkpoint. The immutable Forward Research Ledger retains all prior source captures unchanged.

No clean independent claim of BetMGM profitability or market mispricing is possible before independently sourced accepted U.S. prices, participation/void, and chronologically mature official outcomes. Status: RESEARCH_ONLY / PASS.
