# Professional bettor information-value shadow

**Research-only. No automatic wagers or production weight changes.**

This prequential shadow compares individually published professional betting
selections against contemporaneous, exact-contract, same-book two-sided no-vig
pricing. It is an incremental validation layer to the existing independent
Walters-style models, Peabody-style probability testing, Andrews/Spanky
line-shopping, and Koerner/PropBetGuy usage research. RAS, Fornelli, Bearman,
Monahan and other source-complete published selections can be entered for
independent assessment. This is not their proprietary forecasting software.

## Required evidence and safeguards

The default research/professional_signal_shadow.json contains no fabricated
picks. For each real individually frozen pick require record_id, analyst,
league (NFL or CFB), event_id, kickoff_utc, published_utc, captured_utc,
source_url, market, period (FULL_GAME), participant, selection, line, outcome
(WIN, LOSS, PUSH, PENDING), and market_pair. Nonpending outcomes require
settled_utc strictly after kickoff. Event and contract must be unambiguous.

Supported markets: GAME_MONEYLINE, GAME_SPREAD, GAME_TOTAL,
PLAYER_RECEIVING_YARDS, PLAYER_RUSHING_YARDS, PLAYER_RECEPTIONS,
PLAYER_PASSING_YARDS. Moneyline line is NA; other lines must be half-points,
because this binary shadow is deliberately not a discrete push-probability
model. Spread line is HOME_SIGNED for both HOME and AWAY selections.
Only exact matching thresholds, participant, period and game count.

market_pair requires bookmaker, region, snapshot_id, quote_utc,
captured_utc, source_url, and sides (two exact-contract records each
containing selection, line and american_odds). A snapshot also must carry\nmatching event_id, market, period and participant. Same two-sided selection\nlabels or line numbers alone cannot prove same-contract comparability. Market\nquote time must be after pick publication, and its capture must precede kickoff.\nSettled outcomes must include outcome_source_url (HTTPS) in addition to\nsettled_utc; the original event_id must map to a consistent kickoff.
Provide optional independent_model_p and independent_model_frozen_utc
only when the independent model prediction was frozen BEFORE the
professional published the selection. Never fabricate missing odds.

## Algorithm

For each pick, compute no-vig probability from the same-book two-way quote.
Only that analyst's previous WIN/LOSS picks from games SETTLED BEFORE the
current post may affect the expert signal. Require at least 25 prior
settlements across 20 separate events. Fit a regularized logit increment
from a predetermined [-1,+1] grid. If prior chronological log-loss does
not improve on the market baseline, keep a zero weight. Compute future
paired log-loss and Brier differences between market and expert-adjusted
predictions; also evaluate the frozen independent-model probability.
Even after 20 scored expert-offset decisions, require at least 20 DISTINCT
games with a learned offset. Report a deterministic 2,000-draw GAME-cluster
bootstrap upper bound for expert-minus-market Brier difference. Adjust the
one-sided upper confidence tail for the number of analysts using Bonferroni,
and permit the incremental-claim diagnostic only when the adjusted upper
bound is strictly below zero. Different props on the same game are CORRELATED
and cannot substitute for new games. These gates detect some false claims,
but do NOT establish executable +EV, eliminate expert publication-selection
bias, or justify actual wagering.

The software ALWAYS returns PASS/bet_authorized=false. No analyst picks
can overwrite the main model automatically. Selection bias, source-complete
publication, independent market price validation, calibration, execution
prices and untouched forward grading remain necessary before model promotion.

## Execute (free Python standard library, pytest for tests)

    python -m pytest -q tests/test_professional_signal_shadow.py
    python -m scripts.professional_signal_shadow --league NFL --input research/professional_signal_shadow.json --output reports/nfl_professional_shadow.json
    python -m scripts.professional_signal_shadow --league CFB --input research/professional_signal_shadow.json --output reports/cfb_professional_shadow.json

The GitHub Actions workflow runs when inputs/code change on main and can
be launched on demand; it publishes research-only result artifacts.
Source collection is not automated, and absence of records is explicitly
NO_REAL_SOURCE_RECORDS rather than invented performance.

## Audit correction (2026-10-12, candidate v4.25.1)

The first shadow audit only required 20 selected contracts, not independent
games, and market_pair did not carry explicit event/market/period/participant
IDs. Those gaps could make cross-game odds or correlated same-game picks look
like market-controlled expert skill. This change rejects mismatched identities,
requires official outcome provenance, enforces internally consistent kickoff,
and adds game-cluster, multiplicity-adjusted evidence checks. It changes
**research software eligibility only**, not forecast parameters or BetMGM
investment status. Existing source registry is empty; there are no expert
performance claims, account-price verifications or qualified betting picks.
