# BetMGM Edge Lab v4.27.0 candidate — Permission-aware manual NFL prop observations

## Purpose and constraint

The earlier v4.26.0 module has no accepted actual NFL player-prop quote records yet. This new stage converts HUMAN-OBSERVED, SOURCE-PERMITTED exact two-way player-prop prices to its verified input format, resolving the NFL GSIS player ID using immutable 2026 pregame player forecasts.

Public odds boards exist, but Edge Labs explicitly forbids scraping or redistributing its content without written permission (https://edgelabs.bet/terms). This software does not scrape anything, prohibits that source domain and its subdomains, and requires source-reuse permission attestation for OTHER domains. Check each source's terms before distributing its observations to this public repository. Public availability does NOT imply reuse permission.

## Human-observed input

Use research/nfl_prop_manual_evidence_2026.json with a top-level records array, initially EMPTY. Never insert synthetic test observations into actual inputs. Every manually entered record must supply quote_id (unique), HTTPS source_url, source_kind PUBLIC_MANUALLY_VIEWED or USER_VERIFIED_BOOK_SCREEN (with archival linkable source), source_reuse_permitted=true after independent verification, source_observed_utc (ACTUAL manual-observation timestamp), source_quote_origin_utc (explicit null unless bookmaker original update time was disclosed), snapshot_id, game_id, team, market, period FULL_GAME, player_display_name, bookmaker, book_region, numeric line, american_odds_over, american_odds_under. Optionally include independently verified GSIS player_id and source_publication_utc (editorial publication time, not a quote origin time).

Only allow source observation made within 10 minutes of executing intake, and only before kickoff and after original model forecast freeze. No late backfill. Input player identity must match game, team, market, exact surname, first name or first initial, and optionally GSIS ID in original 2026 frozen forecast. Ambiguous or absent players fail closed. Numeric lines are half-points or integers; both sides share exactly the same original snapshot and provenance. All sources remain manually-attributed, not bookmaker-authenticated.

The tool writes first-seen SHA-protected observations to research/nfl_prop_market_quote_candidates_2026.json. Existing IDs cannot be silently changed and it never overwrites old quotes. The downstream v4.26 research validator checks exact CDF contract, original 2024 residual catalog, times, source/line consistency and computes CONDITIONAL over/under/push probabilities, with no comparable model-vs-market delta or EV.

Execute:
    python -m scripts.nfl_prop_manual_evidence_intake
    python -m scripts.nfl_prop_exact_public_market_shadow

The read-only GitHub workflow runs focused synthetic tests and empty-record PASS checks; it does NOT fetch prices or start cron collection. The permission-restricted source, missing origin quote timestamp, unsupported participation/void settlement and accepted BetMGM U.S. account price remain blockers. An editorial market price, publicly viewed page or screenshot is NOT a filled account offer.

Status remains PASS / RESEARCH_ONLY, not an independently profitable player-prop model. All previously frozen 698 primary player anchors, 4,886 correlated research thresholds and rejected hypotheses remain unchanged. No paid data APIs, automatic betting or publication of forbidden source data.
