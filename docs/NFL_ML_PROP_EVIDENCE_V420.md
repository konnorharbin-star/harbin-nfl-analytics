# BetMGM Edge Lab v4.20.0 — NFL Moneyline and Prop Research Validation

**Updated 2026-10-12 UTC. Research-only.** NFL moneylines and player props only. Spread/total betting research paused. No validated profitable BetMGM edge.

## A. Moneyline: Elo does NOT improve historical archived market probabilities

Files: scripts/nfl_moneyline_archived_shrinkage_audit.py; tests/test_nfl_moneyline_archived_shrinkage_audit.py; .github/workflows/edge-lab-moneyline-market-audit.yml.
Successful cloud research run: https://github.com/konnorharbin-star/harbin-nfl-analytics/actions/runs/38205020652

Pre-registered market-Elo blend evaluated on 1,050 NFL games from 2019–2022 using log-loss; available development grid Elo weights 0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0. BEST ELO WEIGHT **0.00**; archived de-vig market probability selected 100%. Dev Elo log loss 0.65613698 versus historical market log loss 0.60864290. The 2023 chronological validation (already retrospective data, not pristine 2026) had 272 paired games; market-only Brier 0.21870951. Previously inspected 2024 (272 games) Brier market 0.20016162, 2025 (271) Brier 0.21213897. Historical archived moneylines lack first-party offer-origin UTC/account verification. Thus adding the simple Elo signal to the market is REJECTED; this is not evidence that a market-only betting strategy is profitable or that a real U.S. BetMGM quote was available.

NFLVERSE schedules moneyline fields: https://nflreadr.nflverse.com/articles/dictionary_schedules.html . No independent sportsbook accepted timestamp.

## B. Future original moneylines matched only to later captured public prices

Files: scripts/nfl_moneyline_forward_public_quote_shadow.py; tests/test_nfl_moneyline_forward_public_quote_shadow.py; .github/workflows/edge-lab-moneyline-forward-price-shadow.yml.
Cloud run SUCCESS: https://github.com/konnorharbin-star/harbin-nfl-analytics/actions/runs/38205140502 (10 unit tests). Reads existing first-seen moneyline forecasts and public Action Network JSONL capture journal. Same game and kickoff required. Only *collector* observations STRICTLY AFTER original frozen model and BOTH before kickoff can enter a descriptive conditional no-vig comparison. Aggregator snapshot SHA256 reverified, two-sided market/prices and BetMGM attribution checked, but account acceptance and original sportsbook price update time remain UNKNOWN; output always PASS with no true EV.

First run: **16 existing public observations preceded frozen moneyline model first-seen**, so rejected; ZERO after-freeze matched games. Daily scheduled shadow audit can pick up future independent source-capture records after the probability freeze if sources remain available. The test does NOT establish any bet.

## C. Player-prop original full-distribution catalog tampering now guarded

Updated scripts/nfl_player_prop_full_distribution.py recalculates and verifies the full frozen 2024 residual catalog SHA256 every time a full-CDF distribution is queried. Added tests for a tampered residual vector with an unchanged claimed hash in tests/test_nfl_player_prop_full_distribution.py. Original residual catalog remains unchanged SHA256: 2f6040e722592c7771d8a75d8022926dbff32b667626a87cf771093664cf8978. Successful cloud verification: https://github.com/konnorharbin-star/harbin-nfl-analytics/actions/runs/38205009748 .

## D. Frozen v2-versus-full CDF source reconciliation

New files: scripts/nfl_prop_original_consistency_audit.py; tests/test_nfl_prop_original_consistency_audit.py; .github/workflows/edge-lab-prop-full-cdf-consistency.yml.
Cloud run SUCCESS: https://github.com/konnorharbin-star/harbin-nfl-analytics/actions/runs/38205231460 (24 tests across full distributions and source consistency).

Input ORIGINAL immutable 2026 files:
- history/prop_participation_shadow_v2_2026.csv — 4,872 frozen 2026 v2 synthetic threshold lines.
- history/player_prop_full_distributions_2026.jsonl — 700 first-seen full player/game/stat distributions at audit time (initially 694, six more later frozen).
- history/player_prop_2024_residual_catalog_v1.json — 2024-fitted original residual vector catalog.

**4,858 exactly matched v2-versus-full player/game/market/line probabilities, ZERO differences above 0.00001**, 14 unmatched original threshold rows representing two player/game/market groups (UNRESOLVED, not silently passed). Confirms probability *representation* integrity on matched frozen contracts, NOT sportsbook profitability, DNP/void accuracy, model calibration, or guaranteed hits. No revision to either first-seen forecast journal.

nflverse reports recent-season detailed participation data are not updated throughout the regular season: https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html . Participation and void source limits continue to block actual prop bet approvals.

## Independent source option, unverified historic timestamp fidelity

The Prediction Tracker current NFL page https://www.thepredictiontracker.com/prednfl.html displayed Oct11 2026 7:59am local unspecified timezone with LA Rams home win probability 0.5748 vs Buffalo, published BEFORE newer Elo shadow freeze. This is **independent model consensus, not sportsbook odds or a tradable moneyline**. Archives https://www.thepredictiontracker.com/nflarchive.html have free per-season CSV links, but row-level as-of and archival publication chronology are not verified; no new model parameter may be promoted on that historical comparison.

## Model status / next gate

Research version 4.20.0 tracks reproducible better chronology, original CDF integrity, and rejection of worthless Elo weight, NOT a profitable betting model. Next: independent pregame QB/opponent-adjusted winner challenger; genuine contemporaneous market-only reference; original book offer timestamps; activity/void settlement; forward-clustered calibration; as-of paper trading and CLV. Original rejected history remains unchanged. Bookmaker acceptable quotes ZERO confirmed. Approved profitable model registry EMPTY. Wagers ZERO. Default PASS.
