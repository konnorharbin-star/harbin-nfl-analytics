# BetMGM Edge Lab v4.12.0 — Free sportsbook observation journal

**Date:** 2026-10-12 UTC (captured during evening 2026-10-11 U.S. Central)  
**Status:** RESEARCH ONLY. **No validated positive-EV U.S. BetMGM betting pick.**

## What is new (actually executed in the cloud)

[Validated cloud run SUCCESS](https://github.com/konnorharbin-star/harbin-nfl-analytics/actions/runs/38197435395), before Bills–Rams (kickoff 2026-10-13 00:15 UTC): passed synthetic quote-origin and capture-journal regression tests, fetched unauthenticated Action Network NFL scoreboard once via the already-existing `nfl/action_network.py` client, rejected 21/21 bookmaker-attributed two-way markets as *original-update-time-verified* (no row's source-origin timestamp), **but accurately recorded what 21 public market pairs showed at our direct collection time**. Cloud job preserved original observation on [GitHub in chronological journal](../history/public_aggregator_quotes_2026.jsonl). No paid API, automatic sportsbook action, fabricated odds, fabricated probabilities, or bearer token.

**Observed at:** 2026-10-12T02:51:53.317538+00:00 = **2026-10-11 21:51:53 CDT**. Record ID SHA-256 `4132a219c2db0367ade345fe697f53d1313b1585b88ed9b05b7f3d1e7538bac7`. Game `2026_05_BUF_LA`, Buffalo at LA Rams. Total 21 public two-sided market pairs representing seven sportsbook labels: BetMGM, DraftKings, FanDuel, BetRivers, PointsBet, WynnBET and ActionNetwork book ID 49 (identity unknown; never substitute a guessed sportsbook). Data are **not original bookmaker quote timestamps** and **not verified U.S. BetMGM user-account offers**. Record includes null `source_original_book_offer_update_utc` and `account_acceptance_verifiable=false`, and no current publicly timestamped prop quotes. The source can refresh asynchronously; source age at the time of collection is unknown.

### Direct captured BetMGM-attributed board (American odds)

| Exact market | Selection | Captured quote | Source provenance |
| --- | --- | --- | --- |
| Full-game moneyline | Rams | -185 | public aggregator only |
| Full-game moneyline | Bills | +150 | public aggregator only |
| Full-game home spread | Rams -3.5 | -102 | public aggregator only |
| Full-game away spread | Bills +3.5 | -115 | public aggregator only |
| Full-game game total | Over 54.5 | -110 | public aggregator only |
| Full-game game total | Under 54.5 | -108 | public aggregator only |

Two other actual attributed book examples in SAME captured public collection: DraftKings Rams spread -3.5 -104, Buffalo spread +3.5 -116, Over 54.5 -110 and Under 54.5 -110; FanDuel spread Rams -3 -110, Buffalo +3 -110, game total 53.5 -110. DO NOT compare FanDuel -3 / 53.5 probabilities as the same contract as BetMGM -3.5 / 54.5: different settlement events. Only one collection time is proved; quotes may be of different actual *age*. There is no independently certified consensus or executable arbitrage in this evidence.

## Code and protections

- [scripts/action_network_contract_feed.py](../scripts/action_network_contract_feed.py) plus [tests/test_action_network_contract_feed.py](../tests/test_action_network_contract_feed.py) — strict conversion to research-only contract scans when BOTH source quote-origin times exist, within 15min and before official kickoff; no cloned capture-time-to-origin timestamp; only exactly complementary same-book two-sided markets. Observed Action Network records fail this strict test because public API omits quote-update times.
- [scripts/public_quote_capture_journal.py](../scripts/public_quote_capture_journal.py) plus [tests/test_public_quote_capture_journal.py](../tests/test_public_quote_capture_journal.py) — separate immutable capture-only observations with exact source URL, original capture date, original bookmaker labeling, paired selection lines/prices, SHA, explicit original-time unknown, actual-account acceptance false; fails after kickoff and disallows same-event/same-time rewrite.
- [scripts/probe_free_nfl_contract_quotes.py](../scripts/probe_free_nfl_contract_quotes.py) — runs original-time strict probe AND capture-only journal from the same API response and records both separately.
- [Auto/scheduled GitHub workflow](../.github/workflows/edge-lab-free-source-probe.yml) — guard future scheduled runs to 2026-10-12, journal all later observed publicly available Bills–Rams quotes and commit chronologically, with a single-workflow concurrency guard. Scheduled runs are not guaranteed to occur at the exact requested minute. Existing GitHub artifact records each source-probe report.

### What a captured price does and does not prove

It **proves** that the chosen public source API returned a quote with that line and attributed book at our authenticated collection time, before scheduled kickoff. It does **not** prove that the underlying sportsbook just updated, that the price remains available, that BetMGM's U.S. account would accept it, or that a model-selected bet earns positive EV. Capture-only records are therefore **NOT admitted to the source-origin-verified scanner**; they can serve for descriptive market-history and prospective paper-price quality research with full uncertainty caveats.

The existing `history/player_prop_probability_forward.csv` holds frozen research-only player probability curves, but there are still **zero independent two-sided, source-time verified, BetMGM U.S. player prop offers** aligned by participant/line. Legacy score model remains a benchmark, not a model for bet authorization.

## Next real experiment

Study how attributed BetMGM full-game spread/total/moneyline prices change in successive captured snapshots. Compare same-contract prices (not mismatched thresholds) with competitors; test differences against actual independent prospective models, quantify calibration/uncertainty and slippage, and log FULL ORIGINAL screenshot/account price separately when the user verifies a BetMGM account offer. The current-source game research decision remains **PASS**, without a manufactured rank or wager.

**Status change:** v4.12.0 records measured improvement in *free source capture/market-data research evidence*, not in verified profitability, betting probabilities, or account execution.
