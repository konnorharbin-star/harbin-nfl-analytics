# BetMGM Edge Lab v4.15.0 — Participation-conditioned NFL player props

**Research completed 2026-10-12 UTC. Status: RESEARCH ONLY, NO VERIFIED POSITIVE-EV BETMGM BET.**

## New independent calibration challenger, run against actual historical source data

The prior prop residual distribution trained on earlier NFL forecasts whose players had no matching official stat row and assigned those missing rows actual value ZERO. It then applied probabilities to betting markets where nonparticipation may cause a void. This mismatched the underlying population. The isolated v2 research challenger now fits statistical errors to forecasts with a matching player/game official stat record, while leaving absent stat records unresolved instead of presuming a zero. Both methods use identical pregame player point forecasts and compare probabilities against the same player-game-stat observations; the difference in the paired study is how the historical residual distribution was trained.

Code: [participation_conditional_prop_v2.py](../scripts/participation_conditional_prop_v2.py). Unit tests: [test_participation_conditional_prop_v2.py](../tests/test_participation_conditional_prop_v2.py). Free GitHub Action: [edge-lab-participation-v2.yml](../.github/workflows/edge-lab-participation-v2.yml). [Successful live cloud run 38199796831](https://github.com/konnorharbin-star/harbin-nfl-analytics/actions/runs/38199796831) passed tests, fetched free nflverse data, replayed two earlier seasons and persisted the future challenger shadow journal.

2024 development sample: **8,870** forecasts with explicitly matching player-stat rows; **1,824** additional player-market forecasts whose official stat row was absent and which the legacy residual model treated as zero. Missing stats have unknown participation/void status, NOT 1,824 proven inactive players.

2025 already-inspected, **not pristine or promotion-eligible** paired diagnostic, Brier score lower is better:

| Stat | Legacy v1 with missing-as-zero calibration | Conditional v2 | v2 minus v1 | 95% game-cluster bootstrap CI of Brier change |
| --- | ---: | ---: | ---: | --- |
| Receptions | 0.11026560 | **0.10923497** | −0.00103063 | [−0.00197892, −0.00010275] |
| Receiving yards | 0.09163502 | **0.09129917** | −0.00033585 | [−0.00103430, +0.00033403] |
| Rushing yards | 0.08209499 | **0.08091200** | −0.00118299 | [−0.00192351, −0.00051414] |
| Passing yards | 0.20817986 | **0.20148878** | −0.00669109 | [−0.01346437, +0.00029371] |

All four show favorable research-only point estimates, but two CIs include zero, 2025 had already been inspected, and there were four comparisons. No evidence of positive EV relative to source-verified BetMGM props or account price. Probability calibration conditional on a weekly stats row is not yet a verified participating-player/void-contract probability.

**FIRST-SEEN, genuinely future evidence:** The cloud workflow saved **4,585 separate synthetic-threshold research-only forecasts** for Week 6, before kickoff, in [history/prop_participation_shadow_v2_2026.csv](../history/prop_participation_shadow_v2_2026.csv). These are correlated player-game-stat grid thresholds, not independent bets, and contain NO executable odds. This distinct v2 lineage does not overwrite [original v1 frozen forecasts](../history/player_prop_probability_forward.csv). Subsequent weeks can grade these original predictions prospectively against appropriately verified participation and outcomes; DO NOT re-fit on them before claiming independent validation.

## Actual GSIS-to-PFR ID and snap evidence audit

We also implemented [scripts/prop_snap_participation_audit.py](../scripts/prop_snap_participation_audit.py) plus [tests](../tests/test_prop_snap_participation_audit.py), cloud [workflow](../.github/workflows/edge-lab-player-snap-audit.yml) and a **successful actual-source cloud run [38199983573](https://github.com/konnorharbin-star/harbin-nfl-analytics/actions/runs/38199983573)**. Free authoritative nflverse sources:
- NFLverse players ID crosswalk: https://nflreadr.nflverse.com/reference/load_players.html ; GSIS player key explicitly linked to Pro Football Reference PFR ID. No player name approximations; ambiguous duplicate IDs rejected.
- Pro Football Reference snap count dataset via nflreadpy: https://nflreadr.nflverse.com/reference/load_snap_counts.html ; pairs are matched by exact PFR player ID, game ID and team.

Actual 2026 source loaded: **24,853** players crosswalk records, **6,064** PFR snap rows and **5,359** current weekly player-stat rows. The original v1 prospective grader withheld **532** missing-stat synthetic threshold rows, representing **76 distinct player/game/stat groups** across games already final; **602** other threshold rows were not yet eligible because game results/time were pending.

**All 76 distinct groups had no verified same-game PFR snap record**, despite the loaded ID crosswalk and an absence of ambiguous mappings or duplicated snap keys in the loaded data. Crucially, this is **NOT proof of DNP or void**. Snap source lag, coverage, roster changes, and eligibility remain possible. The audit has **zero** confirmed positive-offense-snap / no-stat-row cases among those 76, but the absent snap record itself cannot settle their under bets as losses or wins, nor justify imputing a zero.

The audit does not modify original frozen forecasts or original graded results, does not confirm an account price, and never places bets.

## Release decision

- **Do not promote** v2 to an approved wagering probability model. It has a favorable nonpristine historical paired signal and 4,585 newly frozen synthetic forward observations, but still lacks fresh prospective settled trade-grade performance, verified participation/void rules, and exact source-time U.S. BetMGM prices.
- Keep v1 and v2 distinct and immutable, treat zero implied outcomes only when participation and market settlement rules are independently resolved.
- Next major effort: identify true player participation using official/team status and PFR snaps when available and verify current player-prop quotes at the exact threshold. Grade v2 versus v1 on genuinely new frozen 2026 outcomes and market odds with pre-registered sample, game-week clustering, uncertainty and no forced bets.
- Do not use 2025 diagnostics or 2026 outcomes to retroactively tune the recorded v2 probabilities.
- **Betting status: PASS / no independently verified positive expected-value U.S. BetMGM opportunity.**

**v4.15.0 is a prospective model challenger and data-accuracy research upgrade, not a profitable strategy release.**
