# BetMGM Edge Lab v4.25.0 — Prospectively Registered Single-Line Player-Prop Anchor Cohort

**Research checkpoint 2026-10-12 UTC. Scope NFL moneylines and player props. Research only; no approved BetMGM bet or quantified market edge.**

## What was actually changed

The previously frozen **4,886** original research-only player-prop synthetic thresholds are not independent. Each of **698** eligible pregame game/GSIS player/stat groups contributed **seven correlated thresholds**. Weighted all-line Brier can overweight player-stat distributions and distort any claim of incremental value. This v4.25 study registers exactly **one PRIMARY ex ante half-point anchor per player/game/market**, chosen using the BASE model only, before kickoff. The seven-line CDF remains available as **secondary sensitivity**.

- Source [scripts/nfl_prop_primary_anchor_cohort.py](../scripts/nfl_prop_primary_anchor_cohort.py); [tests/test_nfl_prop_primary_anchor_cohort.py](../tests/test_nfl_prop_primary_anchor_cohort.py); [workflow](../.github/workflows/edge-lab-prop-primary-anchor.yml).
- Primary score [scripts/nfl_prop_primary_anchor_score.py](../scripts/nfl_prop_primary_anchor_score.py); [tests/test_nfl_prop_primary_anchor_score.py](../tests/test_nfl_prop_primary_anchor_score.py); integrated into existing [scripts/nfl_prop_paired_cdf_forward_grader.py](../scripts/nfl_prop_paired_cdf_forward_grader.py).
- New immutable [history/player_prop_primary_anchor_v1_2026.jsonl](../history/player_prop_primary_anchor_v1_2026.jsonl) of **698** original anchors at **2026-10-12T21:32:45 UTC**, all before earliest target-game kickoff, existing 4,886 originals preserved.
- Independent cloud initial anchor data success https://github.com/konnorharbin-star/harbin-nfl-analytics/actions/runs/38311242309 (**8 tests**), later expanded CI https://github.com/konnorharbin-star/harbin-nfl-analytics/actions/runs/38311331898 (**16 tests**), re-running after 698 original anchored records already exist and adding zero new origins.
- Full integrated anchored primary/gridded secondary forward scoring workflow https://github.com/konnorharbin-star/harbin-nfl-analytics/actions/runs/38311336542 **SUCCESS, 30 tests**, verified cohort 698 and all 4,886 earlier pairs still unchanged.

## Fixed anchor selection before outcome

For each complete seven-line ORIGINAL 2026 paired group with unchanged first-seen SHA, find the synthetic half-point `line` that minimizes **round(abs(original conditional P(over) - 0.50), 10)**, resolving equal distances to the **lower numerical line**. Do not use the snap candidate's own probability, game result, final player snap share, sportsbook quote, editorial line, bookmaker odds, or prior postgame distribution. Validate monotonic NON-INCREASING P(over) over all seven thresholds **for both original and snap CDF**; reject missing/duplicated thresholds, invalid source/probability hashes, mismatched first-seen source provenance, nonfinite or nonmonotone original CDF, and late postkickoff anchor attempts. New immutable anchor row links exact constituent base/snap frozen row hash, seven source SHA hashes, source kickoff and its own first-seen SHA.

First unit run initially had a synthetic floating-point tie misrepresentation (0.55 vs 0.45). Repaired selection to explicit original **10-decimal rounding** and re-ran successfully. Preserve first failed CI run https://github.com/konnorharbin-star/harbin-nfl-analytics/actions/runs/38311084950 as a test-regression correction, not an original model performance result. **No original source CDFs were overwritten.**

## Exact prospective registration result

| NFL player prop | Unique player/game/stat anchors | Mean absolute ORIGINAL-vs-snap conditional P(over) disagreement at the anchor |
|---|---:|---:|
| Receptions | **267** | **0.04141060** |
| Receiving yards | **267** | **0.04672319** |
| Rushing yards | **133** | **0.02971312** |
| Passing yards | **31** | **0.00888696** |
| **Total** | **698** | Not reported as a single independent estimate |

The discrepancy reflects model **disagreement**, not accuracy, price mispricing, profit, evidence of sportsbook underpricing, or a bet. One player can contribute multiple statistics and games, and anchor cohort games share opponents and environments; the 698 rows are NOT 698 independent games or bookmaker tickets. No actual BetMGM offered line/price is verified.

## Postgame grading plan (already automated, not simulated)

The primary score joins an anchored original SHA to the unchanged original paired probability and a separately verified **explicit** official player-stat-row outcome, after completed game plus **at least 72 hours** and source-completeness check, using the same game/team/player/market/line. Missing row unknown; no forced zeros/unders/voids. Score anchor rows as PRIMARY on paired conditional Brier, assess uncertainty by **game-cluster resampling** once >=20 different graded games in a market; include 95% and four-market adjusted **98.75%** CI. The seven-line pooled sensitivity remains separately labeled secondary; do not mine the secondary grid for a winner. An append-only official source correction replaces the interpreted latest grade but does not delete old grade or original pregame anchor.

Cloud first scoring audit at **2026-10-12T21:33Z**: 698 registered anchors present, **0 valid mature anchor outcomes**, **all 4,886** underlying synthetic original lines `GAME_NOT_COMPLETED`, zero Brier or profit evidence. This is proper protection against any 2026 look-ahead.

## Nonnegotiable boundaries

The previous retrospective 2025 snap-conditioned CDF analysis produced no supported improvement after a four-market familywise correction. 2026 anchor registration itself says nothing about more accurate player stats. The target is conditional `P(over | explicit official game-stat row)`, NOT active/inactive, accepted two-sided BetMGM price, wager push/void, actual player-prop refund rules or unconditional **P(win/push/loss)**. Original account-available U.S. BetMGM quotation and market-only no-vig reference, actual settlement, prospectively independent profit/paper trades still absent. Zero accepted/bookmaker-original priced wagers, zero true EV estimates, approved profitable model registry EMPTY, all selections PASS.

**v4.25.0 is a source-cohort quality and statistically rigorous evaluation upgrade, not a betting-model promotion.**
