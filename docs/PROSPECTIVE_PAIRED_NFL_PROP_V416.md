# BetMGM Edge Lab v4.16.0 — Continuous original NFL player props and paired forward validation

**Source audit:** 2026-10-12 UTC. Status: **RESEARCH ONLY; zero proven profitable U.S. BetMGM wagers.**

## Fixed: 2026 v2 pregame coverage would stop after Week 6

The previous v4.15.0 model froze Week 6 player-prop probabilities once and scheduled postgame grading, but future Week 7+ source records were not auto-frozen. Without prospective source prediction timestamps before EVERY later game, a performance model can be selected after observing outcomes.

Added [scripts/auto_freeze_participation_v2.py](../scripts/auto_freeze_participation_v2.py), [tests/test_auto_freeze_participation_v2.py](../tests/test_auto_freeze_participation_v2.py) and daily scheduled [GitHub workflow](../.github/workflows/edge-lab-auto-freeze-v2.yml). During registered season 2026, select upcoming REG football weeks with a verified scheduled kickoff within the next 10 days. Preserve existing original probability records, only append new event/player/market/halfpoint probabilities before each scheduled kickoff; no after-game backfill. Refit neither the 2024 residual training window nor probability parameters with 2026 outcomes. GitHub scheduled checks are best-effort and can be delayed.

**Verified real run:** [38200660893, successful](https://github.com/konnorharbin-star/harbin-nfl-analytics/actions/runs/38200660893). Added **287** new original halfpoint synthetic prop research probabilities for the still-upcoming Week 5 Bills at Rams Monday game and kept the previously frozen **4,585** Week 6 original player-stat threshold probabilities unchanged. Frozen v2 research cohort now totals **4,872** correlated synthetic threshold observations. [Original v2 first-seen file](../history/prop_participation_shadow_v2_2026.csv). Not bets, not sportsbook prices.

## Fair benchmark: preserve the original model before those SAME upcoming games

Added [scripts/auto_freeze_legacy_prop_baseline.py](../scripts/auto_freeze_legacy_prop_baseline.py), [tests/test_auto_freeze_legacy_prop_baseline.py](../tests/test_auto_freeze_legacy_prop_baseline.py) and scheduled [legacy v1 source workflow](../.github/workflows/edge-lab-auto-freeze-v1-benchmark.yml). This is NOT permission to bet using the retired score model. It freezes unchanged original v1 statistical prop predictions as a control in [history/player_prop_probability_forward.csv](../history/player_prop_probability_forward.csv), with separate point estimates in [history/player_prop_forward.csv](../history/player_prop_forward.csv), before the same game kickoffs.

**Verified GitHub run:** [38200974571, SUCCESS](https://github.com/konnorharbin-star/harbin-nfl-analytics/actions/runs/38200974571) recorded 41 original v1 player-game-stat group predictions, representing 287 correlated lines, for Week 5 Monday Bills–Rams; and 655 original v1 player-market groups, representing 4,585 correlated lines, for the 14 NFL Week 6 scheduled games. Earlier first attempt [38200873830](https://github.com/konnorharbin-star/harbin-nfl-analytics/actions/runs/38200873830) failed source COMMIT after generating forecasts: legacy code also modified presentation dashboards under docs/, preventing Git rebase of a clean worktree. That problem was resolved by restoring ONLY ephemeral docs/ edits before rebasing and committing ONLY source-original frozen history records. No force push or retroactive prediction editing.

## Future exact-contract v1 vs v2 statistical scorecard

Added [scripts/paired_v1_v2_forward_prop_audit.py](../scripts/paired_v1_v2_forward_prop_audit.py), [tests/test_paired_v1_v2_forward_prop_audit.py](../tests/test_paired_v1_v2_forward_prop_audit.py) and daily [paired forward score workflow](../.github/workflows/edge-lab-paired-prop-v1-v2.yml). This reads only first postgame grades previously appended by the v1/v2 source grading workflows. It joins on exactly identical event, player ID, market and numerical halfpoint threshold. It requires BOTH original forecasts to predate the same kickoff; official final stat/outcome identity equality; no duplicate original contract; no fabricated cash ROI, bookmaker price or retroactive probability change. Only already-graded future game forecasts contribute to the cohort.

The pre-registered statistic is **delta Brier = Brier(v2) − Brier(v1)**, with negative improvement. Game-level block bootstrap provides descriptive 95% uncertainty while grouping correlated player/stat/threshold outcomes. A minimum 40 distinct NFL games and 500 player-game-stat-market groups per market plus bootstrap upper bound below zero gates a research-only improvement flag. It is NOT a betting approval even if passed: multiple statistical markets and missing sportsbook price/void risk remain.

**First successful actual-source screen:** [38200747251](https://github.com/konnorharbin-star/harbin-nfl-analytics/actions/runs/38200747251) showed **0 currently settled paired contracts**, correctly because these frozen v2 Week 5 Monday and Week 6 forecasts are still for future games. v1/v2 existing scheduled daily official-stat graders will populate separately after game settlement. This output will not retroactively backfill games not originally frozen in both models.

## Important remaining blockers

- The previously inspected 2025 model performance is NOT an independent fresh holdout. Newly frozen 2026 Week 5 Monday/Week 6 forecasts cannot be graded until official games end. No unvalidated coefficient has been promoted.
- 2024-to-2026 v2 residual probabilities are conditioned on the presence of an official player stat row, which does not fully certify player participation or sportsbook void rules. Existing exact GSIS-to-PFR snap count audit left 76 missing player-game-market groups unresolved. Missing official stats must not be imputed to zero or declared void.
- Free public market capture presently provides sportsbook-attributed full-game markets; contemporaneously account-accepted U.S. BetMGM player props at exact lines with original bookmaker update timestamps remain UNVERIFIED. Therefore calibrated true EV, CLV, cash/paper ROI and profitable edges cannot be claimed from the model's synthetic Brier score.
- These repeated daily workflows are free and timestamp-preserving but GitHub schedule is best effort, not instant execution. No automatic wagering.

**Release:** v4.16.0 is a verified forward validation and model-comparison design improvement. Registered profitable-bet model list stays EMPTY. Decision PASS until independent same-contract market-adjusted edge evidence exists.
