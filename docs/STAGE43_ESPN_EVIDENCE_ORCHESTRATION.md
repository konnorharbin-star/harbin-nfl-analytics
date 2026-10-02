# Stage 43 — ESPN historical evidence orchestration

Stage 43 fixes the first post-merge execution failure from Stage 42 and removes a
generated-evidence writer race.

## History-window correction

Stage 42 targets verified archived opening/closing evidence from 2024–2025 while using
2023 games as chronological probability-training history.

Building the 2023 projection history itself requires 2022 completed games for the
first 2023 fair-score fits. The first full Stage 42 run loaded only 2023–2025 schedules,
so 2023 Week 1 correctly failed the fair-score minimum-history contract.

The evidence runner now loads 2022–2025 schedules when the evidence window is
2024–2025. A regression test locks that season window.

## Canonical evidence writer ordering

The free nflverse market backtest and the ESPN archived-market backtest both contribute
to the canonical historical proof and can write:

- reports/evidence_report.json
- reports/production_policy.json

Running them concurrently can create a true generated-state conflict even though both
writers are individually correct.

The writer chain is now explicit:

1. Free NFL Market Backtest runs first and refreshes canonical free research evidence.
2. Free ESPN Historical Market Backtest is triggered only after that workflow completes
   successfully.
3. The ESPN backtest checks out the latest main branch so it reads the newest canonical
   free reports before adding verified archived opening/closing evidence.
4. A successful ESPN evidence run triggers NFL Model + Operations to recalculate the
   release gate and publication state.

Scheduled free-market refreshes do not automatically repeat the expensive ESPN
historical archive build; source changes and manual dispatch remain the intended
historical rebuild paths.

## Safety properties retained

The Stage 40 generated-state source-revision guard remains unchanged. True content
conflicts still fail closed, and stale writers still cannot cross newer source
revisions.

No football model, probability model, evidence threshold, policy threshold, Kelly rule,
or release criterion changes in Stage 43.
