# Stage 31 — Persisted NFL candidate benchmark evidence

Stage 14 implemented a common rolling-origin benchmark for the main independent
fair-score research candidates, but the workflow had never produced a GitHub Actions
run or a persisted benchmark artifact.

Stage 31 operationalizes that benchmark without changing the canonical model.

## Candidates

The benchmark compares:

- score-level recency weighting;
- prior-season scoring shrinkage;
- opponent-adjusted play-by-play residuals;
- quarterback-state residuals.

Margin and total are evaluated separately where the candidate design permits.

## Chronology

The common development folds remain:

- 2023 validation -> 2024 holdout;
- 2024 validation -> 2025 holdout.

Regular-season Weeks 5-18 are used. These seasons are development evidence because
they have already informed prior research in this repository. The 2026 season remains
reserved for prospective evidence.

## Fail-closed selection

Baseline is always an allowed result. A nonzero candidate must clear every required
chronological development fold and produce positive weighted MAE/RMSE improvement.

Any historical winner is only a SHADOW_CANDIDATE. The benchmark cannot alter the
canonical score, market probabilities, policy thresholds, release state, or stake.

## Persistence

The workflow now writes and retains:

- reports/candidate_benchmark.json
- docs/candidate_benchmark.json
- a GitHub Actions artifact containing the JSON report and execution log

The report is regenerated when candidate benchmark code changes on main and may also
be run manually.
