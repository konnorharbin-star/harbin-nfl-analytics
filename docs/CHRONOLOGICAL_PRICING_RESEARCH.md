# Chronological pricing research v1

The independent model underperformed paired no-vig market probabilities in existing
archive diagnostics. This research tests whether regularized calibration or a learned
football residual improves the market forecast. It changes no production scores,
release gate, betting threshold or recommendation ledger.

For each evaluation season, train only on seasons before the immediately preceding
season. Tune on that preceding season; score the next season without using its
outcomes for fitting or selection. Candidate families are market-only calibration
and market calibration plus independent-model log-odds residual. Ridge penalties
are fixed at 10, 100 and 1000. Market temperature is bounded to [0.5,1.5]; football
residual weight to [0,1]. Penalization shrinks toward the unmodified market. Selection
requires both Brier and log loss to improve on tuning; otherwise retain exactly the
market with zero adjustment. Require 100 games and eight season/week clusters for
training and tuning. There is no further threshold or subgroup search.

Report every evaluation fold, selected parameters, development metrics, pooled
evaluation metrics and paired whole-week bootstrap intervals (10,000 draws,
Bonferroni across two families and two endpoints). Zero improvement is not evidence
of an advantage. Both lower interval bounds must exceed zero even for a descriptive
archive score-improvement flag. Economic betting authorization remains false.

These archives were already inspected during earlier research. The chronological
algorithm prevents outcome access during fitting, but these are **not pristine
untouched holdouts or forecasts actually made before those historical games**.
Source prices lack verified executable entry timestamps. ROI and CLV are deliberately
not inferred; probability improvement would still require economic validation and
new prospective assessment. No successful picks or historical recommendations are
created. The pre-existing prospective market-first experiment remains unchanged.

Run the dedicated Pricing Research workflow after changing this fixed specification.
It rebuilds paired CFB inputs using the canonical free archive, records source hashes,
validates regression tests and publishes report/dashboard summaries. It has no paid
credentials, no betting execution and no automatic model promotion.
