# Default NFL model simplification — October 14, 2026

Owner authorized removing components when justified by evidence. The main NFL
MODEL workflow now disables four experimental branches by default:

| Optional branch | Reason to remove from the default run |
| --- | --- |
| Clean efficiency residual | Previously inspected 2025 margin MAE worsened from 10.6115 to 10.6222; totals tuning chose zero adjustment. |
| Quarter-state workload | Dropback-count MAE worsened from 6.6640 to 6.6937; the Brier improvement interval spans zero. |
| Regulation/OT score challenger | Possession abstraction is not calibrated against future outcomes and does not coherently update player stats. |
| Conditional injury scenarios | Hypothetical unavailable-player scenarios do not verify actual absence or calibrated injury effects. |

These branches already had zero betting weight. Disabling their default execution
reduces extra downloads, calculations and alternative reports; it does **not**
improve the core forecast mathematically. No measured runtime saving or predictive
accuracy gain is claimed. The inspected diagnostics are not pristine proof.

The default run retains own scores/spreads/totals/player forecasts, public football
and injury/weather evidence, game/prop price capture, exact-contract comparisons,
and immutable forward capture/grading. Web evidence remains evidence requiring
verification, not automatic numerical adjustments. Price references and research
comparisons remain subject to existing data and release gates.

To deliberately compare the experiments, manually dispatch NFL MODEL with
`include_experimental_diagnostics=true`. Scheduled runs, code-triggered runs, and
ordinary manual runs leave them off. Experimental-file-only edits no longer trigger
the main model workflow. Their code, tests, historical evidence and prior artifacts
are retained for reproducibility.

Every run archives `reports/nfl_run_scope.json` and displays whether experiments
were enabled. The immutable journal receives this scope record; an intentionally
disabled conditional-scenario file is not requested as a missing source.

This supersedes the default monitoring instructions in the video, workload and
overtime research documentation. Six core predictive fingerprint files, model
coefficients, coupled forecasts, original cohorts and recommendation thresholds
are unchanged. No source/validation gate was removed to force bets. A future core
feature removal must earn its benefit in a declared chronological ablation and,
when needed, begin a separate predictive cohort.
