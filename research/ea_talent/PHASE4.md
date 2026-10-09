# Phase 4 — Evidence sufficiency gate

Phase 4 is **blocked by absent verified historical EA ratings and pregame lineup archives** until real files have been acquired lawfully and dated. Do not fabricate or time-travel current EA ratings onto earlier games. The repositories' existing model backtests cannot establish an EA improvement without point-in-time paired feature and baseline archives.

Command (from research/ea_talent):
```bash
python ea_evidence_gate.py example_evidence_manifest.json --output ea_evidence_report.json
python -m unittest discover -s . -p 'test_*.py'
```

The provided example manifest deliberately fails closed: production_weight=0 and metrics=null. Each source must be reviewed and saved locally with a real path; a boolean verified field is an audit attestation, not a substitute for inspecting source provenance. This tool checks file existence and a subset of timestamp structure, not authenticity, dataset contents, or real forward precommitment. All fields passing makes descriptive score metrics available, **not** a release approval.

Once real historical snapshots and immutable forecasts exist: validate each event's feature-observation time and cutoff, run rolling-origin training with an untouched final holdout and clustered confidence intervals, compare paired base/challenger MAE and RMSE and probability Brier/log loss, measure actual verified closing-line value, and block promotion on significant regression or unverified lineups. None of these production gates is bypassed here.
