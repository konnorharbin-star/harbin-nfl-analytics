# Pregame QB evidence archive — research, not wagering

## Why this exists

The NFL projection table already exposes **expected QB** IDs, names, source and freshness flags. Those are model estimates, not independently verified starters. The CFB dashboard has injury-risk information but **does not currently supply named, verified starting QBs or backups**.

This research archive freezes what was visible *when a GitHub Actions run checked out the model output*. It **does not** create fabricated lineup identities or assume the highest EA-rated quarterback is healthy. The scheduled workflow is read-only, free, and uploads time-stamped artifacts; it never pushes generated-state changes or modifies betting decisions.

## Scheduled captures

`.github/workflows/qb-pregame-evidence.yml` runs Thursday–Sunday UTC on GitHub-hosted runners and can be triggered manually under Actions. It:
1. Tests source cutoff/identity rules.
2. Selects the existing NFL `outputs/current_predictions.csv` or newest available CFB `outputs/cfb_model_*.csv`.
3. Excludes games whose kickoff has passed and limits the horizon to 14 days.
4. Captures current model QB identity data, paired home and away, and computes **unverified** evidence states.
5. Attaches a SHA256 hash of the source model file, run-time capture stamp and JSON/CSV artifacts for 90 days.

**Crucial:** The capture time is the archive observation time, not the model prediction time. The repo output may itself be older. It does not demonstrate that the player statuses were valid at kickoff. GitHub-hosted artifact expiry means the evidence must be exported into a lasting store before 90 days for long-run prospective evaluation.

## Independent source CSV (future manual/legitimate source adapter)

Optional input `--verified-lineups path.csv` has:
`game_id,team,model_player_id,role,depth_rank,observed_at,obtained_at,source_url,verified`

Roles: `starter,reserve,out,uncertain`. One **independently verified** named starter per team; backup only when clearly identified. Preserve raw source, actual first-obtained timestamp, and player-ID mapping before asserting `verified=true`. The CSV boolean is a human-review attestation, not cryptographic proof, and the code does not scrape, purchase, or fabricate roster statuses.

The validator blocks future-dated evidence, stale observations (older than 72h), duplicate IDs, inconsistent depth ranks, or ambiguity. Disagreement between model QB and independently confirmed QB is surfaced as `STARTER_CONFLICT_WITH_MODEL`, **not resolved in favor of the model**.

An external EA ratings dataset still must pass the stricter `research/ea_talent` provenance and crosswalk gate. Until then `ea_player_gap_rating_units` is always null and `eligible_for_talent_backtest=false`. No point adjustment or ROI uplift is invented.

## Run manually

NFL:

```bash
python research/qb_evidence/capture.py --sport nfl \
  --predictions outputs/current_predictions.csv \
  --capture-at "2026-10-09T18:00:00Z" \
  --csv-output reports/research_qb_evidence_rows.csv \
  --json-output reports/research_qb_evidence_summary.json
```

College football uses `--sport cfb --predictions "outputs/cfb_model_*.csv"`. Run the unit tests with `python -m pytest tests/test_qb_evidence_capture.py -q`.

## Statistical release gate

Do not evaluate a fitted QB replacement challenger until an independently confirmed prospective starter/backup dataset **and** legitimately acquired, point-in-time player ratings are available for the same games. Fit only on earlier data; freeze the challenger and benchmark before kickoff; then test paired forward margin/total MAE, RMSE, win probabilities, tail errors and the risk-adjusted betting outcomes. Unverified starts and missing EA IDs must be excluded from value evaluation—not silently assigned zero talent gaps.

This stage is **evidence collection**, not a proven improvement in betting accuracy.
