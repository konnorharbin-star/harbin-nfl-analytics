# NFL Step 7 — Full-run publication integrity and pregame reliability

**Operational reliability only. NFL fair scores, sportsbook model
probabilities, release evidence gates, current suggested bets, and
stakes remain unchanged. No research result is reclassified as
executable or profitable.**

## Why this upgrade exists

The prior model workflow verified that `outputs/current_model.json`
and its PNGs reached the remote GitHub branch, but the public
validator only checked that `latest.png`, cache-safe **page 1**,
and the first stable page agreed. On a two-page board, an obsolete
page 2 could be shipped without failing that local validation.
The README's run timestamp and linked per-run filenames were not
explicitly reconciled with the canonical model generation timestamp.
The remote step also did not check the actual published suggested
bets CSV, portfolio recommendations, public Docs JSON/CSV copies
or weekly HTML. An older GitHub mobile preview could therefore
look plausible when the run was incomplete.

## Verified current-run manifest

`outputs/publication_manifest.json` and its exact
`docs/publication_manifest.json` mirror list SHA-256 hashes and
byte sizes for **the active run only**:

- canonical model JSON and generated predictions CSV;
- suggested bets and allocated recommendations CSV;
- current quant card, public quant card and public weekly board;
- README, season/week, release state and user-facing update timestamp;
- `outputs/latest.png`, `docs/latest.png`, every stable PNG and
  **every cache-safe PNG**, including page 2+;
- current `docs/latest.json`, `docs/latest.csv` and audit snapshot.

All files must exist and be non-empty. The manifest validates:

1. Canonical run timestamps are offset-aware and the report does not
   precede its model metadata.
2. The filename timestamp from `America/Chicago` exactly matches
   `publication.run_tag` for this run, including daylight saving.
   Run-specific filenames cannot silently retain a previous timestamp.
3. Stable and cache-safe page counts and sequential page numbers
   match the actual number of board games; no duplicate paths.
4. Every stable and fresh PNG page agrees byte-for-byte. With an
   actual run tag, every page must also have a PNG signature.
5. README's updated date, release state, week and page links match
   the canonical JSON, and the public Docs copies are identical.
6. After writing the manifest, validation rebuilds every hash and
   refuses to report PASS if files were modified.

This manifest is **not a sportsbook fill proof, betting backtest
or cryptographic attestation of the underlying injury/odds providers.**

## Remote publication gate

After the generated-state commit and race-safe Git push, the NFL
operational workflow runs
`python scripts/verify_remote_publication.py --branch main`
(or the relevant target branch). It verifies **every manifest path**
against `git show origin/<branch>:<path>`, including the two
identical manifest files themselves.

If any page, recommendation CSV, readme, public HTML, model or
manifest is missing/stale remotely, the workflow fails even if the
local model computation and initial Git push succeeded.

The result is scoped to the remote Git branch the workflow fetched.
GitHub UI image previews may still cache previously opened image
URLs, so the per-run filename remains the preferred way to view
new pictures.

## Read-only pregame watchdog

A separate **NFL Published Board Freshness Watchdog** workflow runs
on NFL game-day windows (and supports manual dispatch), checking
published `main` without writing, fetching odds or re-running
the model. It checks the full local manifest on its checkout
and measures the canonical model age against upcoming kickoff times,
deduplicated by game rather than market.

The watchdog is intended to run *after* the model's existing
15-minute refresh checkpoints, not simultaneously with them.
A game within the next 120 minutes is considered critical; the
following conservative staleness limits apply:

| Time to kickoff | Maximum model age |
|---|---|
| (60, 120] minutes | 130 minutes |
| (15, 60] minutes | 85 minutes |
| (0, 15] minutes | 35 minutes |

The time limits intentionally allow for GitHub-hosted runner
latency but detect omitted refreshes; they are **engineering
thresholds, not wagering thresholds**. Outside critical windows,
an older snapshot by itself does not cause repeated failure
notifications. Missing manifest, invalid run timestamps, inconsistent
kickoffs, and nonproduction positive stakes always fail closed.

A failed scheduled workflow can be seen in GitHub Actions.
Actual email/push alerts depend on the repository and account's
notification settings. GitHub's schedule is best-effort, so this
does not guarantee second-by-second monitoring.

## Regression evidence

The test suite covers all-page checksum reconciliation, stale
page 2, old README timestamps, wrong run tag, changed bets CSV,
missing or modified Docs files, duplicated page counts, path
traversal, and verification against a separate local bare Git
remote where a second writer changes page 2 after a successful
model commit. Separate watchdog tests cover the critical pregame
time windows, no-game/no-alarm state, future/absent clocks, nonproduction
stake anomalies and inconsistent same-game kickoff.

Promotion/release policy is unchanged, including the presently
blocked injury-report source provenance and historically unsupported
market edge evidence.
