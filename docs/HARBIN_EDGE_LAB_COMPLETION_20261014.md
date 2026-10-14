# Harbin Sports Edge Lab Final Research & Completion Report

Research date: October 13, 2026 America/Chicago (October 14 UTC).
Decision: **NO VALIDATED EDGE. Current selections: PASS.**

## 1. Operational completion

Both existing repositories were audited and their full test suites run. NFL PRs
196–207 and 209 were merged at audit; PR 208 remained open. That PR's separate
2024-training / 2025-diagnostic probability-residual experiment was replayed,
reviewed and incorporated here; its unrelated alternate underdog module was
not adopted. College model expansion remains deferred. No wagers are executed.

The existing NFL system already collects public football data, produces
independent game and player forecasts, captures public game odds, and freezes
separate legacy prop cohorts. It is a functioning research system. It is **not
a complete validated betting recommendation model**: current joint simulation
omits unusual scoring and explicit overtime/game-state mechanics; confirmed
QB/participation, void rules, source vintages, market-relative calibration and
executable quote provenance remain incomplete.

The latest coupled game/prop workflow had an operational gap: forecasts existed
only in 45-day artifacts and grading required a manually supplied source run ID.
This change adds a distinct durable first-seen cohort under
`history/coupled_primary_v1_2026/`. Original scan, replay and available web
sidecars are retained as SHA-addressed JSON. One primary forecast per game is
registered before kickoff. Repeated snapshots do not replace originals or
increase independent sample counts. Model implementation changes require a
separate cohort; unrelated repository commits do not change the fingerprint.
Source failures/missing sidecars remain visible. Archive hashes detect local
corruption; they do not independently attest provider truth.

The same scheduled workflow now grades the durable originals with the existing
72-hour maturity, event/team/GSIS matching and missing-stat safeguards. First
grades and subsequent source corrections append to a hash-protected journal.
Repeated unchanged grades add nothing. No absent player row becomes zero or
a void. Cumulative point errors and game counts survive reruns. Two fixed paired
point-error comparisons receive week-clustered 97.5% descriptive intervals only
after at least 128 distinct games and eight week clusters. This is a research
review threshold, never betting promotion.

## 2. Strongest externally grounded hypothesis tested

Krieger and Shank, *Do Sportsbooks Accurately Price Money Line Odds?*
DOI https://doi.org/10.5750/jpm.v19i2.2325 reports historical moneyline returns
for small NFL underdogs. The paper's public abstract reports 6.55% for spreads
at most 3 and 2.17% for spreads above 3 through 7. Publication metadata differ
across repository/index versions; no exact temporal replication of the authors'
original dataset is claimed. This is an external-rule recent-window robustness
test, not reproduction of their original research sample.

Source: free nflverse 2022–2025 completed regular-season schedules.
Preserved exact source fields:
`research/nfl_small_dog_source_2022_2025.csv`.
Report: `reports/nfl_small_dog_historical_replication.json`, with capture time
and input SHA256. Historical prices are undated archive prices, not independently
authenticated accepted sportsbook offers.

An audit caught a material defect in the earlier replication: negative signed
spreads were discarded. nflverse positive spread means home favorite, so this
excluded home underdogs. The corrected test uses absolute spread magnitude,
includes home and away underdogs, validates finite integer score outcomes and
price orientation, and explicitly counts ties as hypothetical stake refunds.
The original incomplete 210/259-game results remain in prior project history;
the new values below supersede them for this corrected sample definition.

| Fixed rule | Games | W / L / assumed refund | Hypothetical ROI | 95% week-cluster interval |
|---|---:|---:|---:|---:|
| Underdog ML, 0 < absolute spread <= 3 | 390 | 155 / 234 / 1 | -10.7846% | -20.9027% to -1.0362% |
| Underdog ML, 3 < absolute spread <= 7 | 425 | 127 / 296 / 2 | -11.1482% | -24.0796% to +1.5276% |

Both cohorts have 72 season-week clusters; fixed one-unit stake, 5,000 bootstrap
resamples. Intervals are descriptive, not adjusted for all previous research.
Different bookmaker/time provenance, excluded pickems/nonpositive underdog
prices and revised schedule data limit comparison with the publication.
**Neither rule establishes a profitable recent strategy.**

## 3. Chronological residual falsification

One archived selected-side record per game/market. Select a weight from the
fixed grid {-0.5,-0.25,-0.1,0,0.1,0.25,0.5} on 2024 Brier score only; apply it
unchanged in 2025 to market probability plus weight × (model minus market).
Exclude pushes from binary proper scores; neither year is pristine because
both were previously researched. These are archived older-model probabilities,
not validation of the 2026 coupled simulator.

| Market | 2025 decided games | 2024-selected weight | Market Brier | Challenger Brier | 95% week-cluster Brier lift |
|---|---:|---:|---:|---:|---:|
| Moneyline | 271 | -0.25 | 0.21536389 | 0.21505319 | -0.00189840 to +0.00250814 |
| Spread | 269 | 0 | 0.25137716 | 0.25137716 | 0 to 0 |
| Total | 272 | -0.10 | 0.24983460 | 0.24971206 | -0.00091617 to +0.00116187 |

Positive lift means lower challenger Brier. No market has a demonstrated
nonzero positive lift. Negative selected weights mean the diagnostic moved
against the old model's residual; they are not a new authorized betting rule.
Log loss, train counts, source hash and uncertainty are retained in
`reports/market_residual_year_split_diagnostic.json`.
No original quote times or actual fills exist for these archive prices, so no
legitimate executable ROI or CLV is inferred.

## 4. Validation and release

The canonical policy's 128-game coupled predictive review, 20-distinct-game
primary prop-family review with four-family correction, and applicable
300 independent priced-decision/CLV gates are unchanged. Minimum sample counts
are necessary, not sufficient: market-relative calibration and positive
cost/uncertainty-adjusted returns still have to pass.

This new durable cohort begins prospectively; it does not retroactively turn
old artifacts into first-seen repository evidence or replace the original
698-anchor/4,886-threshold study. The local live source smoke produced 14 future
Week 6 forecasts and zero mature outcomes. A live cloud persistence run must
separately succeed before operational deployment is considered verified.

## 5. Forward operating system

Workflow: `.github/workflows/nfl-coupled-shadow-research.yml`.
Daily 13:11 UTC during September–December 2026; extra Sunday 16:11 and 19:11 UTC
scans. GitHub scheduling is best effort. The runner refuses unreviewed future
years. Each run checks journal integrity, grades existing mature originals,
collects free current source data, generates upcoming-week football forecasts,
fetches public injury/news/NWS context and conditional scenarios, archives only
new primary games, and commits the journals. Rebase/push conflicts fail visibly.
The workflow does not trigger itself on history-only commits.

Existing `edge-lab-weekly-public-nfl.yml` separately captures public game
moneyline/spread/total pairs at 07:09, 15:09 and 22:09 UTC. Those are source
observations, not actual account-executable prices. Existing exact-contract and
crossbook scanners reject mismatches and missing provenance. Automated complete
first-party player-prop price coverage remains unavailable. No forecast-to-price
disagreement is promoted to a wager without canonical validation.

## 6. Remaining proof

For a 14-game new primary cohort with zero mature results: all 128 required
distinct mature game outcomes are still missing; if all current 14 qualify,
at least 114 additional future games are needed. Each registered primary prop
market needs 20 source-qualified distinct games; the older cohort covers only
15 games in total, so at least five further distinct games are necessary and
may be insufficient for individual markets with missing player outcomes.
The wider v1/v2 paired study separately requires 40 distinct games and 500
player-game-market groups per family. No sample from a different family can
substitute for these registered studies.

For profit certification, collect at least the applicable 300 independent,
originally timestamped priced decisions with authentic entry and matching
closing sources, verified participation/settlement, and independent prospective
returns. There is no finite observation count that guarantees an edge will
exist or its confidence interval will be positive.

Research priority: preserve and test fixed workload/receiving-prop hypotheses
and contemporaneous same-contract price discrepancies. Earlier receiving
accuracy gains are diagnostic candidates, not sportsbook-profit evidence.
Walters' public process supports independent handicapping, diligent information
collection, price shopping and disciplined limits; it does not supply verified
private coefficients. Source https://realbillywalters.com/handicapping-system/
was retrieved October 14 UTC. No additional unvalidated injury/weather or
player scoring coefficients were introduced.

## 7. Deployment evidence

The merged PR, cloud CI/run IDs, durable receipt count and canonical document
readback are recorded in the final live Google Drive change entry and the
completion follow-up below. Until those writes/runs are verified, this report
describes tested changes awaiting deployment rather than completed cloud work.
