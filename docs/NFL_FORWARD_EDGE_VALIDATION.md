# NFL Step 4 — Prospective Frozen Market-Edge Validation

**Research-only. No change to bets, the fair-score model, stake sizing or release gates.**

The NFL already saves some first pre-kickoff paper/shadow betting decisions
and a separate home-win calibration shadow. However, the betting ledger only
includes **stake-positive** paper/portfolio selections; it does not cover
the full market-model comparison cohort, including PASS candidates. Step 4
adds its own immutable market-relative research cohort across the current
selected best-book candidate for each game and market.

## 1. Strict first-snapshot capture

Each NFL operational model invocation calls
`append_forward_candidates` **after** the existing portfolio policy,
edge-discovery and timing assessment. Only first valid pre-kickoff snapshots
for 2026 games can be written to the append-only file:
`history/edge_forward_candidates_v1.csv`.

The exact **first selected observation per game/market** is frozen. Later
runs cannot overwrite its side, book, price, line or predicted probabilities
even if later odds become more attractive. The first row is retained
even if the quote is stale, fallback-only, invalid, or unverified. Those
rows remain explicit in the denominator, but are excluded from the
eligible Brier/EV sample. There is **no retroactive reconstruction or
backfill of games that already kicked off**.

Frozen data includes:
- season/week/game/teams, capture and kickoff UTC timestamps;
- selected side, canonicalizable bookmaker name, odds, handicap and
  sportsbook quote-capture timestamp;
- model probability and **matched same-book two-way no-vig probability**,
  exact model edge and EV **as actually computed before kickoff**;
- contemporaneous sportsbook execution/source/timestamp/sanity flags;
- frozen production/research actions and stake size, NFL model family,
  edge-reliability tier and timing assessment.

The grader validates `p_model - p_book = edge` and
`p_model * decimal_odds - 1 = raw_EV` to tolerance. It requires timezone-aware
captured and quote timestamps, a quote no more than 30 minutes stale,
quote <= decision < kickoff, supported side/market, and source verified
at capture. The diagnostic quote is **NOT proof of an actual book fill**.
No fresh fetch or post-kickoff update is permitted to change frozen values.

## 2. Separate post-game grade

`grade_forward_edge.py` loads the immutable ledger and separately loaded
completed 2026 regular-season NFL scores. It joins strictly on game and
team/season identity and settles spread, total and moneyline at **the
actual frozen handicap**. Unplayed games remain PENDING_RESULT. Pushes
are scored as zero hypothetical return and excluded from binary Brier and
log loss, preventing fake 50%-win outcomes.

It computes model-vs-matched-no-vig **paired Brier and log-loss**
comparison on exactly the same nonpush candidate rows, market-relative
realized hit rate, raw EV vs hypothetical fixed-quote return, push counts,
per-market coverage and missing/ineligible quote rates. It reports all
three markets regardless of apparent performance. The fixed cohort is
selected before observing outcomes, but uses each market's highest
pre-decision model EV; therefore selection bias remains.

A deterministic **2026 kickoff-week clustered bootstrap** produces 95%
confidence intervals only after at least 100 decided cases from 80
distinct games and 8 distinct kickoff weeks. Intervals are descriptive;
insufficient evidence is reported as null, not zero. All subgroups,
even if strongly profitable, remain unapproved.

## 3. Near-kickoff same-book comparison (not official close)

The grader separately checks later snapshots from the **same canonical
sportsbook**, game, market, selected side and kickoff timestamp, strictly
after the first decision and strictly before kickoff. It only calls a
snapshot `near_kickoff` when observed within 90 minutes of kickoff.
Missing observations are not imputed. Its exact quote is *not* a verified
sportsbook closing price or a filled wager.

For an unchanged spread/total handicap, odds price improvement is measured
in implied break-even percentage points. If the point spread/total changes,
the price/EV comparison is marked
`HANDICAP_CHANGED_PRICE_UTILITY_UNPRICED`. A cover probability at -3.5
is NEVER reused to price -4.5. The amount and direction of point movement
remain separately visible. No market probabilities are learned from the
future snapshot.

## 4. Files and ownership

The model owns and publishes
`history/edge_forward_candidates_v1.csv`. The independent
NFL Live Shadow Grading workflow owns synchronized
`reports/forward_edge_validation.json`,
`reports/forward_edge_graded.csv`, copies under
`outputs/` and `docs/`. The model workflow explicitly excludes these
grading-owned files from its commits to avoid stale cross-workflow races.
Both workflows run tests, and the public model output README links
the JSON status and auditable graded CSV.

No previous paper-bet ledger, bet PNG, recommendation file, calibrated
probability, core projection, release gate, or real/stated stake is
modified by Step 4. A missing ledger produces PENDING_FORWARD with zero
graded observations, never invented forward evidence.

## 5. Decision rules

The output is **NOT a production trading signal**. Even a positive
market-relative Brier interval cannot independently authorize wagering.
Research stays fail-closed until all existing forward execution integrity,
independent probability/regime calibration, sufficient new game samples,
market-only baseline comparisons, and production release controls clear
under a separately reviewed predeclared policy.

The historical 2025 data is already inspected. This 2026 cohort is newly
frozen as games occur; we cannot retrospectively register earlier games.
Subsequent phases can add calendar-based coverage audits and forward
monitoring, but not retune thresholds on this same prospective sample
and then claim it as untouched holdout evidence.
