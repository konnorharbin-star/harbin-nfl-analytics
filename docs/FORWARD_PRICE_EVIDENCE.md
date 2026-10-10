# Forward price provenance and paper-result audit (v1)

**Research-only, never a betting authorization.** This stage exists because
the repositories have no graded prospective paper recommendations and no
verified closing-price samples. Published price-scan observations are useful
diagnostics, **not** proof that a named sportsbook offered executable odds.
A person would need access to the book and to independently verify the
source origin, price, game, settlement and time before calling it a candidate.

## Current-state audit

\`scripts/forward_price_validation.py\` inspects all frozen
\`history/price_scan_v1/captures/*.json\` files. It hashes exact JSON record
bytes to avoid duplicate counting of the same observation across captures and
reports (a) total source rows, (b) structurally valid two-sided pairs,
(c) provenance flags, and (d) machine-consistent observations.

Required original source evidence:
- real originating HTTPS source URL (not merely the aggregator collector URL);
- independently mapped bookmaker identity;
- publisher-side quote update time (timezone-aware);
- actual collector acquisition time before kickoff and within 15 minutes of
  the publisher's quote timestamp;
- evidence that the exact price was available for a legitimate wager.

The current aggregator rows generally lack these fields, therefore a
fresh collector **cannot** turn them into verified sportsbook observations.
Explicit flags are *human attestations* and do not independently certify
them; even source-complete records remain under manual review.

## Prospective paper receipts, if independently sourced

For an actually observed and independently reviewed hypothetical selection,
commit a uniquely named JSON receipt in
\`history/forward_verified_quotes_v1/entries/<receipt_id>.json\`
**before the game begins**. The file must not be revised. The audit verifies
the first Git addition commit and checks that its bytes still match. This
Git-based cutoff is necessary, not sufficient: Git commit timestamps can be
manipulated and do not replace independently trustworthy sportsbook times.

The receipt's keys are:

\`receipt_id,sport,game_id,kickoff,market,side,line,book,
american_odds,opposite_american_odds,source_url,source_quote_at,
observed_at,frozen_at,book_identity_verified,source_quote_time_verified,
executable_price_verified,book_access_verified,settlement_rules_verified,
frozen_model_sha256,model_side_probability\`

Important semantics:
- Moneyline uses \`line: null\`. Spread has **selected-side handicap** (home
  -3.5 means \`side:home,line:-3.5\`; away +3.5 means \`side:away,line:3.5\`).
  Total has the over/under points threshold.
- Both prices are from the **same named book** at the **same exact line**.
  No invented -110 defaults; full American odds must be supplied.
- \`frozen_model_sha256\` is a full 64-digit SHA256 of the unchanged independent
  forecast source. It is an integrity field, not by itself external proof
  of model quality or chronology.
- \`model_side_probability\` is the contemporaneous model's **unrounded**
  probability from an independently frozen prediction. For spread/total
  with push probability, do not interpret this standalone as a valid EV.
- All verification flags must reflect actual inspection; do not set them
  true just to make a paper receipt pass validation.
- The first Git addition must precede kickoff; entry/origin/observation dates
  must be strictly chronological. A record added after kickoff fails, even if
  its JSON contains an earlier time.

Optionally add exactly one immutable late-market quote to
\`history/forward_verified_quotes_v1/late_quotes/<receipt_id>.json\`.
It must preserve game, same named book, market, selected side and **exact
handicap**. Price origin must postdate the original observation, and it must
be collected/published within 30 minutes before kickoff. Both selected and
opposite-side odds and all original source verification fields are required.
The paired, no-vigged selected-side probability shift is reported as a
**late-market movement proxy**; it is never called certified final CLV.
A missing/incomparable late quote is **null**, not a zero movement.

Optionally attach a final-score evidence record to
\`history/forward_verified_quotes_v1/finals/<receipt_id>.json\` using:
\`receipt_id,sport,game_id,status:"FINAL",verified_result_source:true,
source_url,observed_at,home_score,away_score\`.
This is a separate manually verified source (or suitably supported source
adapter) and must be after kickoff. Without an unambiguous valid final,
the paper entry remains pending, not a loss. This artifact is not a
scoreboard API; source verification is still manual.

Each qualifying first-published entry is counted once, for a uniform **one
hypothetical unit stake**, regardless of result. The report gives wins,
losses, pushes, ROI per *settled* entry, and the number with comparable late
quotes. It also exposes pending, blocked, missing-close and missing-final
states; never backfills or imputes a result. It does not use fractional
Kelly, issue wagers, publish picks, or override a model release guard.
Paper returns and a late-market proxy do not themselves establish a
repeatable market advantage. Small samples and missing-data bias remain
explicit limitations.

## Reproduce without paid APIs

\`\`\`bash
python -m pytest -q tests/test_forward_price_validation.py
python scripts/forward_price_validation.py \
  --captures history/price_scan_v1/captures \
  --paper-root history/forward_verified_quotes_v1 \
  --out reports/forward_price_evidence.json
\`\`\`

The manually callable **Forward Price Evidence Audit** GitHub workflow runs
the isolated tests, analyzes the actual checked-in immutable quote captures,
and uploads one report artifact. The workflow does not write to \`main\`,
collect paid prices, fabricate prices, open positions, or publish bets. The
full repository CI must also pass on any PR.

### Research release gate

Until exact-source independent evidence, chronological first capture,
user-accessible sportsbook executable odds, and prospective sample
outcomes exist: economic edge is **NOT VERIFIED**, model promotion is
**NOT AUTHORIZED**, and stake recommendation is **NO BET**.

## First real-capture outcome — October 10, 2026

The new research workflow ran against the repositories' immutable captured
market snapshots. After deduplicating exact source-record observations:

| Sport | Distinct quote observations | Source provenance complete | Frozen paper entries | Settled paper entries | Late-market comparison samples |
|---|---:|---:|---:|---:|---:|
| NFL | 1,210 | **0** | 0 | 0 | 0 |
| CFB | 1,344 | **0** | 0 | 0 | 0 |

The archived aggregator quotes fail source-origin/executability verification;
zero provenance-complete entries is **not evidence that all prices are wrong**.
No sportsbook economic edge, realized ROI or verified CLV is supported.
These data must not be promoted to wagers. Re-run the workflow to obtain
newer counts as free captures accumulate.
