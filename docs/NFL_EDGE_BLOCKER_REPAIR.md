# NFL edge gate repair — source integrity versus predictive skill

The October 8, 2026 TB @ DAL example exposed a presentation and source-provenance
problem: raw positive model EV was easy to mistake for a tradable edge even though
the holdout favored a no-vig market baseline, current personnel was unverified,
and the opponent/price regime was not reliable.

## What has been corrected

- The published `docs/manual_review.csv` and `docs/manual_review.json` are
  derived from the existing model forecast at publication time. They report
  `PASS`, `NO_VERIFIED_MARKET`, or `REVIEW_ONLY`, and enumerate blockers.
  **`REVIEW_ONLY` is purely a human research invitation; it never initiates
  a bet or authorizes a stake.**
- A massive positive `quant_ev` is preserved strictly as a raw research
  diagnostic. A positive `conservative_ev` is only displayed when
  `edge_shrunk_ev` is positive AND the regime, model holdout, injury/QB context,
  price verification, and other evidence checks are independently passing.
- Public RESEARCH and PAPER HTML boards no longer show raw `BET` or `STRONG`
  badges when those evidence checks fail, and link to the blocker report.
- Injury feeds select the best **source-reported** timezone-aware timestamp
  field, including alternate field names. A lone dated player does not certify
  hundreds of undated injuries: 95% or more current-week coverage is necessary
  for overall feed freshness. Neither scraping/capture time nor a naive date
  can be fabricated into an injury report date.

## What code alone cannot repair

- **Current injury data:** the observed weekly source contains rows with no
  source-origin timestamp column. Those reports remain UNKNOWN/STALE. A truly
  dated and adequately covered free injury feed or timestamped official NFL
  report archive must be added and verified before personnel gets a FRESH label.
  Never stamp old source rows with the time the collector downloaded them.
- **QB replacement:** when the identified starting quarterback changes from
  the last-observed QB used by the model, and the replacement effect has not
  passed untouched validation, the QB certainty veto must remain.
- **Market edge:** the model's historical regime audit currently finds
  unreliable spread, total and moneyline groups. The shrinkage validation
  chooses market-only (alpha = 0) in affected cases; a negative shrunk EV is
  evidence to **PASS**, regardless of an attractive raw EV. To overturn this
  requires new point-in-time, out-of-sample predictive evidence, not bypassing
  flags or raising the raw forecast probability.
- **Prices:** models observe quotes, they do not certify wager fillability
  or guaranteed profit. A quote can expire before kickoff.
  
## Safe future work

1. Obtain and validate a no-cost, permitted current injury feed with real
   reported-at provenance; cover both teams of each game, including roster,
   depth and expected starting QB. Keep incomplete games blocked.
2. Accumulate a strictly frozen, forward-verified entry ledger with prices,
   bookmaker, as-of timestamp, and independently corroborated final scores.
3. Research better score features only on chronological training seasons and
   compare with a **no-vig market-only baseline** on a sealed holdout season.
   Do not deploy any adjusted probability model that fails against the market.
4. Review `docs/manual_review.csv` after each live model refresh. A row with
   `PASS` is **not** a recommended wager, however large `quant_ev` appears.

**No automated betting. No paid APIs, databases or subscriptions.**
