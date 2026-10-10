# 2025 NFL raw-EV calibration failure audit (Walters research)

**Purpose:** investigate why apparent NFL +20–30% raw spread EV in current web odds is not a trustworthy economic edge. This historical audit operates separately from the 2026 key-number challenger and **never changes live prices, probabilities, stake units or the production no-bet gate**.

Historical source: `reports/verified_market_bets.csv` from provider-labeled NFL archived market prices. It contains one archive-selected spread per game, 2024 and 2025. It **does not** certify point-in-time executable book odds: original source-side timestamps are not independently verified, and the archived `probability_model_family` is an **older** model, not the 2026 3/7 exact-score challenger. All profit/ROI values are hypothetical *archive settlements*, not the outcome of wagers or a backtest on verified executable odds.

## Design and locked chronology

- Require valid distinct game IDs by season, 2024 or 2025 only, resolved WIN/LOSS/PUSH and one-unit settlement values.
- For Brier and log loss, condition on settled outcomes and exclude pushes because the archived model did not estimate push probability. New live key-number models use separate correct three-way probability scores.
- Compare market **no-vig** probability on the selected side with raw older-model probability. Blend *log odds*, not margins: `logit(p)=logit(p_market)+alpha*(logit(p_model)-logit(p_market))`.
- Fit/tune **only on 2024**, choosing lowest 2024 Brier among fixed weights **0, 0.10, 0.25, 0.50, 0.75, 1.00**; 0 ties are preferred. Evaluate the chosen weight and endpoints on 2025. Both seasons were previously inspected. **2025 is not a pristine holdout.**
- Report every unfiltered 2025 spread outcome, historical raw EV (not the 2026 model), and archived settlements by non-overlapping pre-specified EV bins. Explicit +20% and +30% tail summaries are overlapping diagnostics only, not selected live wagers.
- Week-cluster bootstrap simultaneous uncertainty intervals for market-minus-full-model Brier/log loss; do not convert a probability-score difference into a staking strategy.

## Current sample (first observed before implementation)

From 544 archived spread selections across 2024–25: 2025 272 entries settled with **−6.44% archive ROI** while mean raw model EV was **+10.8%**. Of 61 with raw EV at least +20%, archived settlements yielded approximately **−5.14% ROI**; 24 at least +30% yielded approximately **−18.78% ROI**. The 2024 tuning market-only Brier was **0.250228**, full model **0.263684**; 2025 market-only **0.251377**, full model **0.265310** (pushes omitted). The 2024 tuning winner among these grid weights was **alpha=0.0**, independently selected without 2025 outcomes. These are descriptive archived results; do not imply model prediction reliability or economic profitability today.

## Run and output

```bash
python -m pytest -q tests/test_walters_raw_ev_reliability.py
python -m scripts.walters_raw_ev_reliability \
  --source reports/verified_market_bets.csv \
  --output reports/walters_nfl_raw_ev_calibration.json
```

The established NFL model workflow generates the audit without overriding any stake gate. The read-only research workflow runs the tests plus the historical audit and uploads an artifact. The output provides no wager direction. **Do not promote new key-number edges unless separately validated on prospectively published football projections and genuine point-in-time executable sportsbook prices, followed by late-price and settlement verification.**
