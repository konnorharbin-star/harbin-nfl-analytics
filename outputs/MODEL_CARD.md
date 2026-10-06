# Harbin NFL Analytics — Model Card

**Release state:** RESEARCH  
**Production eligible:** False  

## Core model
- Fair score: ridge-regularized team offense/defense scoring model
- Sportsbook prices in score model: False
- Prior-season weight: 0.1
- Quarterback layer: historical QB shadow subsystem plus current expected-starter identity/certainty gate

## Probability and markets
- Probability method: chronologically validated score distribution with optional conditional Student-t uncertainty and logistic moneyline calibration
- Historical market source: free nflverse archive
- Current market source: ESPN public endpoints
- Optional enrichment: The Odds API or other verified multi-book source

## Risk controls
- Staking: capped fractional Kelly
- Portfolio caps: True
- Drawdown throttle: True

## Non-negotiables
- No target-week/future leakage.
- No sportsbook line in the independent fair-score engine.
- NFL parameters are validated independently from CFB.
- Missing quotes/context are not invented.
- Current betting requires an identified, decision-ready expected starting QB for both teams.
- A starter change relative to the last-observed QB remains betting-blocked until a validated replacement-QB adjustment exists.
- Prior-week injury designations cannot carry forward as current risk; injury, depth, and roster context must be FRESH.
- Probability changes must improve an untouched chronological holdout; unvalidated reliability blocks betting rather than increasing confidence.
- The 58%-62% confidence band is audited explicitly when the holdout contains an adequate sample.
- Historical archive fallbacks do not become verified opening entries by relabeling.
- No production staking until every hard release gate passes.
