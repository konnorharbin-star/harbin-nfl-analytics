# Harbin NFL Analytics — Model Card

**Release state:** PAPER  
**Production eligible:** False  

## Core model
- Fair score: ridge-regularized team offense/defense scoring model
- Sportsbook prices in score model: False
- Prior-season weight: 0.1
- Quarterback layer: separate NFL-specific research/shadow subsystem

## Probability and markets
- Probability method: chronologically fitted Gaussian score residual distribution
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
- Historical archive fallbacks do not become verified opening entries by relabeling.
- No production staking until every hard release gate passes.
