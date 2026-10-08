# Harbin NFL Run Report

**Season / Week:** 2026 / 5  
**Publication status:** FAIL  
**Release state:** RESEARCH  
**Reconciliation:** FAIL  

## Probability validation
- Home-win Brier: **0.2287**.
- Margin / total 80% coverage: **78.31% / 80.51%**.

## Monitoring and execution
- Live readiness: **79.9/100**.
- Portfolio mode: **PAPER**; proposed **0.00u**; approved **0.00u**.
- Historical evidence: **2584 bets**, ROI **-7.84%**, CLV **—**.
- Independent forward evidence: **11 bets**, ROI **12.33%**, CLV **4.70%**.

## Current blockers
- quarterback_context_coverage: >=95% current games with identified, decision-ready expected starting-QB state
- injury_personnel_freshness: >=95% current games have FRESH injury, depth-chart, and roster context; STALE/UNKNOWN state fails closed
- context_coverage: >=90% current injury/rest/weather/travel context coverage
- regime_edge_reliability: at least two markets clear fixed regime-specific probability and edge reliability on validation plus untouched holdout
- live_monitoring: engineering readiness >=90/100; multi-book breadth gated separately
- historical_clv_coverage: >=90% of verified historical bets have same-book closing CLV
- historical_market_edge: ROBUST verified-entry NFL evidence with positive ROI confidence lower bound and CLV across markets/seasons
- production_policy: frozen nested chronological policy is production-validated with at least two enabled markets
- live_shadow_evidence: >=300 graded portfolio-verified live/shadow bets, non-negative ROI, positive CLV, complete entry provenance, and >=90% CLV coverage

## Operational alerts
- 18 market rows have high model-vs-consensus disagreement
- fewer than two markets clear regime-specific edge reliability

## Interpretation
A green software run, high readiness score, or low model error does not establish a profitable betting edge. Historical and independently graded forward evidence remain separate release requirements.
