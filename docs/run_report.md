# Harbin NFL Run Report

**Season / Week:** 2026 / 4  
**Publication status:** WARN  
**Release state:** PAPER  
**Reconciliation:** PASS  

## Probability validation
- Home-win Brier: **0.2300**.
- Margin / total 80% coverage: **78.31% / 80.51%**.

## Monitoring and execution
- Live readiness: **95.7/100**.
- Portfolio mode: **PAPER**; proposed **1.25u**; approved **0.00u**.
- Historical evidence: **2584 bets**, ROI **-7.84%**, CLV **—**.
- Independent forward evidence: **0 bets**, ROI **—**, CLV **—**.

## Current blockers
- historical_clv_coverage: >=90% of verified historical bets have same-book closing CLV
- historical_market_edge: ROBUST verified-entry NFL evidence with positive ROI confidence lower bound and CLV across markets/seasons
- production_policy: frozen nested chronological policy is production-validated with at least two enabled markets
- forward_entry_integrity: 100% graded forward bets have execution-ready, timestamp-valid entry quotes
- forward_clv_coverage: >=90% of graded forward bets have a valid later pre-kickoff closing snapshot
- live_shadow_evidence: >=300 graded portfolio-verified live/shadow bets, non-negative ROI, positive CLV, complete entry provenance, and >=90% CLV coverage

## Operational alerts
- None.

## Interpretation
A green software run, high readiness score, or low model error does not establish a profitable betting edge. Historical and independently graded forward evidence remain separate release requirements.
