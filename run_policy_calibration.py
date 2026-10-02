from __future__ import annotations

import argparse
import json

from nfl.policy_calibration import derive_production_policy


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive the frozen NFL market policy from verified chronological evidence."
    )
    parser.add_argument("--bets", default="reports/free_market_bets.csv")
    parser.add_argument("--verified-bets", default="reports/verified_market_bets.csv")
    parser.add_argument("--evidence", default="reports/evidence_report.json")
    parser.add_argument("--output", default="reports/production_policy.json")
    args = parser.parse_args()

    policy = derive_production_policy(
        bets_path=args.bets,
        verified_bets_path=args.verified_bets,
        evidence_path=args.evidence,
        output_path=args.output,
    )
    print(json.dumps(policy, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
