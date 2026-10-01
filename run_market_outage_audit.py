"""Deterministic audit of fail-closed current-market outage handling."""

from __future__ import annotations

import json

import polars as pl

from nfl import pipeline
from nfl.contracts import DataContractError


def main() -> None:
    targets = pl.DataFrame(
        [
            {
                "season": 2026,
                "week": 4,
                "game_id": "2026_04_AAA_BBB",
                "home_team": "BBB",
                "away_team": "AAA",
                "gameday": "2026-10-04",
            }
        ]
    )

    original = pipeline.collect_current_markets

    def fail(*_args: object, **_kwargs: object) -> tuple[list[object], dict[str, object]]:
        raise DataContractError("simulated current-market outage")

    pipeline.collect_current_markets = fail
    try:
        markets, meta = pipeline._collect_current_markets_fail_closed(targets, week=4)
    finally:
        pipeline.collect_current_markets = original

    payload = {
        "status": meta.get("status"),
        "markets": len(markets),
        "books": meta.get("books"),
        "multi_book_coverage": meta.get("multi_book_coverage"),
        "source_errors": meta.get("source_errors"),
        "betting_blocked": not markets and meta.get("status") == "BLOCKED",
    }
    if not payload["betting_blocked"]:
        raise RuntimeError("current-market outage did not fail closed")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
