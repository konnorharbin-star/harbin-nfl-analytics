"""Compare the fixed receiving model with real prop history. No paid access or wagers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.error
import urllib.request
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import nflreadpy as nfl
import polars as pl

from nfl.prop_challenger import source_frames
from nfl.receiving_market_benchmark import fit_fixed_candidate, import_histories, score_quotes
from scripts.nfl_receiving_usage_edge_test import normalize_players
from scripts.usage_prop_research import add_opportunity_columns


def select_players(players, count):
    if type(count) is not int or not 1 <= count <= 20:
        raise ValueError("player count must be 1..20 (at most 200 provider credits)")
    targets = Counter()
    for row in players.to_dicts():
        if (
            row.get("season") == 2024
            and row.get("season_type") == "REG"
            and row.get("position") in ("WR", "TE")
        ):
            targets[row["player_id"]] += row.get("targets") or 0
    return sorted(targets, key=lambda p: (-targets[p], p))[:count]


def fetch_history(player_ids, api_key, *, opener=urllib.request.urlopen, checkpoint=None):
    if not api_key:
        raise ValueError("ENDZONE_API_KEY_REQUIRED_FROM_FREE_ACCOUNT")
    if len(player_ids) > 20 or len(set(player_ids)) != len(player_ids):
        raise ValueError("at most 20 distinct players per manual import")
    responses = []
    for player_id in player_ids:
        body = {
            "player_id": player_id,
            "market": "player_reception_yds",
            "bookmaker": "draftkings",
            "from_date": "2025-09-01",
            "to_date": "2026-02-01",
            "limit": 200,
        }
        request = urllib.request.Request(
            "https://mcp.endzoneapi.com/v1/get_prop_history",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "x-api-key": api_key},
            method="POST",
        )
        try:
            with opener(request, timeout=30) as response:
                result = json.load(response)
        except urllib.error.HTTPError as error:
            # No response body, key, paid fallback, retry or quota top-up.
            raise RuntimeError(f"PROVIDER_HTTP_{error.code}; stopped without retry") from None
        except (OSError, ValueError):
            raise RuntimeError("PROVIDER_TRANSPORT_OR_JSON_ERROR; stopped without retry") from None
        if not isinstance(result, dict) or "data" not in result or result.get("error"):
            raise RuntimeError("INVALID_PROVIDER_RESPONSE")
        responses.append(result)
        if checkpoint is not None:
            checkpoint(
                {
                    "responses": responses,
                    "requested_player_ids": player_ids,
                    "complete": len(responses) == len(player_ids),
                    "maximum_requested_provider_credits": len(player_ids) * 10,
                    "captured_utc": datetime.now(UTC).isoformat(),
                }
            )
    return {
        "responses": responses,
        "requested_player_ids": player_ids,
        "selection_basis": "2024_REG_WR_TE_TARGETS_DESCENDING_THEN_GSIS_ID",
        "maximum_requested_provider_credits": len(player_ids) * 10,
        "captured_utc": datetime.now(UTC).isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--history", type=Path, default=Path("reports/receiving_actual_market_history.json")
    )
    parser.add_argument("--fetch-free-history", action="store_true")
    parser.add_argument("--players", type=int, default=20)
    parser.add_argument("--cache-prefix", type=Path)
    parser.add_argument(
        "--out", type=Path, default=Path("reports/receiving_real_price_benchmark.json")
    )
    args = parser.parse_args()
    report = {
        "status": "SOURCE_BLOCKED",
        "validated_edges": 0,
        "promotion_earned": False,
        "betting_authorized": False,
        "captured_utc": datetime.now(UTC).isoformat(),
    }
    try:
        if args.fetch_free_history and not os.getenv("ENDZONE_API_KEY"):
            raise ValueError("ENDZONE_API_KEY_REQUIRED_FROM_FREE_ACCOUNT")
        if not args.fetch_free_history and not args.history.is_file():
            raise ValueError("ACTUAL_PROP_HISTORY_REQUIRED; synthetic lines are not accepted")
        if args.cache_prefix:
            schedules = pl.read_parquet(args.cache_prefix / "schedules_research.parquet")
            players = pl.read_parquet(args.cache_prefix / "player_research.parquet")
        else:
            schedules = nfl.load_schedules([2024, 2025])
            players = nfl.load_player_stats([2024, 2025])
        if args.fetch_free_history:
            if args.history.exists():
                raise ValueError(
                    "HISTORY_ALREADY_EXISTS; import existing snapshot without fetching"
                )
            args.history.parent.mkdir(parents=True, exist_ok=True)

            def checkpoint(snapshot):
                args.history.write_text(
                    json.dumps(snapshot, sort_keys=True, indent=2) + "\n", encoding="utf-8"
                )

            history = fetch_history(
                select_players(players, args.players),
                os.environ["ENDZONE_API_KEY"],
                checkpoint=checkpoint,
            )
            checkpoint(history)
        else:
            history = json.loads(args.history.read_text(encoding="utf-8"))
        normalized = normalize_players(schedules, players)
        games, stats, quality = source_frames(schedules, normalized)
        stats = add_opportunity_columns(stats, normalized)
        forecasts, fit = fit_fixed_candidate(games, stats)
        quotes, audit = import_histories(history)
        report.update(
            score_quotes(forecasts, quotes, audit),
            fit=fit,
            source_quality=quality,
            history_sha256=hashlib.sha256(args.history.read_bytes()).hexdigest(),
            source="PROVIDER_REPORTED_DRAFTKINGS_PROP_HISTORY",
        )
    except (ValueError, RuntimeError, OSError) as error:
        report["reason"] = str(error)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {k: report[k] for k in ("status", "validated_edges", "promotion_earned")}
            | {"reason": report.get("reason"), "artifact": str(args.out)}
        )
    )
    if report["status"] == "SOURCE_BLOCKED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
