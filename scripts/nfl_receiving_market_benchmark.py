"""Offline real-price benchmark from public-web evidence exports. No API/key access."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import nflreadpy as nfl
import polars as pl

from nfl.prop_challenger import source_frames
from nfl.receiving_market_benchmark import fit_fixed_candidate, import_histories, score_quotes
from nfl.web_model_intake import approved_url
from scripts.nfl_receiving_usage_edge_test import normalize_players
from scripts.usage_prop_research import add_opportunity_columns


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--history", type=Path, required=True)
    p.add_argument("--cache-prefix", type=Path)
    p.add_argument("--out", type=Path, default=Path("reports/receiving_real_price_benchmark.json"))
    a = p.parse_args()
    report = {
        "status": "SOURCE_BLOCKED",
        "validated_edges": 0,
        "promotion_earned": False,
        "betting_authorized": False,
        "api_keys_required": False,
        "captured_utc": datetime.now(UTC).isoformat(),
    }
    try:
        history = json.loads(a.history.read_text())
        urls = history.get("source_urls")
        if (
            history.get("source_kind") != "PUBLIC_WEB_RESEARCH_HISTORY"
            or not isinstance(urls, list)
            or not urls
            or not all(isinstance(u, str) and approved_url(u) for u in urls)
        ):
            raise ValueError("PUBLIC_WEB_SOURCE_PROVENANCE_REQUIRED")
        if a.cache_prefix:
            schedules = pl.read_parquet(a.cache_prefix / "schedules_research.parquet")
            players = pl.read_parquet(a.cache_prefix / "player_research.parquet")
        else:
            schedules = nfl.load_schedules([2024, 2025])
            players = nfl.load_player_stats([2024, 2025])
        normalized = normalize_players(schedules, players)
        games, stats, quality = source_frames(schedules, normalized)
        stats = add_opportunity_columns(stats, normalized)
        forecasts, fit = fit_fixed_candidate(games, stats)
        quotes, audit = import_histories(history)
        report.update(
            score_quotes(forecasts, quotes, audit),
            fit=fit,
            source_quality=quality,
            history_sha256=hashlib.sha256(a.history.read_bytes()).hexdigest(),
            source="IMPORTED_PUBLIC_WEB_HISTORY",
        )
    except (ValueError, RuntimeError, OSError) as error:
        report["reason"] = str(error)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "validated_edges", "api_keys_required")}))
    if report["status"] == "SOURCE_BLOCKED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
