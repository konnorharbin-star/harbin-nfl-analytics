from __future__ import annotations

import argparse
import json
from pathlib import Path

from nfl.backtest_audit import audit_backtest_bets
from nfl.backtest_runtime import write_backtest_runtime
from nfl.contracts import DataContractError
from nfl.data import NFLDataClient
from nfl.entry_integrity import annotate_historical_entry_integrity
from nfl.free_market import FreeNFLMarketStore, load_nflverse_initial_lines
from nfl.free_market_backtest import (
    build_archive_projection_dataset,
    build_free_archive_bets,
    evaluate_archive_holdout,
    summarize_archive_bets,
)
from nfl.proof import write_evidence_report


def _initial_line_coverage(
    initial_lines: object,
    *,
    start_season: int,
    end_season: int,
) -> dict[str, object]:
    if initial_lines is None:
        return {
            "available": False,
            "rows": 0,
            "seasons": [],
            "market_types": [],
            "backtest_window_rows": 0,
            "overlaps_backtest_window": False,
        }

    season_column = initial_lines.get_column("season").drop_nulls()
    seasons = sorted({int(value) for value in season_column.to_list()})
    market_types = sorted(
        {
            str(value).upper()
            for value in initial_lines.get_column("type").drop_nulls().to_list()
        }
    )
    window_rows = sum(
        start_season <= int(value) <= end_season for value in season_column.to_list()
    )
    return {
        "available": True,
        "rows": initial_lines.height,
        "seasons": seasons,
        "first_season": seasons[0] if seasons else None,
        "last_season": seasons[-1] if seasons else None,
        "market_types": market_types,
        "backtest_window_rows": window_rows,
        "overlaps_backtest_window": window_rows > 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the free nflverse NFL market archive backtest."
    )
    parser.add_argument("--start-season", type=int, default=2022)
    parser.add_argument("--end-season", type=int, default=2025)
    parser.add_argument("--validation-season", type=int, default=2024)
    parser.add_argument("--holdout-season", type=int, default=2025)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--reports-dir", default="reports")
    args = parser.parse_args()

    if args.start_season >= args.end_season:
        raise SystemExit("--start-season must be earlier than --end-season")
    if args.validation_season >= args.holdout_season:
        raise SystemExit("--validation-season must be earlier than --holdout-season")

    client = NFLDataClient()
    seasons = list(range(args.start_season - 1, args.end_season + 1))
    schedules = client.load_schedules(seasons, refresh=args.refresh)

    initial_error: str | None = None
    try:
        initial_lines = load_nflverse_initial_lines(refresh=args.refresh)
    except DataContractError as exc:
        initial_lines = None
        initial_error = str(exc)
    initial_coverage = _initial_line_coverage(
        initial_lines,
        start_season=args.start_season,
        end_season=args.end_season,
    )

    market_store = FreeNFLMarketStore(schedules, initial_lines=initial_lines)
    projections = build_archive_projection_dataset(
        schedules,
        start_season=args.start_season,
        end_season=args.end_season,
    )
    bets = annotate_historical_entry_integrity(
        build_free_archive_bets(projections, market_store)
    )
    if bets.is_empty():
        raise SystemExit("free archive backtest produced no market opportunities")

    evaluations = {}
    for market_type in ("moneyline", "spread", "total"):
        evaluations[market_type] = evaluate_archive_holdout(
            bets,
            market_type=market_type,
            validation_season=args.validation_season,
            holdout_season=args.holdout_season,
        ).to_dict()

    by_market = {
        market_type: summarize_archive_bets(
            bets.filter(bets.get_column("market_type") == market_type)
        ).to_dict()
        for market_type in ("moneyline", "spread", "total")
    }
    by_season = {
        str(season): summarize_archive_bets(
            bets.filter(bets.get_column("season") == season)
        ).to_dict()
        for season in range(args.start_season, args.end_season + 1)
        if bets.filter(bets.get_column("season") == season).height
    }

    reports = Path(args.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    projections.write_csv(reports / "free_market_predictions.csv")
    bets.write_csv(reports / "free_market_bets.csv")
    backtest_audit = audit_backtest_bets(bets)
    (reports / "backtest_audit.json").write_text(
        json.dumps(backtest_audit, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if backtest_audit["errors"]:
        raise RuntimeError("historical quote-integrity audit failed")
    runtime = write_backtest_runtime(bets, projections, reports_dir=reports)

    payload = {
        "source": {
            "historical_market": "nflverse schedules/nfldata public archive",
            "opening_market": "nflverse nfldata initial_lines.csv when available",
            "api_key_required": False,
            "initial_lines_error": initial_error,
            "initial_lines_coverage": initial_coverage,
            "clv_label": "opening-to-archive-final CLV proxy",
            "entry_price_rule": (
                "Only the exact price observed at entry can qualify as verified entry-price "
                "evidence. Opening spread/total lines without opening juice remain research-only."
            ),
            "notes": (
                "Archive-final values are not claimed to be timestamped official closes. "
                "When a distinct opening is unavailable, the final archive value is an "
                "explicit execution fallback and no CLV proxy is recorded."
            ),
        },
        "start_season": args.start_season,
        "end_season": args.end_season,
        "validation_season": args.validation_season,
        "holdout_season": args.holdout_season,
        "projection_games": projections.height,
        "market_opportunities": bets.height,
        "overall": summarize_archive_bets(bets).to_dict(),
        "by_market": by_market,
        "by_season": by_season,
        "holdout_evaluations": evaluations,
        "runtime_diagnostics": runtime,
        "quote_integrity": backtest_audit["quote_integrity"],
        "backtest_audit_status": backtest_audit["status"],
    }
    output = reports / "free_market_backtest.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    evidence = write_evidence_report(
        output=reports / "evidence_report.json",
        backtest_path=output,
        bets_path=reports / "free_market_bets.csv",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(
        json.dumps(
            {
                "evidence_status": evidence["status"],
                "promotion_sample": evidence["promotion_sample"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
