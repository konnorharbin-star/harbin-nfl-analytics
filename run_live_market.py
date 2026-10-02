"""Compare current NFL fair scores with free verified market sources."""

from __future__ import annotations

import argparse
import json

import polars as pl

from nfl.current import run_current_projection, unplayed_regular_games
from nfl.data import NFLDataClient
from nfl.free_market_backtest import build_archive_projection_dataset
from nfl.market import MarketQuote, compare_two_way_market
from nfl.pro_market import collect_current_markets
from nfl.probability import GaussianScoreDistribution


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("season", type=int)
    parser.add_argument("--week", type=int)
    parser.add_argument("--history-seasons", type=int, default=4)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    if args.history_seasons < 2:
        raise SystemExit("--history-seasons must be >= 2")

    client = NFLDataClient()
    projection, audit = run_current_projection(
        args.season,
        args.week,
        client=client,
        refresh=args.refresh,
    )
    target_week = audit.week

    history_start = args.season - args.history_seasons
    schedules = client.load_schedules(
        list(range(history_start - 1, args.season + 1)),
        refresh=args.refresh,
    )
    targets = unplayed_regular_games(schedules, args.season, target_week)
    markets, source_meta = collect_current_markets(
        targets,
        week=target_week,
    )

    historical = build_archive_projection_dataset(
        schedules,
        start_season=history_start,
        end_season=args.season,
    ).filter(
        (pl.col("season") < args.season)
        | ((pl.col("season") == args.season) & (pl.col("week") < target_week))
    )
    distribution = GaussianScoreDistribution().fit(historical)
    projection_map = {
        str(row["game_id"]): row for row in projection.iter_rows(named=True)
    }

    comparisons: list[dict[str, object]] = []
    for market in markets:
        row = projection_map.get(market.game_id)
        if row is None:
            continue
        first = MarketQuote(
            market_type=market.market_type,
            side=market.first_side,
            line=market.first_line,
            american_odds=market.first_american_odds,
            book=market.book,
            captured_at=market.captured_at,
        )
        second = MarketQuote(
            market_type=market.market_type,
            side=market.second_side,
            line=market.second_line,
            american_odds=market.second_american_odds,
            book=market.book,
            captured_at=market.captured_at,
        )
        pair = compare_two_way_market(
            distribution,
            projected_home_margin=float(row["baseline_home_margin"]),
            projected_total=float(row["baseline_total"]),
            first=first,
            second=second,
        )
        ranked = sorted(
            pair,
            key=lambda item: (
                item.expected_value_per_unit,
                item.probability_edge,
                item.american_odds,
            ),
            reverse=True,
        )
        selected_side = ranked[0].side
        for comparison in pair:
            output = comparison.to_dict()
            output.update(
                {
                    "season": args.season,
                    "week": target_week,
                    "game_id": market.game_id,
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "projected_home_margin": float(row["baseline_home_margin"]),
                    "projected_total": float(row["baseline_total"]),
                    "provider": market.provider,
                    "source_event_id": market.source_event_id,
                    "captured_at": market.captured_at.isoformat(),
                    "best_research_side": comparison.side == selected_side,
                    "release_state": "RESEARCH",
                }
            )
            comparisons.append(output)

    result = {
        "status": "READY",
        "season": args.season,
        "week": target_week,
        "market_source": "canonical current NFL market aggregation",
        "source_breadth": source_meta,
        "api_key_required": False,
        "football_projection": "canonical baseline",
        "market_release_state": "RESEARCH",
        "note": (
            "Best research side is the highest-EV side at an observed verified price. "
            "It is not a production betting recommendation or approved stake."
        ),
        "projection_audit": audit.to_dict(),
        "probability_training_games": historical.height,
        "markets": comparisons,
    }
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
