"""Canonical NFL market-intelligence rows aligned with the CFB operational schema."""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from .book_identity import canonical_book_identity
from .contracts import DataContractError, require_columns
from .espn_market import ESPNTwoWayMarket
from .market import MarketComparison, MarketQuote, compare_two_way_market
from .policy import (
    DEFAULT_POLICY,
    fractional_kelly_units,
    load_policy,
    signal_from_policy,
)
from .pregame import kickoff_iso_map
from .probability import GaussianScoreDistribution
from .schedule_market import is_research_only_market


def _book_key(market: ESPNTwoWayMarket) -> str:
    return canonical_book_identity(market.book or market.provider)


def _pair_best(
    distribution: GaussianScoreDistribution,
    game: dict[str, object],
    market: ESPNTwoWayMarket,
) -> MarketComparison:
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
        projected_home_margin=float(game["baseline_home_margin"]),
        projected_total=float(game["baseline_total"]),
        first=first,
        second=second,
    )
    return max(
        pair,
        key=lambda value: (
            value.expected_value_per_unit,
            value.probability_edge,
            value.american_odds,
            value.side,
        ),
    )


def build_market_intelligence(
    projection: pl.DataFrame,
    targets: pl.DataFrame,
    markets: Sequence[ESPNTwoWayMarket],
    historical: pl.DataFrame,
    *,
    policy: dict[str, object] | None = None,
) -> tuple[pl.DataFrame, dict[str, object]]:
    """Line-shop downstream NFL opportunities without contaminating football ratings.

    Research-only schedule snapshots can be compared to the model when verified live
    sources are unavailable, but they never count as verified sportsbook breadth and
    their policy signal is forced to PASS for allocation/execution purposes.
    """

    require_columns(
        projection,
        {
            "season",
            "week",
            "game_id",
            "home_team",
            "away_team",
            "baseline_home_margin",
            "baseline_total",
        },
        "current_projection",
    )
    if historical.height < 64:
        raise DataContractError("market intelligence requires at least 64 prior games")

    active = policy or load_policy()
    portfolio = active.get("portfolio")
    if not isinstance(portfolio, dict):
        portfolio = {}
    kelly_fraction = float(portfolio.get("kelly_fraction", 0.20))
    max_units = float(portfolio.get("max_single_bet_units", 1.0))

    distribution = GaussianScoreDistribution().fit(historical)
    projected = {str(row["game_id"]): row for row in projection.iter_rows(named=True)}
    kickoff = kickoff_iso_map(targets)

    grouped: dict[
        tuple[str, str],
        list[tuple[MarketComparison, ESPNTwoWayMarket]],
    ] = {}
    verified_books_by_game: dict[str, set[str]] = {}
    for market in markets:
        game = projected.get(market.game_id)
        if game is None:
            continue
        chosen = _pair_best(distribution, game, market)
        grouped.setdefault((market.game_id, market.market_type), []).append((chosen, market))
        if not is_research_only_market(market):
            verified_books_by_game.setdefault(market.game_id, set()).add(_book_key(market))

    rows: list[dict[str, object]] = []
    for (game_id, _market_type), candidates in sorted(grouped.items()):
        game = projected[game_id]
        verified_books = sorted(
            {
                _book_key(market)
                for _, market in candidates
                if not is_research_only_market(market)
            }
        )
        chosen, source_market = max(
            candidates,
            key=lambda item: (
                item[0].expected_value_per_unit,
                item[0].probability_edge,
                item[0].american_odds,
                item[0].side,
                str(item[1].book),
            ),
        )
        research_signal = signal_from_policy(
            chosen.expected_value_per_unit,
            chosen.probability_edge,
            chosen.model_probability,
            chosen.market_type,
            week=int(game["week"]),
            policy=DEFAULT_POLICY,
        )
        production_signal = signal_from_policy(
            chosen.expected_value_per_unit,
            chosen.probability_edge,
            chosen.model_probability,
            chosen.market_type,
            week=int(game["week"]),
            policy=active,
        )
        execution_verified = not is_research_only_market(source_market)
        signal = production_signal if execution_verified else "PASS"
        research_stake = 0.0
        if research_signal != "PASS":
            research_stake = fractional_kelly_units(
                chosen.model_probability,
                chosen.american_odds,
                kelly_fraction=kelly_fraction,
                max_units=max_units,
            )
        stake = research_stake if signal != "PASS" else 0.0
        home_probability = distribution.home_win_probability(
            float(game["baseline_home_margin"])
        )

        row: dict[str, object] = {
            "season": int(game["season"]),
            "week": int(game["week"]),
            "game_id": game_id,
            "date": game.get("gameday"),
            "kickoff": kickoff.get(game_id),
            "away_team": game["away_team"],
            "home_team": game["home_team"],
            "model_margin_home": float(game["baseline_home_margin"]),
            "model_total": float(game["baseline_total"]),
            "calibrated_home_probability": home_probability,
            "quant_signal": signal,
            "research_signal": research_signal,
            "production_signal": production_signal,
            "quant_market": chosen.market_type,
            "quant_side": chosen.side,
            "quant_book": chosen.book,
            "quant_price": chosen.line,
            "quant_odds": chosen.american_odds,
            "quant_quote_at": source_market.captured_at.isoformat(),
            "quant_probability": chosen.model_probability,
            "quant_market_probability": chosen.no_vig_probability,
            "quant_ev": chosen.expected_value_per_unit,
            "quant_edge": chosen.probability_edge,
            "market_book_count": len(verified_books),
            "market_books": ";".join(verified_books),
            "market_provider": source_market.provider,
            "market_execution_verified": execution_verified,
            "market_quote_timestamp_verified": execution_verified,
            "market_source_role": "verified_live" if execution_verified else "research_fallback",
            "source_event_id": source_market.source_event_id,
            "stake_units": stake,
            "research_stake_units": research_stake,
        }
        for key, value in game.items():
            if key.startswith(("home_qb_", "away_qb_", "qb_", "recent_form_")):
                row[key] = value
        rows.append(row)

    frame = pl.DataFrame(rows) if rows else pl.DataFrame()
    research_counts = {market: 0 for market in ("moneyline", "spread", "total")}
    verified_counts = {market: 0 for market in ("moneyline", "spread", "total")}
    if not frame.is_empty():
        for market_name in research_counts:
            research_counts[market_name] = frame.filter(
                pl.col("quant_market") == market_name
            ).height
            verified_counts[market_name] = frame.filter(
                (pl.col("quant_market") == market_name)
                & pl.col("market_execution_verified")
            ).height
    games = projection.height
    multi_book_games = sum(
        len(books) >= 2 for books in verified_books_by_game.values()
    )
    metadata = {
        "games": games,
        **verified_counts,
        "research_moneyline": research_counts["moneyline"],
        "research_spread": research_counts["spread"],
        "research_total": research_counts["total"],
        "complete_market_coverage": (
            min(verified_counts.values()) / games if games else 0.0
        ),
        "research_market_coverage": (
            min(research_counts.values()) / games if games else 0.0
        ),
        "multi_book_games": multi_book_games,
        "multi_book_coverage": multi_book_games / games if games else 0.0,
        "distinct_books": len(
            {
                book
                for books in verified_books_by_game.values()
                for book in books
            }
        ),
        "research_fallback_rows": (
            frame.filter(~pl.col("market_execution_verified")).height
            if not frame.is_empty()
            else 0
        ),
        "verified_market_rows": (
            frame.filter(pl.col("market_execution_verified")).height
            if not frame.is_empty()
            else 0
        ),
        "market_source": "canonical current NFL market aggregation",
        "api_key_required": False,
        "probability_training_games": historical.height,
        "release_state": "RESEARCH",
    }
    return frame.sort(["game_id", "quant_market"]) if not frame.is_empty() else frame, metadata