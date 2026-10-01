"""Canonical NFL market-intelligence rows aligned with the CFB operational schema."""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from .contracts import DataContractError, require_columns
from .espn_market import ESPNTwoWayMarket
from .market import MarketQuote, compare_two_way_market
from .policy import fractional_kelly_units, load_policy, signal_from_policy
from .probability import GaussianScoreDistribution


def _kickoff_map(targets: pl.DataFrame) -> dict[str, object]:
    require_columns(targets, {"game_id", "gameday"}, "current_targets")
    mapping: dict[str, object] = {}
    for row in targets.iter_rows(named=True):
        game_id = str(row["game_id"])
        kickoff = row.get("gameday")
        if row.get("gametime") not in {None, ""}:
            kickoff = f"{row['gameday']}T{row['gametime']}"
        mapping[game_id] = kickoff
    return mapping


def build_market_intelligence(
    projection: pl.DataFrame,
    targets: pl.DataFrame,
    markets: Sequence[ESPNTwoWayMarket],
    historical: pl.DataFrame,
    *,
    policy: dict[str, object] | None = None,
) -> tuple[pl.DataFrame, dict[str, object]]:
    """Select one downstream research opportunity per game/market.

    This is the NFL equivalent of the CFB canonical market-intelligence layer. The
    football projection and probability distribution already exist before quotes are
    evaluated. No sportsbook value is fed back into football ratings.
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
    kickoff = _kickoff_map(targets)
    rows: list[dict[str, object]] = []

    for market in markets:
        game = projected.get(market.game_id)
        if game is None:
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
            projected_home_margin=float(game["baseline_home_margin"]),
            projected_total=float(game["baseline_total"]),
            first=first,
            second=second,
        )
        chosen = max(
            pair,
            key=lambda value: (
                value.expected_value_per_unit,
                value.probability_edge,
                value.american_odds,
            ),
        )
        signal = signal_from_policy(
            chosen.expected_value_per_unit,
            chosen.probability_edge,
            chosen.model_probability,
            chosen.market_type,
            week=int(game["week"]),
            policy=active,
        )
        stake = 0.0
        if signal != "PASS":
            stake = fractional_kelly_units(
                chosen.model_probability,
                chosen.american_odds,
                kelly_fraction=kelly_fraction,
                max_units=max_units,
            )
        home_probability = distribution.home_win_probability(float(game["baseline_home_margin"]))

        row: dict[str, object] = {
            "season": int(game["season"]),
            "week": int(game["week"]),
            "game_id": market.game_id,
            "date": game.get("gameday"),
            "kickoff": kickoff.get(market.game_id),
            "away_team": game["away_team"],
            "home_team": game["home_team"],
            "model_margin_home": float(game["baseline_home_margin"]),
            "model_total": float(game["baseline_total"]),
            "calibrated_home_probability": home_probability,
            "quant_signal": signal,
            "quant_market": chosen.market_type,
            "quant_side": chosen.side,
            "quant_book": chosen.book,
            "quant_price": chosen.line,
            "quant_odds": chosen.american_odds,
            "quant_quote_at": market.captured_at.isoformat(),
            "quant_probability": chosen.model_probability,
            "quant_market_probability": chosen.no_vig_probability,
            "quant_ev": chosen.expected_value_per_unit,
            "quant_edge": chosen.probability_edge,
            "market_book_count": 1,
            "market_provider": market.provider,
            "source_event_id": market.source_event_id,
            "stake_units": stake,
        }
        for key, value in game.items():
            if key.startswith("home_qb_") or key.startswith("away_qb_") or key.startswith("qb_"):
                row[key] = value
        rows.append(row)

    frame = pl.DataFrame(rows) if rows else pl.DataFrame()
    market_counts = {market: 0 for market in ("moneyline", "spread", "total")}
    if not frame.is_empty():
        for market_name in market_counts:
            market_counts[market_name] = frame.filter(
                pl.col("quant_market") == market_name
            ).height
    games = projection.height
    metadata = {
        "games": games,
        **market_counts,
        "complete_market_coverage": (
            min(market_counts.values()) / games if games else 0.0
        ),
        "multi_book_coverage": 0.0,
        "market_source": "ESPN public endpoints",
        "api_key_required": False,
        "probability_training_games": historical.height,
        "release_state": "RESEARCH",
    }
    return frame.sort(["game_id", "quant_market"]) if not frame.is_empty() else frame, metadata
