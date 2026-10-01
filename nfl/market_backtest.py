"""Chronological market comparison and grading over point-in-time sportsbook quotes.

The football distribution and football projections are inputs. Sportsbook prices enter
only after those objects already exist. This module preserves quote provenance through
comparison and grading so any historical result can be traced to the exact observed
provider snapshot and simulated decision time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import polars as pl

from .contracts import DataContractError, require_columns, require_unique
from .market import MarketQuote, compare_two_way_market
from .market_history import MARKET_HISTORY_REQUIRED, SNAPSHOT_KEYS, validate_market_history
from .probability import GaussianScoreDistribution

PROJECTION_REQUIRED = {"game_id", "projected_home_margin", "projected_total"}
OUTCOME_REQUIRED = {"game_id", "actual_home_margin", "actual_total"}
COMPARISON_REQUIRED = {
    "game_id",
    "market_type",
    "side",
    "line",
    "american_odds",
    "decimal_odds",
    "model_probability",
    "market_implied_probability",
    "no_vig_probability",
    "probability_edge",
    "expected_value_per_unit",
    "provider",
    "book",
    "captured_at",
    "snapshot_id",
    "source_event_id",
    "decision_time",
}


@dataclass(frozen=True)
class BacktestSummary:
    bets: int
    wins: int
    losses: int
    pushes: int
    net_units: float
    roi_per_unit_staked: float
    average_probability_edge: float
    average_expected_value_per_unit: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def compare_selected_market_snapshots(
    projections: pl.DataFrame,
    selected_quotes: pl.DataFrame,
    distribution: GaussianScoreDistribution,
) -> pl.DataFrame:
    """Compare complete as-of sportsbook snapshots with the football distribution."""

    require_columns(projections, PROJECTION_REQUIRED, "market_projection_input")
    require_unique(projections, ["game_id"], "market_projection_input")
    require_columns(selected_quotes, MARKET_HISTORY_REQUIRED | {"decision_time"}, "selected_quotes")
    if selected_quotes.is_empty():
        return pl.DataFrame()
    validate_market_history(selected_quotes.select(sorted(MARKET_HISTORY_REQUIRED)))

    projection_map = {
        str(row["game_id"]): row for row in projections.iter_rows(named=True)
    }
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in selected_quotes.iter_rows(named=True):
        key = tuple(row[column] for column in SNAPSHOT_KEYS)
        grouped.setdefault(key, []).append(row)

    output: list[dict[str, object]] = []
    for key in sorted(grouped, key=lambda value: tuple(str(item) for item in value)):
        rows = grouped[key]
        if len(rows) != 2:
            raise DataContractError("selected market snapshot is not a complete two-way pair")
        game_id = str(rows[0]["game_id"])
        if game_id not in projection_map:
            raise DataContractError(f"missing football projection for market game {game_id}")
        projection = projection_map[game_id]

        quotes = [
            MarketQuote(
                market_type=row["market_type"],
                side=row["side"],
                american_odds=int(row["american_odds"]),
                line=None if row["line"] is None else float(row["line"]),
                book=str(row["book"]),
                captured_at=row["captured_at"],
            )
            for row in rows
        ]
        comparisons = compare_two_way_market(
            distribution,
            projected_home_margin=float(projection["projected_home_margin"]),
            projected_total=float(projection["projected_total"]),
            first=quotes[0],
            second=quotes[1],
        )
        source_by_side = {str(row["side"]): row for row in rows}
        for comparison in comparisons:
            source = source_by_side[comparison.side]
            result = comparison.to_dict()
            result.update(
                {
                    "game_id": game_id,
                    "provider": str(source["provider"]),
                    "captured_at": source["captured_at"],
                    "snapshot_id": str(source["snapshot_id"]),
                    "source_event_id": str(source["source_event_id"]),
                    "decision_time": source["decision_time"],
                    "projected_home_margin": float(projection["projected_home_margin"]),
                    "projected_total": float(projection["projected_total"]),
                }
            )
            output.append(result)

    return pl.DataFrame(output).sort(
        ["game_id", "provider", "book", "market_type", "snapshot_id", "side"]
    )


def _grade_value(row: dict[str, object]) -> float:
    market_type = str(row["market_type"])
    side = str(row["side"])
    actual_margin = float(row["actual_home_margin"])
    actual_total = float(row["actual_total"])
    line = row["line"]

    if market_type == "moneyline":
        return actual_margin if side == "home" else -actual_margin
    if line is None:
        raise DataContractError(f"graded {market_type} row is missing a line")
    numeric_line = float(line)
    if market_type == "spread":
        return actual_margin + numeric_line if side == "home" else -actual_margin + numeric_line
    if market_type == "total":
        return actual_total - numeric_line if side == "over" else numeric_line - actual_total
    raise DataContractError(f"unsupported market_type while grading: {market_type}")


def grade_market_comparisons(
    comparisons: pl.DataFrame,
    outcomes: pl.DataFrame,
) -> pl.DataFrame:
    """Grade each quoted side at its actual executable historical price."""

    require_columns(comparisons, COMPARISON_REQUIRED, "market_comparisons")
    require_columns(outcomes, OUTCOME_REQUIRED, "market_outcomes")
    require_unique(outcomes, ["game_id"], "market_outcomes")
    if comparisons.is_empty():
        return comparisons

    joined = comparisons.join(
        outcomes.select(sorted(OUTCOME_REQUIRED)),
        on="game_id",
        how="left",
    )
    if joined.get_column("actual_home_margin").null_count() or joined.get_column(
        "actual_total"
    ).null_count():
        raise DataContractError("market grading is missing one or more game outcomes")

    rows: list[dict[str, object]] = []
    for row in joined.iter_rows(named=True):
        value = _grade_value(row)
        if value > 1e-12:
            result = "win"
            net_units = float(row["decimal_odds"]) - 1.0
        elif value < -1e-12:
            result = "loss"
            net_units = -1.0
        else:
            result = "push"
            net_units = 0.0
        row["result"] = result
        row["net_units"] = net_units
        rows.append(row)
    return pl.DataFrame(rows).sort(
        ["game_id", "provider", "book", "market_type", "snapshot_id", "side"]
    )


def summarize_qualified_bets(
    graded: pl.DataFrame,
    *,
    min_probability_edge: float,
    min_expected_value_per_unit: float,
) -> BacktestSummary:
    """Summarize an explicitly supplied research qualification rule.

    The thresholds are inputs rather than defaults so this helper cannot silently turn
    every positive-looking historical edge into a claimed betting strategy.
    """

    require_columns(
        graded,
        {"result", "net_units", "probability_edge", "expected_value_per_unit"},
        "graded_market_comparisons",
    )
    selected = graded.filter(
        (pl.col("probability_edge") >= min_probability_edge)
        & (pl.col("expected_value_per_unit") >= min_expected_value_per_unit)
    )
    if selected.is_empty():
        return BacktestSummary(
            bets=0,
            wins=0,
            losses=0,
            pushes=0,
            net_units=0.0,
            roi_per_unit_staked=0.0,
            average_probability_edge=0.0,
            average_expected_value_per_unit=0.0,
        )

    bets = selected.height
    wins = selected.filter(pl.col("result") == "win").height
    losses = selected.filter(pl.col("result") == "loss").height
    pushes = selected.filter(pl.col("result") == "push").height
    net_units = float(selected.get_column("net_units").sum())
    return BacktestSummary(
        bets=bets,
        wins=wins,
        losses=losses,
        pushes=pushes,
        net_units=net_units,
        roi_per_unit_staked=net_units / bets,
        average_probability_edge=float(selected.get_column("probability_edge").mean()),
        average_expected_value_per_unit=float(
            selected.get_column("expected_value_per_unit").mean()
        ),
    )
