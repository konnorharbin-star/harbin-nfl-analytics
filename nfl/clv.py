"""Closing-line value diagnostics for point-in-time NFL market observations.

CLV is computed only from observed decision and later pre-kickoff closing snapshots.
No closing quote is allowed to feed back into the football score or probability model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import polars as pl

from .contracts import DataContractError, require_columns
from .market import remove_two_way_vig
from .market_history import SNAPSHOT_KEYS, _complete_snapshot_summary, validate_market_history

DECISION_COMPARISON_REQUIRED = {
    "game_id",
    "market_type",
    "side",
    "line",
    "american_odds",
    "no_vig_probability",
    "probability_edge",
    "expected_value_per_unit",
    "provider",
    "book",
    "captured_at",
    "snapshot_id",
}


@dataclass(frozen=True)
class CLVSummary:
    observations: int
    positive_probability_clv_rate: float
    average_probability_clv: float
    average_spread_line_clv: float | None
    average_total_line_clv: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def closing_quote_probabilities(closing_quotes: pl.DataFrame) -> pl.DataFrame:
    """Attach proportional two-way no-vig probabilities to complete closing snapshots."""

    if closing_quotes.is_empty():
        return pl.DataFrame()
    validate_market_history(closing_quotes)
    complete = _complete_snapshot_summary(closing_quotes)
    if complete.is_empty():
        return pl.DataFrame()
    valid = closing_quotes.join(complete.select(SNAPSHOT_KEYS), on=SNAPSHOT_KEYS, how="inner")

    output: list[dict[str, object]] = []
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in valid.iter_rows(named=True):
        key = tuple(row[column] for column in SNAPSHOT_KEYS)
        grouped.setdefault(key, []).append(row)

    for key in sorted(grouped, key=lambda value: tuple(str(item) for item in value)):
        rows = grouped[key]
        if len(rows) != 2:
            raise DataContractError("closing snapshot is not a complete two-way market")
        first, second = rows
        first_probability, second_probability = remove_two_way_vig(
            int(first["american_odds"]),
            int(second["american_odds"]),
        )
        for row, probability in ((first, first_probability), (second, second_probability)):
            output.append(
                {
                    **row,
                    "closing_no_vig_probability": probability,
                }
            )
    return pl.DataFrame(output).sort(
        ["game_id", "provider", "book", "market_type", "snapshot_id", "side"]
    )


def _line_clv(market_type: str, side: str, decision_line: object, closing_line: object) -> float | None:
    if market_type == "moneyline":
        return None
    if decision_line is None or closing_line is None:
        raise DataContractError(f"{market_type} CLV requires decision and closing lines")
    decision = float(decision_line)
    closing = float(closing_line)
    if market_type == "spread":
        return decision - closing
    if market_type == "total" and side == "over":
        return closing - decision
    if market_type == "total" and side == "under":
        return decision - closing
    raise DataContractError(f"unsupported market/side for CLV: {market_type}/{side}")


def attach_closing_line_value(
    decision_comparisons: pl.DataFrame,
    closing_quotes: pl.DataFrame,
) -> pl.DataFrame:
    """Join each decision-side quote to the same book/market/side near kickoff."""

    require_columns(
        decision_comparisons,
        DECISION_COMPARISON_REQUIRED,
        "decision_market_comparisons",
    )
    if decision_comparisons.is_empty() or closing_quotes.is_empty():
        return pl.DataFrame()
    closing = closing_quote_probabilities(closing_quotes)
    if closing.is_empty():
        return pl.DataFrame()

    closing_rows = closing.select(
        "game_id",
        "provider",
        "book",
        "market_type",
        "side",
        pl.col("line").alias("closing_line"),
        pl.col("american_odds").alias("closing_american_odds"),
        pl.col("captured_at").alias("closing_captured_at"),
        pl.col("snapshot_id").alias("closing_snapshot_id"),
        "closing_no_vig_probability",
    )
    joined = decision_comparisons.join(
        closing_rows,
        on=["game_id", "provider", "book", "market_type", "side"],
        how="inner",
    )
    if joined.is_empty():
        return joined

    rows: list[dict[str, object]] = []
    for row in joined.iter_rows(named=True):
        decision_time = row["captured_at"]
        closing_time = row["closing_captured_at"]
        if closing_time < decision_time:
            raise DataContractError("closing quote predates the decision quote")
        row["probability_clv"] = float(row["closing_no_vig_probability"]) - float(
            row["no_vig_probability"]
        )
        row["line_clv"] = _line_clv(
            str(row["market_type"]),
            str(row["side"]),
            row["line"],
            row["closing_line"],
        )
        rows.append(row)
    return pl.DataFrame(rows).sort(["game_id", "book", "market_type", "side"])


def summarize_clv(frame: pl.DataFrame) -> CLVSummary:
    require_columns(frame, {"market_type", "probability_clv", "line_clv"}, "clv_frame")
    if frame.is_empty():
        return CLVSummary(
            observations=0,
            positive_probability_clv_rate=0.0,
            average_probability_clv=0.0,
            average_spread_line_clv=None,
            average_total_line_clv=None,
        )
    spread = frame.filter(pl.col("market_type") == "spread").get_column("line_clv").drop_nulls()
    total = frame.filter(pl.col("market_type") == "total").get_column("line_clv").drop_nulls()
    return CLVSummary(
        observations=frame.height,
        positive_probability_clv_rate=float((frame.get_column("probability_clv") > 0).mean()),
        average_probability_clv=float(frame.get_column("probability_clv").mean()),
        average_spread_line_clv=None if spread.is_empty() else float(spread.mean()),
        average_total_line_clv=None if total.is_empty() else float(total.mean()),
    )
