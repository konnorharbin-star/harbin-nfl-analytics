"""Point-in-time sportsbook quote history with explicit provenance.

Historical market rows are kept strictly downstream of the independent football model.
Every quote must identify when it was captured, where it came from, the sportsbook,
and the provider event/snapshot identifiers needed to reproduce the observation.
Selection is always as-of an explicit decision time; later quotes are never eligible.
"""

from __future__ import annotations

import polars as pl

from .contracts import DataContractError, require_columns, require_unique

MARKET_HISTORY_REQUIRED = {
    "game_id",
    "market_type",
    "side",
    "line",
    "american_odds",
    "provider",
    "book",
    "captured_at",
    "snapshot_id",
    "source_event_id",
}
DECISION_REQUIRED = {"game_id", "decision_time"}
SNAPSHOT_KEYS = ["game_id", "provider", "book", "market_type", "snapshot_id"]
QUOTE_KEYS = [*SNAPSHOT_KEYS, "side"]
LATEST_MARKET_KEYS = ["game_id", "provider", "book", "market_type"]


def validate_market_history(frame: pl.DataFrame) -> None:
    """Validate the minimum reproducible historical sportsbook quote contract."""

    require_columns(frame, MARKET_HISTORY_REQUIRED, "market_history")
    if frame.is_empty():
        raise DataContractError("market_history is empty")

    nonnull = MARKET_HISTORY_REQUIRED.difference({"line"})
    for column in sorted(nonnull):
        if frame.get_column(column).null_count():
            raise DataContractError(f"market_history contains null {column} values")

    for column in ("game_id", "provider", "book", "snapshot_id", "source_event_id"):
        empty = frame.filter(pl.col(column).cast(pl.Utf8).str.strip_chars() == "")
        if empty.height:
            raise DataContractError(f"market_history contains empty {column} values")

    allowed = (
        ((pl.col("market_type") == "moneyline") & pl.col("side").is_in(["home", "away"]))
        | ((pl.col("market_type") == "spread") & pl.col("side").is_in(["home", "away"]))
        | ((pl.col("market_type") == "total") & pl.col("side").is_in(["over", "under"]))
    )
    if frame.filter(~allowed).height:
        raise DataContractError("market_history contains invalid market_type/side combinations")

    invalid_odds = frame.filter(
        (pl.col("american_odds") == 0)
        | ((pl.col("american_odds") > -100) & (pl.col("american_odds") < 100))
    )
    if invalid_odds.height:
        raise DataContractError("market_history contains invalid American odds")

    invalid_lines = frame.filter(
        ((pl.col("market_type") == "moneyline") & pl.col("line").is_not_null())
        | ((pl.col("market_type") != "moneyline") & pl.col("line").is_null())
    )
    if invalid_lines.height:
        raise DataContractError("market_history contains invalid market line nullability")

    require_unique(frame, QUOTE_KEYS, "market_history")

    consistency = frame.group_by(["game_id", "provider", "book", "snapshot_id"]).agg(
        pl.col("captured_at").n_unique().alias("captured_times"),
        pl.col("source_event_id").n_unique().alias("source_events"),
    )
    inconsistent = consistency.filter(
        (pl.col("captured_times") != 1) | (pl.col("source_events") != 1)
    )
    if inconsistent.height:
        raise DataContractError(
            "market_history snapshot_id must map to one capture time and source event"
        )


def validate_decisions(decisions: pl.DataFrame) -> None:
    """Validate one explicit simulated decision time per game."""

    require_columns(decisions, DECISION_REQUIRED, "market_decisions")
    if decisions.is_empty():
        raise DataContractError("market_decisions is empty")
    if decisions.get_column("game_id").null_count() or decisions.get_column(
        "decision_time"
    ).null_count():
        raise DataContractError("market_decisions contains null values")
    require_unique(decisions, ["game_id"], "market_decisions")


def _complete_snapshot_summary(eligible: pl.DataFrame) -> pl.DataFrame:
    summary = eligible.group_by(SNAPSHOT_KEYS).agg(
        pl.len().alias("quote_rows"),
        pl.col("side").n_unique().alias("side_count"),
        pl.col("line").sum().alias("line_sum"),
        pl.col("line").n_unique().alias("line_count"),
        pl.col("captured_at").min().alias("snapshot_captured_at"),
        pl.col("captured_at").max().alias("snapshot_captured_at_max"),
    )
    structurally_paired = (
        (pl.col("market_type") == "moneyline")
        | ((pl.col("market_type") == "spread") & (pl.col("line_sum").abs() <= 1e-9))
        | ((pl.col("market_type") == "total") & (pl.col("line_count") == 1))
    )
    return summary.filter(
        (pl.col("quote_rows") == 2)
        & (pl.col("side_count") == 2)
        & (pl.col("snapshot_captured_at") == pl.col("snapshot_captured_at_max"))
        & structurally_paired
    )


def select_market_snapshots_asof(
    history: pl.DataFrame,
    decisions: pl.DataFrame,
    *,
    max_quote_age_minutes: int | None = None,
) -> pl.DataFrame:
    """Select the latest complete same-snapshot market available at decision time.

    Sides are never mixed across provider snapshots. If the newest eligible snapshot is
    incomplete, selection falls back to the latest earlier *complete* snapshot instead
    of borrowing the missing side from a different capture.
    """

    validate_market_history(history)
    validate_decisions(decisions)
    if max_quote_age_minutes is not None and max_quote_age_minutes < 0:
        raise ValueError("max_quote_age_minutes must be >= 0")

    eligible = history.join(decisions, on="game_id", how="inner").filter(
        pl.col("captured_at") <= pl.col("decision_time")
    )
    if max_quote_age_minutes is not None:
        eligible = eligible.filter(
            (pl.col("decision_time") - pl.col("captured_at"))
            <= pl.duration(minutes=max_quote_age_minutes)
        )
    if eligible.is_empty():
        return eligible.sort(["game_id", "provider", "book", "market_type", "side"])

    complete = _complete_snapshot_summary(eligible)
    if complete.is_empty():
        return eligible.head(0).sort(
            ["game_id", "provider", "book", "market_type", "side"]
        )

    latest = (
        complete.sort([*LATEST_MARKET_KEYS, "snapshot_captured_at", "snapshot_id"])
        .unique(subset=LATEST_MARKET_KEYS, keep="last")
        .select(SNAPSHOT_KEYS)
    )
    selected = eligible.join(latest, on=SNAPSHOT_KEYS, how="inner")
    return selected.sort(
        ["game_id", "provider", "book", "market_type", "snapshot_id", "side"]
    )
