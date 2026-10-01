"""Historical NFL entry-provenance classification.

nflverse's free ``initial_lines.csv`` can provide an actual opening moneyline price,
but spread/total rows provide the opening *line* without opening juice. The research
backtest may still evaluate those line observations using the archive-final price, but
that combination is not promotion-quality entry-price evidence.
"""

from __future__ import annotations

import polars as pl

from .contracts import require_columns

ENTRY_INTEGRITY_REQUIRED = {
    "market_type",
    "has_distinct_open",
    "price_stage",
}


def annotate_historical_entry_integrity(frame: pl.DataFrame) -> pl.DataFrame:
    """Add explicit line/price provenance without changing any bet economics.

    ``entry_line_observed`` means a separate opening observation exists.
    ``entry_price_verified`` means the price actually used for the simulated bet is
    itself an observed opening price. With the current free nflverse source that is
    possible for moneyline only; spread/total opening lines use archive-final juice.
    """

    if frame.is_empty():
        return frame
    require_columns(frame, ENTRY_INTEGRITY_REQUIRED, "historical_entry_integrity")

    distinct = (
        pl.col("has_distinct_open")
        .cast(pl.Boolean, strict=False)
        .fill_null(False)
    )
    market = pl.col("market_type").cast(pl.String).str.to_lowercase()
    verified_price = distinct & (market == "moneyline")

    return frame.with_columns(
        distinct.alias("entry_line_observed"),
        verified_price.alias("entry_price_verified"),
        pl.when(verified_price)
        .then(pl.lit("archive_open_price"))
        .when(distinct)
        .then(pl.lit("archive_open_line_final_price"))
        .otherwise(pl.lit("archive_final_fallback"))
        .alias("entry_price_stage"),
    )
