"""Historical NFL entry-provenance classification.

nflverse's current free ``initial_lines.csv`` contains 2021 spread/total opening
lines only. It has no opening juice, timestamps, or 2022-2025 promotion-sample coverage.
The integrity layer remains future-compatible with a genuinely observed moneyline
opener, but the present nflverse source does not supply one.
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
