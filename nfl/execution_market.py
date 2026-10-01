"""Execution-market validation kept downstream of NFL pricing research."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from math import isfinite


def _finite(value: object) -> bool:
    try:
        return isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _parse_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif value is None:
        return None
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def validate_execution_row(
    row: Mapping[str, object],
    *,
    limits: Mapping[str, object] | None = None,
    now: datetime | None = None,
) -> tuple[bool, str]:
    """Fail closed when a proposed NFL bet lacks executable price provenance."""

    config = limits or {}
    if row.get("market_execution_verified") is False:
        return False, "market source is research-only and not executable"
    if row.get("market_quote_timestamp_verified") is False:
        return False, "market source lacks a verified quote timestamp"

    market = str(row.get("quant_market") or "").lower()
    side = str(row.get("quant_side") or "").strip()
    if market not in {"moneyline", "spread", "total"}:
        return False, "unsupported or missing market"
    if not side:
        return False, "missing market side"
    if not _finite(row.get("quant_odds")):
        return False, "missing executable American odds"
    odds = int(round(float(row["quant_odds"])))
    if odds == 0 or -100 < odds < 100:
        return False, "invalid executable American odds"
    if market in {"spread", "total"} and not _finite(row.get("quant_price")):
        return False, "missing executable line"

    book = str(row.get("quant_book") or "").strip()
    if bool(config.get("require_executable_book", True)) and not book:
        return False, "missing executable sportsbook provenance"

    minimum_books = max(
        1,
        int(float(config.get("min_market_book_count_for_execution", 1) or 1)),
    )
    if not _finite(row.get("market_book_count")):
        return False, "missing market book count"
    if int(float(row["market_book_count"])) < minimum_books:
        return False, f"market book count below {minimum_books}"

    if bool(config.get("require_quote_timestamp_for_execution", True)):
        quote_at = _parse_datetime(row.get("quant_quote_at"))
        if quote_at is None:
            return False, "missing or invalid quote timestamp"
        kickoff = _parse_datetime(row.get("kickoff", row.get("date")))
        if kickoff is None:
            return False, "missing kickoff timestamp for quote validation"
        if quote_at >= kickoff:
            return False, "executable quote is not strictly pre-kickoff"

        reference = (now or datetime.now(UTC)).astimezone(UTC)
        if quote_at > reference:
            return False, "quote timestamp is in the future"
        max_age = float(config.get("max_quote_age_minutes", 60) or 60)
        age_minutes = (reference - quote_at).total_seconds() / 60.0
        if age_minutes > max_age:
            return False, f"quote is stale ({age_minutes:.1f} minutes old)"

    return True, "executable quote verified"
