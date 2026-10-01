"""Execution-market validation kept downstream of NFL pricing research."""

from __future__ import annotations

from datetime import UTC, datetime
from math import isfinite
from typing import Mapping


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

    if bool(config.get("require_quote_timestamp_for_execution", True)):
        quote_at = _parse_datetime(row.get("quant_quote_at"))
        if quote_at is None:
            return False, "missing or invalid quote timestamp"
        reference = (now or datetime.now(UTC)).astimezone(UTC)
        if quote_at > reference:
            return False, "quote timestamp is in the future"
        max_age = float(config.get("max_quote_age_minutes", 60) or 60)
        age_minutes = (reference - quote_at).total_seconds() / 60.0
        if age_minutes > max_age:
            return False, f"quote is stale ({age_minutes:.1f} minutes old)"

    return True, "executable quote verified"
