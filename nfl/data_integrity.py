"""Point-in-time NFL operational data-integrity diagnostics.

The integrity layer detects structural mismatches, invalid quote chronology, missing
core fields, duplicate games, and source failures. It never fills missing values with
invented data and can fail the release gate on hard structural violations.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import polars as pl

from .espn_market import ESPNTwoWayMarket
from .schedule_market import is_research_only_market

NFL_SCHEDULE_TIMEZONE = ZoneInfo("America/New_York")


def _kickoff_map(targets: pl.DataFrame) -> dict[str, datetime]:
    output: dict[str, datetime] = {}
    if not {"game_id", "gameday", "gametime"}.issubset(targets.columns):
        return output
    for row in targets.iter_rows(named=True):
        if row.get("gameday") is None or row.get("gametime") in {None, ""}:
            continue
        try:
            local = datetime.fromisoformat(
                f"{row['gameday']}T{row['gametime']}"
            )
        except ValueError:
            continue
        if local.tzinfo is None:
            local = local.replace(tzinfo=NFL_SCHEDULE_TIMEZONE)
        output[str(row["game_id"])] = local.astimezone(UTC)
    return output


def _duplicates(frame: pl.DataFrame, column: str) -> list[str]:
    if frame.is_empty() or column not in frame.columns:
        return []
    values = [str(value) for value in frame.get_column(column).drop_nulls().to_list()]
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


def _missing_core_rows(
    frame: pl.DataFrame,
    required: set[str],
) -> tuple[int, list[str]]:
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        return frame.height, missing_columns
    if frame.is_empty():
        return 0, []
    expression = None
    for column in sorted(required):
        condition = pl.col(column).is_null()
        expression = condition if expression is None else (expression | condition)
    assert expression is not None
    return frame.filter(expression).height, []


def assess_data_integrity(
    projection: pl.DataFrame,
    targets: pl.DataFrame,
    markets: Sequence[ESPNTwoWayMarket],
    candidates: pl.DataFrame,
    *,
    source_errors: Sequence[str] = (),
    now: datetime | None = None,
) -> dict[str, object]:
    """Return machine-readable integrity state for the current model run."""

    reference = now or datetime.now(UTC)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=UTC)
    reference = reference.astimezone(UTC)

    errors: list[str] = []
    warnings: list[str] = [str(value) for value in source_errors if str(value)]
    projection_required = {
        "season",
        "week",
        "game_id",
        "home_team",
        "away_team",
        "baseline_home_margin",
        "baseline_total",
    }
    target_required = {
        "season",
        "week",
        "game_id",
        "home_team",
        "away_team",
        "gameday",
        "gametime",
    }
    projection_missing, projection_missing_columns = _missing_core_rows(
        projection, projection_required
    )
    target_missing, target_missing_columns = _missing_core_rows(
        targets, target_required
    )
    if projection_missing_columns:
        errors.append(
            "projection missing required columns: "
            + ", ".join(projection_missing_columns)
        )
    if target_missing_columns:
        errors.append(
            "targets missing required columns: "
            + ", ".join(target_missing_columns)
        )
    if projection_missing:
        errors.append(f"{projection_missing} projection rows have null core fields")
    if target_missing:
        errors.append(f"{target_missing} target rows have null core fields")

    projection_duplicates = _duplicates(projection, "game_id")
    target_duplicates = _duplicates(targets, "game_id")
    if projection_duplicates:
        errors.append("duplicate projection game_ids: " + ", ".join(projection_duplicates))
    if target_duplicates:
        errors.append("duplicate target game_ids: " + ", ".join(target_duplicates))

    projection_games = (
        {str(value) for value in projection.get_column("game_id").drop_nulls().to_list()}
        if "game_id" in projection.columns
        else set()
    )
    target_games = (
        {str(value) for value in targets.get_column("game_id").drop_nulls().to_list()}
        if "game_id" in targets.columns
        else set()
    )
    missing_projection_games = sorted(target_games - projection_games)
    extra_projection_games = sorted(projection_games - target_games)
    if missing_projection_games:
        errors.append(
            "target games missing from projection: " + ", ".join(missing_projection_games)
        )
    if extra_projection_games:
        errors.append(
            "projection contains non-target games: " + ", ".join(extra_projection_games)
        )

    kickoff = _kickoff_map(targets)
    unmatched_markets = 0
    future_quotes = 0
    post_kickoff_quotes = 0
    verified_quotes = 0
    market_signatures: set[tuple[object, ...]] = set()
    duplicate_market_rows = 0
    for market in markets:
        if market.game_id not in target_games:
            unmatched_markets += 1
        signature = (
            market.game_id,
            market.market_type,
            market.provider,
            market.book,
            market.first_side,
            market.first_line,
            market.first_american_odds,
            market.second_side,
            market.second_line,
            market.second_american_odds,
            market.captured_at.isoformat(),
        )
        if signature in market_signatures:
            duplicate_market_rows += 1
        market_signatures.add(signature)
        if is_research_only_market(market):
            continue
        verified_quotes += 1
        stamp = market.captured_at
        if stamp.tzinfo is None:
            errors.append(
                f"verified market quote has naive timestamp: {market.game_id}/{market.market_type}"
            )
            continue
        stamp = stamp.astimezone(UTC)
        if stamp > reference:
            future_quotes += 1
        game_kickoff = kickoff.get(market.game_id)
        if game_kickoff is not None and stamp >= game_kickoff:
            post_kickoff_quotes += 1

    if unmatched_markets:
        warnings.append(f"{unmatched_markets} market rows do not match target games")
    if duplicate_market_rows:
        warnings.append(f"{duplicate_market_rows} exact duplicate market rows detected")
    if future_quotes:
        errors.append(f"{future_quotes} verified quotes are timestamped in the future")
    if post_kickoff_quotes:
        errors.append(f"{post_kickoff_quotes} verified quotes are not pre-kickoff")
    if target_games and verified_quotes == 0:
        warnings.append("no verified live sportsbook quotes are available")

    candidate_games = (
        {str(value) for value in candidates.get_column("game_id").drop_nulls().to_list()}
        if "game_id" in candidates.columns
        else set()
    )
    games_without_candidate_rows = sorted(target_games - candidate_games)
    if games_without_candidate_rows:
        warnings.append(
            f"{len(games_without_candidate_rows)} target games have no market-intelligence rows"
        )

    status = "FAIL" if errors else "WARN" if warnings else "OK"
    return {
        "status": status,
        "projection_games": projection.height,
        "target_games": targets.height,
        "market_rows": len(markets),
        "candidate_rows": candidates.height,
        "missing_data_count": projection_missing + target_missing,
        "duplicate_game_count": len(projection_duplicates) + len(target_duplicates),
        "duplicate_market_rows": duplicate_market_rows,
        "unmatched_market_rows": unmatched_markets,
        "verified_quote_rows": verified_quotes,
        "future_verified_quote_rows": future_quotes,
        "post_kickoff_verified_quote_rows": post_kickoff_quotes,
        "games_without_candidate_rows": games_without_candidate_rows,
        "missing_projection_games": missing_projection_games,
        "extra_projection_games": extra_projection_games,
        "errors": list(dict.fromkeys(errors)),
        "warnings": list(dict.fromkeys(warnings)),
        "source_health": {
            "status": "WARN" if source_errors else "OK",
            "errors": [str(value) for value in source_errors],
        },
        "meaning": (
            "current-run structural and chronological data integrity; missing values are "
            "never fabricated"
        ),
    }
