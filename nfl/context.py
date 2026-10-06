"""Current-only NFL context layer aligned with the CFB platform philosophy.

Context is attached after the independent football projection. It may reduce confidence
or block release, but it does not alter the canonical score until a point-in-time
historical validation demonstrates that a specific adjustment improves NFL holdout data.
"""

from __future__ import annotations

from datetime import UTC, datetime
from math import isfinite

import polars as pl

from .contracts import DataContractError, require_columns
from .injuries import (
    injury_feed_freshness,
    normalize_injuries,
    summarize_team_injuries,
)
from .personnel import (
    normalize_depth_charts,
    normalize_rosters,
    summarize_team_personnel,
)
from .qb_current import (
    attach_expected_qb_context,
    build_expected_qb_state,
    qb_game_coverage,
)
from .weather import (
    TEAM_HOME,
    OpenMeteoNFLWeather,
    haversine_miles,
    is_indoor_roof,
    kickoff_utc,
    team_travel_context,
    weather_risk,
)


def current_nfl_season(as_of: datetime | None = None) -> int:
    stamp = (as_of or datetime.now(UTC)).astimezone(UTC)
    return stamp.year - 1 if stamp.month <= 2 else stamp.year


def _number(value: object) -> float | None:
    if value in {None, ""}:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if isfinite(numeric) else None


def _team_map(frame: pl.DataFrame) -> dict[str, dict[str, object]]:
    if frame.is_empty() or "team" not in frame.columns:
        return {}
    return {str(row["team"]): row for row in frame.iter_rows(named=True)}


def _neutral(row: dict[str, object]) -> bool:
    return "neutral" in str(row.get("location") or "").lower()


def _neutral_travel(
    *,
    home_team: str,
    away_team: str,
    coordinates: tuple[float, float] | None,
) -> dict[str, object]:
    if coordinates is None:
        return {
            "away_travel_miles": None,
            "home_travel_miles": None,
            "timezone_shift_hours": None,
            "travel_source_available": False,
        }
    home = TEAM_HOME.get(home_team)
    away = TEAM_HOME.get(away_team)
    if home is None or away is None:
        return {
            "away_travel_miles": None,
            "home_travel_miles": None,
            "timezone_shift_hours": None,
            "travel_source_available": False,
        }
    return {
        "away_travel_miles": haversine_miles(away[0], away[1], *coordinates),
        "home_travel_miles": haversine_miles(home[0], home[1], *coordinates),
        "timezone_shift_hours": None,
        "travel_source_available": True,
    }



CONTEXT_VETO_INJURY_GAP = 0.40
CONTEXT_VETO_STARTER_GAP = 0.25
CONTEXT_VETO_REST_GAP_DAYS = -2.0


def apply_context_freshness_veto(candidates: pl.DataFrame) -> pl.DataFrame:
    """Fail closed when current injury/personnel state is stale or unknown."""

    if candidates.is_empty():
        return candidates

    rows: list[dict[str, object]] = []
    required = {"context_injuries_personnel_fresh"}
    missing = not required.issubset(candidates.columns)
    for row in candidates.iter_rows(named=True):
        fresh = (
            False
            if missing
            else bool(row.get("context_injuries_personnel_fresh"))
        )
        veto = not fresh
        reason = str(row.get("context_freshness_reason") or "")
        if veto and not reason:
            reason = (
                "current injury/personnel freshness is unavailable"
                if missing
                else "injury/personnel context is stale or unknown"
            )
        item = dict(row)
        item["context_freshness_veto"] = veto
        item["context_freshness_veto_reason"] = reason if veto else ""
        if veto:
            for field in (
                "quant_signal",
                "production_signal",
                "research_signal",
                "portfolio_signal",
            ):
                if field in item:
                    item[field] = "PASS"
            for field in ("stake_units", "research_stake_units"):
                if field in item:
                    item[field] = 0.0
        rows.append(item)
    return pl.DataFrame(rows)


def apply_context_confidence_veto(candidates: pl.DataFrame) -> pl.DataFrame:
    """Fail closed on side bets facing stacked, severe adverse current context.

    This is deliberately a confidence/risk veto, not a score adjustment. It only
    suppresses moneyline/spread bet signals when current injury/personnel feeds and
    rest context all agree that the selected side is materially disadvantaged.
    """

    if candidates.is_empty():
        return candidates.with_columns(
            pl.lit(False).alias("context_veto"),
            pl.lit("").alias("context_veto_reason"),
        )

    required = {
        "quant_market",
        "quant_side",
        "context_injuries_personnel_fresh",
        "context_rest_travel_available",
        "home_injury_risk",
        "away_injury_risk",
        "home_starter_injury_risk",
        "away_starter_injury_risk",
        "home_rest_advantage_days",
    }
    if not required.issubset(candidates.columns):
        return candidates.with_columns(
            pl.lit(False).alias("context_veto"),
            pl.lit("").alias("context_veto_reason"),
        )

    rows: list[dict[str, object]] = []
    signal_fields = ("quant_signal", "production_signal", "research_signal", "portfolio_signal")
    stake_fields = ("stake_units", "research_stake_units")

    for row in candidates.iter_rows(named=True):
        veto = False
        reason = ""
        market = str(row.get("quant_market") or "").lower()
        side = str(row.get("quant_side") or "").lower()
        context_ready = bool(row.get("context_injuries_personnel_fresh")) and bool(
            row.get("context_rest_travel_available")
        )

        if context_ready and market in {"moneyline", "spread"} and side in {"home", "away"}:
            home_injury = float(row.get("home_injury_risk") or 0.0)
            away_injury = float(row.get("away_injury_risk") or 0.0)
            home_starter = float(row.get("home_starter_injury_risk") or 0.0)
            away_starter = float(row.get("away_starter_injury_risk") or 0.0)
            home_rest_adv = float(row.get("home_rest_advantage_days") or 0.0)

            if side == "home":
                injury_gap = home_injury - away_injury
                starter_gap = home_starter - away_starter
                selected_rest_gap = home_rest_adv
            else:
                injury_gap = away_injury - home_injury
                starter_gap = away_starter - home_starter
                selected_rest_gap = -home_rest_adv

            veto = (
                injury_gap >= CONTEXT_VETO_INJURY_GAP
                and starter_gap >= CONTEXT_VETO_STARTER_GAP
                and selected_rest_gap <= CONTEXT_VETO_REST_GAP_DAYS
            )
            if veto:
                reason = (
                    "stacked adverse context: selected side has "
                    f"{injury_gap:.2f} injury-risk gap, "
                    f"{starter_gap:.2f} starter-risk gap, and "
                    f"{selected_rest_gap:.1f} rest-day gap"
                )

        item = dict(row)
        item["context_veto"] = veto
        item["context_veto_reason"] = reason
        if veto:
            for field in signal_fields:
                if field in item:
                    item[field] = "PASS"
            for field in stake_fields:
                if field in item:
                    item[field] = 0.0
        rows.append(item)

    return pl.DataFrame(rows)

def build_current_context(
    targets: pl.DataFrame,
    *,
    season: int,
    week: int,
    injuries: pl.DataFrame | None = None,
    depth_charts: pl.DataFrame | None = None,
    rosters: pl.DataFrame | None = None,
    projection: pl.DataFrame | None = None,
    as_of: datetime | None = None,
    weather_client: OpenMeteoNFLWeather | None = None,
    allow_historical: bool = False,
) -> tuple[pl.DataFrame, dict[str, object]]:
    """Build one post-prediction context row per current target game.

    Historical calls fail closed by default because current context must not be
    backfilled into an old game. Explicit historical research may opt in only when the
    caller has supplied timestamp-correct injury/depth/roster/weather inputs.
    """

    require_columns(
        targets,
        {"season", "week", "game_id", "home_team", "away_team", "gameday"},
        "context_targets",
    )
    stamp = (as_of or datetime.now(UTC)).astimezone(UTC)
    expected = current_nfl_season(stamp)
    if season != expected and not allow_historical:
        message = (
            f"current context is disabled for historical season {season}; "
            f"current season is {expected}"
        )
        raise DataContractError(message)
    if week < 1:
        raise DataContractError("week must be >= 1")

    injury_source = injuries if injuries is not None else pl.DataFrame()
    depth_source = depth_charts if depth_charts is not None else pl.DataFrame()
    roster_source = rosters if rosters is not None else pl.DataFrame()

    injury_feed_state = (
        injury_feed_freshness(
            injury_source,
            season=season,
            week=week,
            as_of=stamp,
        )
        if not injury_source.is_empty()
        else {
            "status": "UNKNOWN",
            "current_week_rows": 0,
            "latest_reported_at": None,
            "latest_age_days": None,
            "reason": "injury feed is empty",
        }
    )

    normalized_injuries = (
        normalize_injuries(
            injury_source,
            season=season,
            week=week,
            as_of=stamp,
        )
        if not injury_source.is_empty()
        else pl.DataFrame()
    )
    injury_summary = summarize_team_injuries(normalized_injuries)

    normalized_depth = (
        normalize_depth_charts(
            depth_source,
            season=season,
            week=week,
            as_of=stamp,
        )
        if not depth_source.is_empty()
        else pl.DataFrame()
    )
    normalized_rosters = (
        normalize_rosters(roster_source, season=season, week=week)
        if not roster_source.is_empty()
        else pl.DataFrame()
    )
    personnel_summary = summarize_team_personnel(
        normalized_depth,
        normalized_rosters,
        normalized_injuries,
    )

    qb_state = pl.DataFrame()
    qb_state_meta: dict[str, object] = {
        "identity_coverage": 0.0,
        "decision_ready_team_coverage": 0.0,
        "status": "BLOCKED",
        "reason": "current projection was not supplied for QB reconciliation",
    }
    if projection is not None and not projection.is_empty():
        try:
            qb_state, qb_state_meta = build_expected_qb_state(
                targets,
                projection,
                depth=normalized_depth,
                rosters=normalized_rosters,
                injuries=normalized_injuries,
                as_of=stamp,
            )
            qb_state_meta["status"] = "READY"
        except Exception as exc:
            qb_state = pl.DataFrame()
            qb_state_meta = {
                "identity_coverage": 0.0,
                "decision_ready_team_coverage": 0.0,
                "status": "BLOCKED",
                "reason": f"{type(exc).__name__}: {exc}",
            }

    injury_map = _team_map(injury_summary)
    personnel_map = _team_map(personnel_summary)
    injury_feed_available = not injury_source.is_empty()
    depth_feed_available = not depth_source.is_empty()
    roster_feed_available = not roster_source.is_empty()
    weather = weather_client or OpenMeteoNFLWeather()

    component_counts = {
        "injuries_personnel": 0,
        "rest_travel": 0,
        "weather_stadium": 0,
    }
    weather_errors: list[str] = []
    rows: list[dict[str, object]] = []

    for game in targets.iter_rows(named=True):
        home_team = str(game["home_team"])
        away_team = str(game["away_team"])
        home_injury = injury_map.get(home_team, {})
        away_injury = injury_map.get(away_team, {})
        home_personnel = personnel_map.get(home_team, {})
        away_personnel = personnel_map.get(away_team, {})

        home_rest = _number(game.get("home_rest"))
        away_rest = _number(game.get("away_rest"))
        rest_available = home_rest is not None and away_rest is not None
        rest_diff = home_rest - away_rest if rest_available else None

        neutral = _neutral(game)
        stadium = game.get("stadium")
        roof = game.get("roof")
        indoor = is_indoor_roof(roof)
        kickoff = kickoff_utc(game.get("gameday"), game.get("gametime"))
        venue_error: str | None = None
        try:
            coordinates = weather.venue_coordinates(
                home_team=home_team,
                stadium=stadium,
                neutral=neutral,
            )
        except DataContractError as exc:
            coordinates = None
            venue_error = str(exc)

        travel = (
            _neutral_travel(
                home_team=home_team,
                away_team=away_team,
                coordinates=coordinates,
            )
            if neutral
            else team_travel_context(away_team, home_team)
        )
        travel_available = bool(travel["travel_source_available"])

        forecast: dict[str, object] = {}
        weather_available = False
        weather_error: str | None = venue_error
        if indoor:
            forecast = {
                "temperature_f": None,
                "precip_probability": 0.0,
                "wind_mph": 0.0,
                "wind_gust_mph": 0.0,
                "forecast_time": None,
                "weather_source": "indoor",
            }
            weather_available = True
            weather_error = None
        elif weather_error is not None:
            pass
        elif kickoff is None:
            weather_error = "missing kickoff time"
        elif coordinates is None:
            weather_error = "missing venue coordinates"
        else:
            try:
                forecast = weather.forecast(
                    kickoff=kickoff,
                    latitude=coordinates[0],
                    longitude=coordinates[1],
                )
                weather_available = True
            except DataContractError as exc:
                weather_error = str(exc)
        if weather_error:
            weather_errors.append(f"{game['game_id']}: {weather_error}")

        home_personnel_available = bool(
            home_personnel.get("depth_source_available")
            or home_personnel.get("roster_source_available")
        )
        away_personnel_available = bool(
            away_personnel.get("depth_source_available")
            or away_personnel.get("roster_source_available")
        )
        personnel_available = (
            injury_feed_available
            and home_personnel_available
            and away_personnel_available
        )

        injury_feed_status = str(
            injury_feed_state.get("status") or "UNKNOWN"
        ).upper()
        home_injury_status = str(
            home_injury.get("injury_freshness_status")
            or injury_feed_status
        ).upper()
        away_injury_status = str(
            away_injury.get("injury_freshness_status")
            or injury_feed_status
        ).upper()
        home_personnel_status = str(
            home_personnel.get("personnel_freshness_status") or "UNKNOWN"
        ).upper()
        away_personnel_status = str(
            away_personnel.get("personnel_freshness_status") or "UNKNOWN"
        ).upper()

        personnel_fresh = (
            injury_feed_status == "FRESH"
            and home_injury_status == "FRESH"
            and away_injury_status == "FRESH"
            and home_personnel_status == "FRESH"
            and away_personnel_status == "FRESH"
        )
        freshness_reasons: list[str] = []
        if injury_feed_status != "FRESH":
            freshness_reasons.append(
                "injury feed "
                + injury_feed_status.lower()
                + ": "
                + str(injury_feed_state.get("reason") or "unverified")
            )
        if home_injury_status != "FRESH":
            freshness_reasons.append(
                f"{home_team} injury state {home_injury_status.lower()}"
            )
        if away_injury_status != "FRESH":
            freshness_reasons.append(
                f"{away_team} injury state {away_injury_status.lower()}"
            )
        if home_personnel_status != "FRESH":
            freshness_reasons.append(
                f"{home_team} personnel {home_personnel_status.lower()}"
            )
        if away_personnel_status != "FRESH":
            freshness_reasons.append(
                f"{away_team} personnel {away_personnel_status.lower()}"
            )

        rest_travel_available = rest_available and travel_available
        stadium_available = roof not in {None, ""} and coordinates is not None
        weather_stadium_available = weather_available and (stadium_available or indoor)

        if personnel_fresh:
            component_counts["injuries_personnel"] += 1
        if rest_travel_available:
            component_counts["rest_travel"] += 1
        if weather_stadium_available:
            component_counts["weather_stadium"] += 1

        quality = (
            float(personnel_fresh)
            + float(rest_travel_available)
            + float(weather_stadium_available)
        ) / 3.0
        rows.append(
            {
                "game_id": str(game["game_id"]),
                "context_as_of": stamp.isoformat(),
                "context_quality": quality,
                "context_injuries_personnel_available": personnel_available,
                "context_injuries_personnel_fresh": personnel_fresh,
                "context_freshness_reason": "; ".join(freshness_reasons),
                "injury_feed_freshness_status": injury_feed_status,
                "injury_feed_latest_reported_at": injury_feed_state.get(
                    "latest_reported_at"
                ),
                "injury_feed_age_days": injury_feed_state.get("latest_age_days"),
                "home_injury_freshness_status": home_injury_status,
                "away_injury_freshness_status": away_injury_status,
                "home_depth_freshness_status": home_personnel.get(
                    "depth_freshness_status"
                ),
                "away_depth_freshness_status": away_personnel.get(
                    "depth_freshness_status"
                ),
                "home_depth_age_days": home_personnel.get("depth_age_days"),
                "away_depth_age_days": away_personnel.get("depth_age_days"),
                "home_roster_freshness_status": home_personnel.get(
                    "roster_freshness_status"
                ),
                "away_roster_freshness_status": away_personnel.get(
                    "roster_freshness_status"
                ),
                "home_roster_week_gap": home_personnel.get("roster_week_gap"),
                "away_roster_week_gap": away_personnel.get("roster_week_gap"),
                "home_personnel_freshness_status": home_personnel_status,
                "away_personnel_freshness_status": away_personnel_status,
                "context_rest_travel_available": rest_travel_available,
                "context_weather_stadium_available": weather_stadium_available,
                "home_injury_count": int(home_injury.get("injury_count", 0) or 0),
                "away_injury_count": int(away_injury.get("injury_count", 0) or 0),
                "home_injury_risk": float(
                    home_injury.get("injury_risk", 0.0) or 0.0
                ),
                "away_injury_risk": float(
                    away_injury.get("injury_risk", 0.0) or 0.0
                ),
                "home_qb_injury_risk": float(
                    home_injury.get("qb_injury_risk", 0.0) or 0.0
                ),
                "away_qb_injury_risk": float(
                    away_injury.get("qb_injury_risk", 0.0) or 0.0
                ),
                "home_starter_injury_risk": float(
                    home_personnel.get("starter_injury_risk", 0.0) or 0.0
                ),
                "away_starter_injury_risk": float(
                    away_personnel.get("starter_injury_risk", 0.0) or 0.0
                ),
                "home_ol_injury_risk": float(
                    home_personnel.get("ol_injury_risk", 0.0) or 0.0
                ),
                "away_ol_injury_risk": float(
                    away_personnel.get("ol_injury_risk", 0.0) or 0.0
                ),
                "home_skill_injury_risk": float(
                    home_personnel.get("skill_injury_risk", 0.0) or 0.0
                ),
                "away_skill_injury_risk": float(
                    away_personnel.get("skill_injury_risk", 0.0) or 0.0
                ),
                "home_defense_injury_risk": float(
                    home_personnel.get("defense_injury_risk", 0.0) or 0.0
                ),
                "away_defense_injury_risk": float(
                    away_personnel.get("defense_injury_risk", 0.0) or 0.0
                ),
                "home_active_qb_count": int(
                    home_personnel.get("active_qb_count", 0) or 0
                ),
                "away_active_qb_count": int(
                    away_personnel.get("active_qb_count", 0) or 0
                ),
                "home_rest_days": home_rest,
                "away_rest_days": away_rest,
                "home_rest_advantage_days": rest_diff,
                "away_travel_miles": travel["away_travel_miles"],
                "home_travel_miles": travel["home_travel_miles"],
                "away_timezone_shift_hours": travel["timezone_shift_hours"],
                "neutral_site": neutral,
                "stadium": stadium,
                "roof": roof,
                "indoor": indoor,
                "temperature_f": forecast.get("temperature_f"),
                "precip_probability": forecast.get("precip_probability"),
                "wind_mph": forecast.get("wind_mph"),
                "wind_gust_mph": forecast.get("wind_gust_mph"),
                "weather_forecast_time": forecast.get("forecast_time"),
                "weather_source": forecast.get("weather_source"),
                "weather_risk": weather_risk(forecast, indoor=indoor),
                "weather_error": weather_error,
            }
        )

    context = pl.DataFrame(rows).sort("game_id") if rows else pl.DataFrame()
    if not context.is_empty() and not qb_state.is_empty():
        context = attach_expected_qb_context(context, targets, qb_state)
    qb_coverage = qb_game_coverage(context)
    games = max(1, targets.height)
    coverage = {name: count / games for name, count in component_counts.items()}
    meta = {
        "status": (
            "READY" if min(coverage.values(), default=0.0) >= 0.90 else "PARTIAL"
        ),
        "games": targets.height,
        "components": coverage,
        "coverage": sum(coverage.values()) / len(coverage) if coverage else 0.0,
        "injury_feed_available": injury_feed_available,
        "injury_feed_freshness": injury_feed_state,
        "injury_feed_fresh": (
            str(injury_feed_state.get("status") or "").upper() == "FRESH"
        ),
        "depth_feed_available": depth_feed_available,
        "roster_feed_available": roster_feed_available,
        "weather_errors": weather_errors,
        "as_of": stamp.isoformat(),
        "expected_qb_identity_coverage": qb_coverage["identity_coverage"],
        "qb_decision_ready_coverage": qb_coverage["decision_ready_coverage"],
        "expected_qb_state": qb_state_meta,
        "score_adjustment_enabled": False,
        "note": (
            "Context is post-prediction risk/confidence only. Injury/personnel "
            "sources must be FRESH to permit betting; stale/unknown state fails closed."
        ),
    }
    return context, meta
