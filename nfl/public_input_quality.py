"""Supplemental input-quality publication, bound to unchanged forecast bytes."""

from datetime import timedelta

from nfl.public_context_inputs import context_contract
from nfl.public_game_inputs import clock

SPEC = "NFL_SUPPLEMENTAL_PUBLIC_INPUT_QUALITY_V1"


def quality_report(intake, forecast, forecast_sha256):
    return {
        "spec": SPEC,
        "captured_utc": intake["capture_utc"],
        "collection_started_utc": intake["collection_started_utc"],
        "source_forecast_sha256": forecast_sha256,
        "source_forecast_generated_utc": forecast["generated_utc"],
        "source_forecast_model_capture_utc": forecast["model_capture_utc"],
        "supplemental_after_forecast": True,
        "forecast_values_changed": False,
        "games": [
            {
                k: game[k]
                for k in (
                    "game_id",
                    "home_team",
                    "away_team",
                    "kickoff_utc",
                    "structured_game_observation",
                    "verified_context_inputs",
                    "input_data_contract",
                )
            }
            for game in intake["games"]
        ],
    }


def publication_quality(report, forecast, *, forecast_sha256, at):
    """Bad supplemental data stays visibly unavailable; preserve baseline forecasts."""
    fallback = {
        "status": "INPUT_CHECKS_UNAVAILABLE",
        "reason": "NO_CURRENT_INPUT_REPORT",
        "games": [],
        "forecast_values_changed": False,
    }
    if report is None:
        return fallback
    try:
        if (
            report.get("spec") != SPEC
            or report.get("source_forecast_sha256") != forecast_sha256
            or report.get("source_forecast_generated_utc") != forecast["generated_utc"]
            or report.get("source_forecast_model_capture_utc") != forecast["model_capture_utc"]
            or report.get("supplemental_after_forecast") is not True
            or report.get("forecast_values_changed") is not False
        ):
            raise ValueError("INPUT_REPORT_FORECAST_BINDING_MISMATCH")
        started, captured, decision = map(
            clock, (report["collection_started_utc"], report["captured_utc"], at)
        )
        if not clock(forecast["generated_utc"]) <= started <= captured <= decision:
            raise ValueError("INPUT_REPORT_CAPTURE_ORDER_INVALID")
        if decision - captured > timedelta(hours=6):
            raise ValueError("INPUT_REPORT_STALE")
        by_id = {}
        for game in report["games"]:
            gid = game["game_id"]
            if not gid or gid in by_id:
                raise ValueError("DUPLICATE_INPUT_REPORT_GAME")
            by_id[gid] = game
        expected = {g["game_id"] for g in forecast["games"] if g.get("predicted_score") is not None}
        if set(by_id) != expected:
            raise ValueError("INPUT_REPORT_GAME_COVERAGE_MISMATCH")
        games = []
        for game in forecast["games"]:
            if game["game_id"] not in expected:
                continue
            source = by_id[game["game_id"]]
            if (
                source["home_team"] != game["home_team"]
                or source["away_team"] != game["away_team"]
                or clock(source["kickoff_utc"]) != clock(game["kickoff_utc"])
            ):
                raise ValueError("INPUT_REPORT_MATCHUP_MISMATCH")
            context = source["verified_context_inputs"]
            sources = [
                source["structured_game_observation"],
                context.get("availability", {}).get("source", {}),
                context.get("weather", {}).get("point_source", {}),
                context.get("weather", {}).get("forecast_source", {}),
                context.get("roof_schedule_snapshot") or {},
                *context.get("official_articles", []),
            ]
            for record in sources:
                if record.get("captured_utc") is not None:
                    if not started <= clock(record["captured_utc"]) <= captured:
                        raise ValueError("SOURCE_CAPTURE_OUTSIDE_INPUT_REPORT")
            contract = context_contract(
                source["structured_game_observation"],
                context,
                game,
                asof=at,
                players=forecast.get("player_props", []),
            )
            games.append(
                {
                    "game_id": game["game_id"],
                    "home_team": game["home_team"],
                    "away_team": game["away_team"],
                    "contract": contract,
                }
            )
        return {
            "status": "SUPPLEMENTAL_INPUT_CHECKS_AVAILABLE",
            "captured_utc": captured.isoformat(),
            "games": games,
            "forecast_values_changed": False,
        }
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        fallback["reason"] = str(error)[:150]
        return fallback
