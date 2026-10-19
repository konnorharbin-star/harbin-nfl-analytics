"""Research-only prospective personnel capture tied to an existing NFL forecast.

This is NOT a quarterback-starter source or a forecasting change. It documents
what dated personnel evidence the independent forecast actually had before
kickoff, including failures and explicit unknowns, for future paired research.
"""
from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path

from nfl.coupled_forward_grading import clock

SPEC = "NFL_PERSONNEL_ASOF_WATCH_V1"
BUNDLE = "NFL_INDEPENDENT_FORECAST_PUBLICATION_V1"
CONTEXT = "NFL_PUBLIC_WEB_CONTEXT_RESEARCH_V1"


def digest(blob):
    return sha256(blob).hexdigest()


def watch(bundle, research, *, captured, bundle_sha, research_sha, run_id):
    """Reject stale/postkickoff context, preserve exact IDs, never infer actives."""
    if captured.tzinfo is None:
        raise ValueError("UNZONED_CAPTURE")
    now = captured.astimezone(UTC)
    if bundle.get("spec") != BUNDLE or research.get("version") != CONTEXT:
        raise ValueError("INVALID_OR_MISMATCHED_FAMILY")
    if not all(isinstance(v, str) and len(v) == 64
               and all(c in "0123456789abcdef" for c in v)
               for v in (bundle_sha, research_sha)):
        raise ValueError("INVALID_SOURCE_SHA256")
    btime, rtime = clock(bundle["model_capture_utc"]), clock(research["captured_utc"])
    if not btime <= rtime <= now or clock(bundle["generated_utc"]) > now:
        raise ValueError("MODEL_OR_RESEARCH_CAPTURE_CHRONOLOGY_INVALID")
    if bundle.get("sportsbook_prices_used") is not False or bundle.get("qualified_bets") != 0:
        raise ValueError("NOT_AN_INDEPENDENT_RESEARCH_FORECAST")
    games = {}
    for game in research.get("games", []):
        gid = game.get("game_id")
        if not gid or gid in games:
            raise ValueError("DUPLICATE_CONTEXT_GAME")
        games[gid] = game
    reviews = {}
    review = research.get("player_evidence_review") or {}
    for item in review.get("games", []):
        gid = item.get("game_id")
        if not gid or gid in reviews:
            raise ValueError("DUPLICATE_PLAYER_CONTEXT_GAME")
        reviews[gid] = item
    all_props = {}
    for player in bundle.get("player_props", []):
        if player.get("is_unallocated_bucket"):
            continue
        key = (player.get("game_id"), player.get("team"), player.get("player_id"))
        if not all(key) or key in all_props:
            raise ValueError("DUPLICATE_OR_UNKNOWN_MODEL_PLAYER")
        all_props[key] = player
    result = []
    for game in bundle.get("games", []):
        if game.get("predicted_score") is None:
            continue
        gid = game["game_id"]
        kickoff = clock(game["kickoff_utc"])
        if now >= kickoff:
            continue  # No postgame reinterpretation of pregame evidence.
        context = games.get(gid)
        if context is None:
            # Unavailable is unknown; still preserve the forecast and source failure.
            context = {}
        if context:
            if (clock(context["kickoff_utc"]) != kickoff
                    or context.get("away_team") != game["away_team"]
                    or context.get("home_team") != game["home_team"]):
                raise ValueError("CONTEXT_EVENT_IDENTITY_MISMATCH")
        supplemental = reviews.get(gid)
        if supplemental:
            if clock(supplemental["kickoff_utc"]) != kickoff:
                raise ValueError("PLAYER_REVIEW_KICKOFF_MISMATCH")
            if not supplemental.get("context_is_supplemental"):
                raise ValueError("REVIEW_MUST_BE_SUPPLEMENTAL")
        teams = {}
        for team in (game["away_team"], game["home_team"]):
            group = ((supplemental or {}).get("teams") or {}).get(team) or {}
            personnel = []
            qb_proxies = []
            seen = set()
            for row in group.get("players", []):
                pid = row.get("player_id")
                key = (gid, team, pid)
                if not isinstance(pid, str) or pid in seen:
                    raise ValueError("DUPLICATE_OR_UNKNOWN_REVIEW_PLAYER")
                seen.add(pid)
                model_player = all_props.get(key)
                if model_player is None:
                    raise ValueError("PERSONNEL_ID_NOT_IN_CURRENT_FORECAST")
                injury = row.get("nflverse_reported_status")
                modified = row.get("nflverse_report_modified_utc")
                if injury is not None:
                    if not modified or clock(modified) > rtime or clock(modified) >= kickoff:
                        raise ValueError("INJURY_ROW_NOT_ASOF_GAME")
                historical = row.get("historical_qb_attempts_proxy_mean")
                item = {
                    "gsis_id": pid, "team": team,
                    "model_player_name": model_player.get("player_name"),
                    "historical_passing_attempts_proxy_mean": historical,
                    "availability_evidence_status": row.get("availability_evidence_status"),
                    "nflverse_report_evidence": row.get("nflverse_report_evidence"),
                    "nflverse_reported_status": injury,
                    "nflverse_report_modified_utc": modified,
                    "es pn_reported_status": row.get("espn_reported_status"),
                    "official_starting_qb_verified": False,
                    "official_game_day_active_verified": False,
                    "point_adjustment": None,
                }
                # Drop the ESPN typo-safe key below to preserve documented schema.
                item["espn_reported_status"] = item.pop("es pn_reported_status")
                personnel.append(item)
                if historical is not None:
                    if isinstance(historical, bool) or not isinstance(historical, (int, float)):
                        raise ValueError("INVALID_HISTORICAL_PASS_VOLUME")
                    qb_proxies.append({
                        "player_id": pid,
                        "name": model_player.get("player_name"),
                        "historical_attempts_proxy": float(historical),
                        "not_verified_starter": True,
                    })
            qb_proxies.sort(key=lambda x: (-x["historical_attempts_proxy"], x["player_id"]))
            teams[team] = {
                "historical_passing_volume_proxies": qb_proxies,
                "starting_qb_id_confirmed_by_first_party": None,
                "starting_qb_verified": False,
                "game_day_inactives_verified": False,
                "players_with_injury_evidence": sum(
                    item["nflverse_reported_status"] is not None for item in personnel
                ),
                "model_player_evidence": personnel,
                "missing_review_means_unknown_not_healthy": not bool(group),
            }
        result.append({
            "spec": SPEC,
            "game_id": gid, "season": game["season"], "week": game["week"],
            "away_team": game["away_team"], "home_team": game["home_team"],
            "kickoff_utc": kickoff.isoformat(),
            "observed_utc": now.isoformat(),
            "forecast_sha256": game["forecast_sha256"],
            "model_capture_utc": game["model_capture_utc"],
            "source_bundle_sha256": bundle_sha,
            "source_web_context_sha256": research_sha,
            "web_context_observed_utc": rtime.isoformat(),
            "web_context_game_observed": bool(context),
            "player_context_game_observed": bool(supplemental),
            "source_run_id": str(run_id),
            "our_frozen_at_capture_projection": {
                "predicted_score": game["predicted_score"],
                "home_spread": game["fair_home_spread"],
                "total": game["predicted_total"],
            },
            "teams": teams,
            "confirmed_qb_effect_applied": False,
            "injury_points_adjustment_applied": False,
            "model_parameters_changed": False,
            "validated_accuracy_gain": False,
            "valid_bet_identified": False,
            "status": "PREGAME_PERSONNEL_RESEARCH_UNVERIFIED",
        })
    return {
        "spec": SPEC,
        "asof_utc": now.isoformat(),
        "original_bundle_sha256": bundle_sha,
        "supplemental_context_sha256": research_sha,
        "games": result,
        "eligible_pregame_games": len(result),
        "source_context_games_missing": sum(not g["web_context_game_observed"] for g in result),
        "starting_qbs_confirmed": 0,
        "game_day_inactives_confirmed": 0,
        "betting_authorized": False,
        "model_predictions_changed": False,
        "research_only": True,
    }


def read_journal(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append(report, path):
    """Immutable per-game original source observation, no rewriting or fill-in."""
    path = Path(path)
    before = read_journal(path)
    seen = set()
    for row in before:
        key = (row["game_id"], row["source_web_context_sha256"])
        if key in seen:
            raise ValueError("DUPLICATE_PREEXISTING_CONTEXT_RECEIPT")
        seen.add(key)
    added = []
    for item in report["games"]:
        key = (item["game_id"], item["source_web_context_sha256"])
        if key not in seen:
            added.append(item)
            seen.add(key)
    if added:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as dest:
            for row in added:
                dest.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    return {"new_observations": len(added), "repeat_observations": len(report["games"])-len(added),
            "total_observations": len(before)+len(added), "betting_authorized": False}
