"""Time-bounded NFL weekly injury-report evidence linked by exact GSIS ID.

A provider row may have been updated after an individual team's game even when
its league-wide week and model capture time look valid. Require BOTH the web
research capture and each game kickoff as time bounds. This is never proof of
a starter, final inactive, or economically valuable point adjustment.
"""
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, UTC

VERSION = "NFL_DATED_GSIS_INJURY_REPORT_RESEARCH_V2"
ALLOWED = {"out", "doubtful", "questionable"}


def _utc(value):
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as err:
            raise ValueError("INVALID_DATE_MODIFIED") from err
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("DATE_MODIFIED_TIMEZONE_REQUIRED")
    return value.astimezone(UTC)


def add_injury_reports(review, rows, *, season, week, captured):
    """Join report statuses by GSIS/team/week with an individual kickoff cutoff.

    A provider date_modified timestamp is NOT the first published time of an
    official PDF or a final team/game inactive declaration. Missing kickoff or
    a postkickoff report capture always fails closed.
    """
    at = _utc(captured)
    if review.get("status") != "SOURCE_LINKED_PLAYER_RISKS_RESEARCH_ONLY":
        raise ValueError("ID_LINKED_RESEARCH_REQUIRED")
    if not isinstance(rows, list) or not isinstance(season, int) or not isinstance(week, int):
        raise ValueError("INVALID_INJURY_REPORT_INPUT")

    entries = defaultdict(list)
    stats = {
        "not_matching_week": 0, "invalid_timestamp": 0, "no_game_status": 0,
        "after_game_kickoff": 0, "missing_game_kickoff": 0,
        "postkickoff_capture": 0,
    }
    for row in rows:
        if not isinstance(row, dict):
            continue
        if (
            str(row.get("season")) != str(season)
            or str(row.get("week")) != str(week)
            or str(row.get("game_type") or row.get("season_type") or "").upper() != "REG"
        ):
            stats["not_matching_week"] += 1
            continue
        key = (row.get("team"), row.get("gsis_id"))
        if not all(isinstance(v, str) and v for v in key):
            continue
        try:
            modified = _utc(row.get("date_modified"))
        except ValueError:
            stats["invalid_timestamp"] += 1
            continue
        if modified > at:
            stats["invalid_timestamp"] += 1
            continue
        status = str(row.get("report_status") or "").strip().lower()
        if status not in ALLOWED:
            stats["no_game_status"] += 1
            continue
        entries[key].append((modified, status))

    report = deepcopy(review)
    count = 0
    for game in report.get("games", []):
        try:
            kickoff = _utc(game.get("kickoff_utc"))
        except ValueError:
            kickoff = None
        for team, group in game.get("teams", {}).items():
            for player in group.get("players", []):
                key = (team, player.get("player_id"))
                player["nflverse_reported_status"] = None
                player["nflverse_report_modified_utc"] = None
                player["source_proves_final_inactive"] = False
                player["source_proves_starting_qb"] = False
                matches = entries.get(key, [])
                if kickoff is None:
                    stats["missing_game_kickoff"] += bool(matches)
                    player["nflverse_report_evidence"] = "MISSING_KICKOFF_FAIL_CLOSED"
                elif at >= kickoff:
                    stats["postkickoff_capture"] += bool(matches)
                    player["nflverse_report_evidence"] = "POSTKICKOFF_CAPTURE_NOT_PREGAME"
                else:
                    eligible = [(t, s) for t, s in matches if t < kickoff]
                    stats["after_game_kickoff"] += len(matches) - len(eligible)
                    if not eligible:
                        player["nflverse_report_evidence"] = (
                            "POSTKICKOFF_SOURCE_ROW_REJECTED" if matches
                            else "NOT_LISTED_DOES_NOT_PROVE_ACTIVE"
                        )
                        continue
                    latest_time = max(t for t, _ in eligible)
                    latest_statuses = {s for t, s in eligible if t == latest_time}
                    if len(latest_statuses) != 1:
                        player["nflverse_report_evidence"] = "AMBIGUOUS_DATE_STATUS"
                    else:
                        player["nflverse_report_evidence"] = "SOURCE_DATED_REPORTED_STATUS"
                        player["nflverse_reported_status"] = next(iter(latest_statuses))
                        player["nflverse_report_modified_utc"] = latest_time.isoformat()
                        count += 1

    report["nflverse_injury_source"] = {
        "version": VERSION, "source": "nflreadpy.load_injuries",
        "season": season, "week": week, "captured_utc": at.isoformat(),
        "matched_player_statuses": count, "source_rejections": stats,
        "date_modified_not_original_publication_proof": True,
        "each_game_kickoff_cutoff_required": True,
        "final_inactives_verified": False, "starting_qbs_verified": False,
        "model_adjusted": False,
    }
    return report
