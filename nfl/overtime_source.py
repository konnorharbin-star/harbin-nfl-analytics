"""Past-only regulation score and possession inputs for a separate OT challenger."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import timedelta

import numpy as np
import polars as pl

from .coupled_nflverse_asof import kickoff_utc
from .data import schedule_to_team_games
from .overtime_research import Possession, append_overtime
from .ratings import FairScoreModel


def regulation_history(schedules, pbp, *, at, season):
    required = {
        "game_id",
        "play_id",
        "qtr",
        "game_seconds_remaining",
        "fixed_drive",
        "posteam",
        "posteam_score",
        "defteam_score",
        "posteam_score_post",
        "defteam_score_post",
        "total_home_score",
        "total_away_score",
        "drive_time_of_possession",
        "drive_quarter_end",
    }
    if missing := required - set(pbp.columns):
        raise ValueError(f"missing regulation source fields: {sorted(missing)}")
    if at.tzinfo is None:
        raise ValueError("capture must be timezone-aware")
    eligible = {}
    for game in schedules.to_dicts():
        if game.get("season") not in (season - 1, season) or game.get("game_type") != "REG":
            continue
        if game.get("home_score") is None or game.get("away_score") is None:
            continue
        if kickoff_utc(game) + timedelta(hours=72) <= at:
            eligible[game["game_id"]] = game
    rows = pbp.filter(pl.col("game_id").is_in(list(eligible))).select(sorted(required))
    groups = {}
    for row in rows.sort(["game_id", "play_id"]).to_dicts():
        groups.setdefault(row["game_id"], []).append(row)
    games, kernel = [], []
    audit = Counter()
    for gid, game in eligible.items():
        original = groups.get(gid, [])
        regulation = [r for r in original if r["qtr"] in (1, 2, 3, 4)]
        audit["overtime_rows_excluded"] += sum(
            r["qtr"] is not None and r["qtr"] > 4 for r in original
        )
        end = [
            r
            for r in regulation
            if r["qtr"] == 4
            and r["game_seconds_remaining"] == 0
            and r["total_home_score"] is not None
            and r["total_away_score"] is not None
        ]
        if not end:
            audit["missing_regulation_end_game"] += 1
            continue
        home, away = end[-1]["total_home_score"], end[-1]["total_away_score"]
        if home < 0 or away < 0 or int(home) != home or int(away) != away:
            audit["invalid_regulation_score_game"] += 1
            continue
        has_ot = any(r["qtr"] is not None and r["qtr"] > 4 for r in original)
        if (has_ot and home != away) or (
            not has_ot and (home != game["home_score"] or away != game["away_score"])
        ):
            audit["regulation_final_reconciliation_failure"] += 1
            continue
        audit["regulation_tied_games"] += home == away
        # The ridge target is regulation score, never full-game schedule points.
        clean_game = {
            k: game[k]
            for k in (
                "game_id",
                "season",
                "week",
                "game_type",
                "gameday",
                "gametime",
                "home_team",
                "away_team",
            )
        }
        games.append({**clean_game, "home_score": home, "away_score": away})
        drives = {}
        for row in regulation:
            if row["fixed_drive"] is not None and row["posteam"] is not None:
                drives.setdefault(row["fixed_drive"], []).append(row)
        for drive in drives.values():
            teams = {r["posteam"] for r in drive}
            try:
                if len(teams) != 1:
                    raise ValueError("mixed possession")
                if any(r["drive_quarter_end"] is None or r["drive_quarter_end"] > 4 for r in drive):
                    raise ValueError("duration includes an unverified or overtime endpoint")
                first, last = drive[0], drive[-1]
                pf = last["posteam_score_post"] - first["posteam_score"]
                pa = last["defteam_score_post"] - first["defteam_score"]
                durations = {
                    r["drive_time_of_possession"]
                    for r in drive
                    if r["drive_time_of_possession"] is not None
                }
                if len(durations) != 1:
                    raise ValueError("missing or conflicting duration")
                minute, second = next(iter(durations)).split(":")
                if not 0 <= int(second) < 60:
                    raise ValueError("invalid clock")
                duration = int(minute) * 60 + int(second)
                if pf == 0 and pa == 0:
                    event = Possession("empty", duration)
                elif pf == 3 and pa == 0:
                    event = Possession("field_goal", duration)
                elif pf in (6, 7, 8) and pa == 0:
                    event = Possession("touchdown", duration, int(pf - 6))
                elif pf == 0 and pa in (6, 7, 8):
                    event = Possession("defensive_td", duration)
                elif pf == 0 and pa == 2:
                    event = Possession("safety", duration)
                else:
                    raise ValueError("unsupported scoring sequence")
                kernel.append(event)
            except (ValueError, TypeError):
                audit["unsupported_or_missing_drive_excluded"] += 1
    audit["eligible_schedule_games"] = len(eligible)
    audit["verified_regulation_games"] = len(games)
    audit["supported_kernel_possessions"] = len(kernel)
    if len(games) < 40 or len(kernel) < 100:
        raise ValueError(f"insufficient regulation history: {dict(audit)}")
    payload = {"games": games, "kernel": [vars(e) for e in kernel]}
    source_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()
    return (
        games,
        kernel,
        {
            **dict(audit),
            "input_sha256": source_hash,
            "capture_utc": at.isoformat(),
            "maturity_hours": 72,
            "original_historical_source_vintages_verified": False,
        },
    )


def score_challenger(games, kernel, target, *, draws=2000, seed=20261014):
    """Own regulation ridge plus paired empirical residuals and explicit OT.

    Residuals are fitted-history dispersion, not held-out calibration evidence.
    Hyperparameters are research defaults; no performance claim or promotion.
    """
    if type(draws) is not int or not 1000 <= draws <= 100000:
        raise ValueError("draws must be an integer in [1000, 100000]")
    if any(kickoff_utc(g) + timedelta(hours=72) > kickoff_utc(target) for g in games):
        raise ValueError("training game is not mature before target kickoff")
    model = FairScoreModel().fit(schedule_to_team_games(pl.DataFrame(games)))
    fair = model.project(target["home_team"], target["away_team"])
    residuals = []
    for game in games:
        fitted = model.project(game["home_team"], game["away_team"])
        residuals.append(
            [game["home_score"] - fitted.home_points, game["away_score"] - fitted.away_points]
        )
    residuals = np.asarray(residuals)
    residuals -= residuals.mean(axis=0)
    rng = np.random.default_rng(seed)
    sampled = residuals[rng.integers(0, len(residuals), size=draws)]
    scores = np.rint(np.maximum(0, sampled + [fair.home_points, fair.away_points]))
    report = append_overtime(scores, kernel, score_phase="VERIFIED_REGULATION_ONLY", seed=seed)
    report.update(
        {
            "game_id": target["game_id"],
            "home_team": target["home_team"],
            "away_team": target["away_team"],
            "kickoff_utc": kickoff_utc(target).isoformat(),
            "regulation_ridge_home_points": fair.home_points,
            "regulation_ridge_away_points": fair.away_points,
            "training_games": len(games),
            "kernel_possessions": len(kernel),
            "residual_calibration": "FITTED_HISTORY_NOT_VALIDATED",
            "kernel": "POOLED_REGULATION_POSSESSION_DURATION_PROXY_NOT_OT_TACTICS",
        }
    )
    return report
