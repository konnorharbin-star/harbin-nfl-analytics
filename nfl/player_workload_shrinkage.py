"""Past-only player workload shrinkage; preserves team production via routing.

No activity/starting-lineup claim and no prices. The old joint engine remains
unchanged; a new predictive family supplies revised reception/carry weights.
"""

import hashlib
import json
import math
from collections import defaultdict
from copy import deepcopy

SPEC = "NFL_PLAYER_WORKLOAD_SHRINKAGE_V2"


def history_index(rows):
    histories = defaultdict(list)
    seen = set()
    for r in rows:
        if r.get("season_type") != "REG":
            continue
        if not r.get("player_id") or not r.get("team"):
            raise ValueError("MISSING_PLAYER_WEEK_IDENTITY")
        key = (r["season"], r["team"], str(r["player_id"]), r["week"])
        if key in seen:
            raise ValueError("DUPLICATE_PLAYER_WEEK")
        seen.add(key)
        histories[key[:3]].append(r)
    for records in histories.values():
        records.sort(key=lambda r: r["week"])
    return histories


def prior_mean(histories, season, week, team, pid, metric, window):
    records = [
        r
        for r in histories.get((season, team, pid), [])
        if r["week"] < week and r.get(metric) is not None
    ]
    records = records[-window:]
    if len(records) < 3 or week - records[-1]["week"] > 2:
        return None
    values = [float(r[metric]) for r in records]
    if not all(math.isfinite(v) and v >= 0 for v in values):
        return None
    return sum(values) / len(values)


def reweight(roles, distribution, histories, *, season, week, settings):
    revised = [dict(r) for r in roles]
    audit = []
    for family, stat, key in (
        ("receiving", "receptions", "target_share"),
        ("rushing", "carries", "carry_share"),
    ):
        window, blend = settings[family]["window"], settings[family]["blend"]
        if window not in (3, 5, 8) or blend not in (0.0, 0.5, 1.0):
            raise ValueError("INVALID_FROZEN_WORKLOAD_SETTING")
        for team in sorted({r["team"] for r in revised}):
            group = [r for r in revised if r["team"] == team]
            values = []
            observed_any = False
            for r in group:
                metric = f"{team}:{r['player_id']}:{stat}"
                if metric not in distribution:
                    raise ValueError("MISSING_BASELINE_PLAYER_DISTRIBUTION")
                base = float(distribution[metric]["mean"])
                own = prior_mean(histories, season, week, team, r["player_id"], stat, window)
                observed_any = observed_any or own is not None
                value = base if own is None else (1 - blend) * base + blend * own
                if not math.isfinite(value) or value < 0:
                    raise ValueError("INVALID_WORKLOAD_MEAN")
                values.append(value)
                audit.append(
                    dict(
                        team=team,
                        player_id=r["player_id"],
                        family=family,
                        prior_mean=own,
                        fallback=own is None,
                    )
                )
            other = float(distribution.get(f"{team}:OTHER:{stat}", {}).get("mean", 0.0))
            total = sum(values) + other
            if total <= 0:
                raise ValueError("NO_BASELINE_TEAM_WORKLOAD")
            for r, value in zip(group, values, strict=True):
                r[key] = value / total if blend and observed_any else r[key]
    return revised, audit


def upgrade(scan, replay, histories, *, at, settings, model_sha, source_sha):
    from nfl.coupled_drive_player_shadow import simulate_joint
    from nfl.coupled_forward_grading import clock

    if (
        scan.get("spec") != "NFL_COUPLED_DRIVE_PLAYER_SHADOW_SCAN_V1"
        or replay.get("spec") != "NFL_COUPLED_FROZEN_REPLAY_INPUTS_V1"
        or scan["capture_utc"] != replay["capture_utc"]
        or clock(scan["capture_utc"]) > at
    ):
        raise ValueError("INVALID_PARENT_FORECAST")
    source = {r["game_id"]: r for r in replay["games"]}
    if len(source) != len(replay["games"]):
        raise ValueError("DUPLICATE_PARENT_REPLAY")
    updated, inputs, audits = [], [], []
    for raw in scan["games"]:
        gid = raw["game_id"]
        if raw.get("status") != "UNVALIDATED_JOINT_DRIVE_PLAYER_RESEARCH":
            updated.append(deepcopy(raw))
            continue
        original = source[gid]
        parent = original["forecast"]
        if (
            parent != raw["frozen_football_score_input"]
            or parent["content_sha256"] != raw["forecast_sha256"]
            or not clock(parent["captured_utc"]) <= at < clock(parent["kickoff_utc"])
        ):
            raise ValueError("PARENT_GAME_IDENTITY_OR_CUTOFF")
        roles, audit = reweight(
            original["roles"],
            raw["distribution"],
            histories,
            season=parent["season"],
            week=parent["week"],
            settings=settings,
        )
        for role in roles:
            role["observed_at_utc"] = at.isoformat()
        forecast = deepcopy(parent)
        forecast.update(
            captured_utc=at.isoformat(),
            model_version=parent["model_version"] + "+" + SPEC,
            parent_forecast_sha256=parent["content_sha256"],
        )
        forecast["content_sha256"] = hashlib.sha256(
            json.dumps(
                dict(
                    parent=parent,
                    roles=roles,
                    settings=settings,
                    model_sha=model_sha,
                    source_sha=source_sha,
                    captured_utc=at.isoformat(),
                ),
                sort_keys=True,
                allow_nan=False,
            ).encode()
        ).hexdigest()
        result = simulate_joint(
            forecast, original["drives"], roles, asof=at, draws=raw["draws"], seed=raw["seed"]
        )
        provenance = deepcopy(original["provenance"])
        provenance.update(
            workload_family=SPEC,
            workload_model_sha256=model_sha,
            workload_history_sha256=source_sha,
            workload_settings=settings,
            workload_capture_utc=at.isoformat(),
            historical_lineups_verified=False,
        )
        result.update(
            frozen_football_score_input=forecast,
            input_provenance=provenance,
            predictive_family=SPEC,
            parent_forecast_sha256=parent["content_sha256"],
        )
        updated.append(result)
        inputs.append(
            dict(
                game_id=gid,
                forecast=forecast,
                drives=original["drives"],
                roles=roles,
                provenance=provenance,
            )
        )
        audits.append(dict(game_id=gid, players=audit))
    newscan = deepcopy(scan)
    newscan.update(
        games=updated,
        capture_utc=at.isoformat(),
        predictive_family=SPEC,
        workload_model_sha256=model_sha,
        parent_capture_utc=scan["capture_utc"],
    )
    newreplay = deepcopy(replay)
    newreplay.update(
        games=inputs,
        capture_utc=at.isoformat(),
        predictive_family=SPEC,
        workload_model_sha256=model_sha,
    )
    return newscan, newreplay, audits
