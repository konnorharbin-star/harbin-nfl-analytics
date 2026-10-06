"""Current expected-starting-QB identification and certainty controls."""

from __future__ import annotations

from datetime import UTC, datetime
from math import isfinite

import polars as pl

from .contracts import require_columns


def _as_utc(value: object) -> datetime | None:
    if value in {None, ""}:
        return None
    if isinstance(value, datetime):
        stamp = value
    else:
        try:
            stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=UTC)
    return stamp.astimezone(UTC)


def _number(value: object, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    return numeric if isfinite(numeric) else float(default)


def _proxy_map(projection: pl.DataFrame) -> dict[str, dict[str, str]]:
    required = {
        "home_team",
        "away_team",
        "home_qb_proxy_id",
        "away_qb_proxy_id",
    }
    require_columns(projection, required, "current_projection")
    output: dict[str, dict[str, str]] = {}
    for row in projection.iter_rows(named=True):
        for side in ("home", "away"):
            team = str(row[f"{side}_team"])
            qb_id = str(row.get(f"{side}_qb_proxy_id") or "")
            qb_name = str(row.get(f"{side}_qb_proxy_name") or "")
            output[team] = {
                "id": qb_id,
                "name": qb_name,
                "key": f"id:{qb_id}" if qb_id else "",
            }
    return output


def _player_rows(
    frame: pl.DataFrame,
    team: str,
    *,
    position: str = "QB",
) -> list[dict[str, object]]:
    if frame.is_empty() or "team" not in frame.columns:
        return []
    subset = frame.filter(pl.col("team") == team)
    if "position" in subset.columns:
        subset = subset.filter(pl.col("position") == position)
    return subset.to_dicts()


def _by_key(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {
        str(row.get("player_key") or ""): row
        for row in rows
        if str(row.get("player_key") or "")
    }


def _candidate_available(
    key: str,
    roster: dict[str, dict[str, object]],
    injury: dict[str, dict[str, object]],
) -> bool:
    roster_row = roster.get(key)
    if roster_row is not None:
        known = bool(roster_row.get("status_known"))
        if known and not bool(roster_row.get("active")):
            return False
    injury_row = injury.get(key)
    if injury_row is not None and _number(injury_row.get("severity")) >= 0.95:
        return False
    return True


def _depth_age_days(row: dict[str, object], as_of: datetime) -> float | None:
    captured = _as_utc(row.get("depth_captured_at"))
    if captured is None:
        return None
    return max(0.0, (as_of - captured).total_seconds() / 86400.0)


def _confidence_label(score: float) -> str:
    if score >= 0.80:
        return "HIGH"
    if score >= 0.65:
        return "MEDIUM"
    if score > 0.0:
        return "LOW"
    return "UNKNOWN"


def _select_team_qb(
    team: str,
    *,
    depth: pl.DataFrame,
    rosters: pl.DataFrame,
    injuries: pl.DataFrame,
    proxy: dict[str, str] | None,
    as_of: datetime,
) -> dict[str, object]:
    depth_rows = _player_rows(depth, team)
    roster_rows = _player_rows(rosters, team)
    injury_rows = _player_rows(injuries, team)
    roster_by_key = _by_key(roster_rows)
    injury_by_key = _by_key(injury_rows)

    ranked_depth = sorted(
        depth_rows,
        key=lambda row: (
            999 if row.get("depth_rank") is None else int(row["depth_rank"]),
            str(row.get("player_name") or ""),
        ),
    )
    rank_one = [row for row in ranked_depth if row.get("depth_rank") == 1]
    rank_one_ambiguity = len(rank_one) > 1

    selected: dict[str, object] | None = None
    source = ""
    for row in ranked_depth:
        key = str(row.get("player_key") or "")
        if not key or not _candidate_available(key, roster_by_key, injury_by_key):
            continue
        selected = row
        source = "depth_chart"
        if row.get("depth_rank") != 1:
            source = "depth_chart_replacement"
        break

    active_roster_qbs = [
        row
        for row in roster_rows
        if bool(row.get("active"))
        and (
            not bool(row.get("status_known"))
            or bool(row.get("active"))
        )
    ]

    if selected is None and len(active_roster_qbs) == 1:
        selected = active_roster_qbs[0]
        source = "unique_active_roster"

    proxy_key = str((proxy or {}).get("key") or "")
    if selected is None and proxy_key:
        proxy_roster = roster_by_key.get(proxy_key)
        if (
            proxy_roster is not None
            and _candidate_available(proxy_key, roster_by_key, injury_by_key)
        ):
            selected = proxy_roster
            source = "last_observed_proxy_fallback"

    if selected is None:
        reason = "no unique available expected starting QB can be established"
        if rank_one_ambiguity:
            reason = "multiple QB1 depth-chart candidates are present"
        return {
            "team": team,
            "expected_qb_id": None,
            "expected_qb_name": None,
            "expected_qb_player_key": None,
            "expected_qb_source": "unresolved",
            "expected_qb_depth_rank": None,
            "expected_qb_depth_age_days": None,
            "expected_qb_roster_status": None,
            "expected_qb_roster_active": None,
            "expected_qb_injury_status": None,
            "expected_qb_practice_status": None,
            "expected_qb_injury_severity": None,
            "expected_qb_confidence": 0.0,
            "expected_qb_confidence_label": "UNKNOWN",
            "expected_qb_matches_last_observed": False,
            "expected_qb_changed_from_last_observed": None,
            "expected_qb_decision_ready": False,
            "expected_qb_reason": reason,
        }

    key = str(selected.get("player_key") or "")
    roster_row = roster_by_key.get(key, {})
    injury_row = injury_by_key.get(key, {})
    depth_row = next(
        (
            row
            for row in ranked_depth
            if str(row.get("player_key") or "") == key
        ),
        {},
    )

    qb_id = str(
        selected.get("gsis_id")
        or roster_row.get("gsis_id")
        or depth_row.get("gsis_id")
        or ""
    )
    qb_name = str(
        selected.get("player_name")
        or roster_row.get("player_name")
        or depth_row.get("player_name")
        or ""
    )
    depth_rank = depth_row.get("depth_rank")
    depth_age = _depth_age_days(depth_row, as_of) if depth_row else None
    roster_known = bool(roster_row.get("status_known")) if roster_row else False
    roster_active = (
        bool(roster_row.get("active"))
        if roster_row
        else None
    )
    severity = (
        _number(injury_row.get("severity"))
        if injury_row
        else 0.0
    )

    if source == "depth_chart":
        confidence = 0.90
    elif source == "depth_chart_replacement":
        confidence = 0.75
    elif source == "unique_active_roster":
        confidence = 0.68
    else:
        confidence = 0.45

    if rank_one_ambiguity:
        confidence = min(confidence, 0.45)
    if depth_row:
        if depth_age is None:
            confidence = min(confidence, 0.80)
        elif depth_age > 7.0:
            confidence = min(confidence, 0.55)
    if roster_known and roster_active is False:
        confidence = 0.0
    if severity >= 0.95:
        confidence = 0.0
    elif severity >= 0.80:
        confidence = min(confidence, 0.35)
    elif severity >= 0.40:
        confidence = min(confidence, 0.55)
    elif severity >= 0.25:
        confidence = min(confidence, 0.70)

    matches_proxy = bool(proxy_key and key == proxy_key)
    changed = None if not proxy_key else not matches_proxy
    # The canonical score does not yet contain a validated replacement-QB adjustment.
    # A newly inferred starter is therefore identified, but not betting-ready.
    decision_ready = confidence >= 0.65 and changed is not True

    reasons: list[str] = [f"selected from {source}"]
    if changed is True:
        reasons.append("differs from last-observed QB used by current QB shadow state")
    if severity >= 0.40:
        reasons.append(
            f"QB injury uncertainty severity={severity:.2f}"
        )
    if depth_age is not None and depth_age > 7.0:
        reasons.append(f"depth snapshot is {depth_age:.1f} days old")
    if rank_one_ambiguity:
        reasons.append("multiple QB1 depth-chart candidates")
    if confidence < 0.65:
        reasons.append("starter certainty below betting threshold")

    return {
        "team": team,
        "expected_qb_id": qb_id or None,
        "expected_qb_name": qb_name or None,
        "expected_qb_player_key": key or None,
        "expected_qb_source": source,
        "expected_qb_depth_rank": depth_rank,
        "expected_qb_depth_age_days": (
            None if depth_age is None else round(depth_age, 4)
        ),
        "expected_qb_roster_status": (
            str(roster_row.get("roster_status") or "")
            if roster_row
            else None
        ),
        "expected_qb_roster_active": roster_active,
        "expected_qb_injury_status": (
            str(injury_row.get("report_status") or "")
            if injury_row
            else None
        ),
        "expected_qb_practice_status": (
            str(injury_row.get("practice_status") or "")
            if injury_row
            else None
        ),
        "expected_qb_injury_severity": round(severity, 4),
        "expected_qb_confidence": round(confidence, 4),
        "expected_qb_confidence_label": _confidence_label(confidence),
        "expected_qb_matches_last_observed": matches_proxy,
        "expected_qb_changed_from_last_observed": changed,
        "expected_qb_decision_ready": decision_ready,
        "expected_qb_reason": "; ".join(reasons),
    }


def build_expected_qb_state(
    targets: pl.DataFrame,
    projection: pl.DataFrame,
    *,
    depth: pl.DataFrame,
    rosters: pl.DataFrame,
    injuries: pl.DataFrame,
    as_of: datetime | None = None,
) -> tuple[pl.DataFrame, dict[str, object]]:
    """Build expected starter state for every team on the active slate."""

    require_columns(
        targets,
        {"game_id", "home_team", "away_team"},
        "current_targets",
    )
    reference = as_of or datetime.now(UTC)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=UTC)
    reference = reference.astimezone(UTC)
    proxies = _proxy_map(projection)

    teams = sorted(
        {
            str(value)
            for column in ("home_team", "away_team")
            for value in targets.get_column(column).to_list()
        }
    )
    rows = [
        _select_team_qb(
            team,
            depth=depth,
            rosters=rosters,
            injuries=injuries,
            proxy=proxies.get(team),
            as_of=reference,
        )
        for team in teams
    ]
    state = pl.DataFrame(rows).sort("team") if rows else pl.DataFrame()

    identified = sum(bool(row.get("expected_qb_id")) for row in rows)
    ready = sum(bool(row.get("expected_qb_decision_ready")) for row in rows)
    total = max(1, len(rows))
    return state, {
        "teams": len(rows),
        "identified_teams": identified,
        "decision_ready_teams": ready,
        "identity_coverage": identified / total,
        "decision_ready_team_coverage": ready / total,
        "as_of": reference.isoformat(),
    }


def attach_expected_qb_context(
    context: pl.DataFrame,
    targets: pl.DataFrame,
    qb_state: pl.DataFrame,
) -> pl.DataFrame:
    """Attach home/away expected starter state to each game context row."""

    if context.is_empty():
        return context
    if qb_state.is_empty():
        return context
    require_columns(context, {"game_id"}, "current_context")
    require_columns(
        targets,
        {"game_id", "home_team", "away_team"},
        "current_targets",
    )
    qb_map = {
        str(row["team"]): row
        for row in qb_state.iter_rows(named=True)
    }
    games = {
        str(row["game_id"]): row
        for row in targets.iter_rows(named=True)
    }

    output: list[dict[str, object]] = []
    fields = [
        "expected_qb_id",
        "expected_qb_name",
        "expected_qb_source",
        "expected_qb_depth_rank",
        "expected_qb_depth_age_days",
        "expected_qb_roster_status",
        "expected_qb_roster_active",
        "expected_qb_injury_status",
        "expected_qb_practice_status",
        "expected_qb_injury_severity",
        "expected_qb_confidence",
        "expected_qb_confidence_label",
        "expected_qb_matches_last_observed",
        "expected_qb_changed_from_last_observed",
        "expected_qb_decision_ready",
        "expected_qb_reason",
    ]
    for row in context.iter_rows(named=True):
        item = dict(row)
        game = games.get(str(row["game_id"]), {})
        home = qb_map.get(str(game.get("home_team") or ""), {})
        away = qb_map.get(str(game.get("away_team") or ""), {})
        for side, state in (("home", home), ("away", away)):
            for field in fields:
                item[f"{side}_{field}"] = state.get(field)
        home_ready = bool(home.get("expected_qb_decision_ready"))
        away_ready = bool(away.get("expected_qb_decision_ready"))
        item["qb_context_ready"] = home_ready and away_ready
        reasons = []
        if not home_ready:
            reasons.append(
                "home: " + str(home.get("expected_qb_reason") or "unresolved")
            )
        if not away_ready:
            reasons.append(
                "away: " + str(away.get("expected_qb_reason") or "unresolved")
            )
        item["qb_context_reason"] = "; ".join(reasons)
        output.append(item)
    return pl.DataFrame(output).sort("game_id")


def qb_game_coverage(context: pl.DataFrame) -> dict[str, float]:
    """Return identity and decision-ready coverage at the game level."""

    if context.is_empty():
        return {
            "identity_coverage": 0.0,
            "decision_ready_coverage": 0.0,
        }
    required = {
        "home_expected_qb_id",
        "away_expected_qb_id",
        "home_expected_qb_decision_ready",
        "away_expected_qb_decision_ready",
    }
    if not required.issubset(context.columns):
        return {
            "identity_coverage": 0.0,
            "decision_ready_coverage": 0.0,
        }
    games = max(1, context.height)
    identity = context.filter(
        pl.col("home_expected_qb_id").is_not_null()
        & pl.col("away_expected_qb_id").is_not_null()
    ).height
    ready = context.filter(
        pl.col("home_expected_qb_decision_ready").fill_null(False)
        & pl.col("away_expected_qb_decision_ready").fill_null(False)
    ).height
    return {
        "identity_coverage": identity / games,
        "decision_ready_coverage": ready / games,
    }


def apply_qb_certainty_veto(candidates: pl.DataFrame) -> pl.DataFrame:
    """Fail closed when expected starter identity/certainty is not decision-ready."""

    if candidates.is_empty():
        return candidates
    required = {
        "home_expected_qb_decision_ready",
        "away_expected_qb_decision_ready",
    }
    if not required.issubset(candidates.columns):
        rows = candidates.to_dicts()
        for row in rows:
            row["qb_certainty_veto"] = True
            row["qb_certainty_veto_reason"] = (
                "expected starting-QB context is unavailable"
            )
            for field in (
                "quant_signal",
                "production_signal",
                "research_signal",
                "portfolio_signal",
            ):
                if field in row:
                    row[field] = "PASS"
            for field in ("stake_units", "research_stake_units"):
                if field in row:
                    row[field] = 0.0
        return pl.DataFrame(rows)

    rows: list[dict[str, object]] = []
    for row in candidates.iter_rows(named=True):
        home_ready = bool(row.get("home_expected_qb_decision_ready"))
        away_ready = bool(row.get("away_expected_qb_decision_ready"))
        veto = not (home_ready and away_ready)
        reason = str(row.get("qb_context_reason") or "")
        if veto and not reason:
            reason = "expected starting-QB certainty is below betting threshold"
        item = dict(row)
        item["qb_certainty_veto"] = veto
        item["qb_certainty_veto_reason"] = reason if veto else ""
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
