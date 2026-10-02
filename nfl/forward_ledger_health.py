"""Integrity and capture-coverage audit for prospective NFL shadow ledgers."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

import polars as pl

from .probability_forward import (
    FROZEN_LOGISTIC_ALPHA,
    FROZEN_MARGIN_SCALE,
    FROZEN_TOTAL_SCALE,
    PROBABILITY_FORWARD_SEASON,
    PROBABILITY_SPEC_VERSION,
    PROBABILITY_TRAINING_SEASONS,
    load_probability_forward_predictions,
)
from .qb_state import QB_PRIOR_DROPBACKS
from .qb_total_forward import (
    QB_TOTAL_FORWARD_SEASON,
    QB_TOTAL_SPEC_VERSION,
    QB_TOTAL_TRAINING_SEASONS,
    load_qb_total_forward_predictions,
)
from .qb_validated import (
    VALIDATED_QB_TOTAL_ALPHA,
    VALIDATED_QB_TOTAL_FEATURE_SET,
)
from .recent_form_forward import load_recent_form_forward_predictions
from .recent_form_shadow import (
    FROZEN_BLEND_WEIGHT,
    FROZEN_FEATURE_SET,
    FROZEN_RECENT_ALPHA,
    FROZEN_RIDGE_ALPHA,
)

NFL_SCHEDULE_TIMEZONE = ZoneInfo("America/New_York")
FORWARD_SEASON = 2026


def _parse_utc(value: object) -> datetime | None:
    if value in {None, ""}:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _schedule_kickoff(row: dict[str, object]) -> datetime | None:
    day = row.get("gameday")
    time = row.get("gametime")
    if day in {None, ""} or time in {None, ""}:
        return None
    try:
        local = datetime.fromisoformat(f"{day}T{time}")
    except ValueError:
        return None
    if local.tzinfo is None:
        local = local.replace(tzinfo=NFL_SCHEDULE_TIMEZONE)
    return local.astimezone(UTC)


def _int_equal(value: object, expected: int) -> bool:
    try:
        return int(value) == int(expected)
    except (TypeError, ValueError):
        return False


def _float_equal(value: object, expected: float) -> bool:
    try:
        return abs(float(value) - float(expected)) <= 1e-12
    except (TypeError, ValueError):
        return False


def _training_signature(seasons: tuple[int, ...]) -> str:
    return ";".join(str(value) for value in seasons)


def _probability_valid(row: dict[str, object]) -> bool:
    return (
        _int_equal(row.get("season"), PROBABILITY_FORWARD_SEASON)
        and str(row.get("spec_version", "")) == PROBABILITY_SPEC_VERSION
        and _float_equal(row.get("margin_scale"), FROZEN_MARGIN_SCALE)
        and _float_equal(row.get("total_scale"), FROZEN_TOTAL_SCALE)
        and _float_equal(row.get("logistic_alpha"), FROZEN_LOGISTIC_ALPHA)
        and str(row.get("training_seasons", ""))
        == _training_signature(PROBABILITY_TRAINING_SEASONS)
        and str(row.get("release_state", "")).upper() == "SHADOW"
    )


def _qb_total_valid(row: dict[str, object]) -> bool:
    return (
        _int_equal(row.get("season"), QB_TOTAL_FORWARD_SEASON)
        and str(row.get("spec_version", "")) == QB_TOTAL_SPEC_VERSION
        and str(row.get("qb_total_feature_set", ""))
        == VALIDATED_QB_TOTAL_FEATURE_SET
        and _float_equal(
            row.get("qb_total_ridge_alpha"),
            VALIDATED_QB_TOTAL_ALPHA,
        )
        and _float_equal(row.get("qb_prior_dropbacks"), QB_PRIOR_DROPBACKS)
        and str(row.get("training_seasons", ""))
        == _training_signature(QB_TOTAL_TRAINING_SEASONS)
        and str(row.get("qb_total_release_state", "")).upper() == "SHADOW"
    )


def _recent_form_valid(row: dict[str, object]) -> bool:
    return (
        _int_equal(row.get("season"), FORWARD_SEASON)
        and _float_equal(row.get("recent_form_alpha"), FROZEN_RECENT_ALPHA)
        and _float_equal(
            row.get("recent_form_ridge_alpha"),
            FROZEN_RIDGE_ALPHA,
        )
        and _float_equal(
            row.get("recent_form_blend_weight"),
            FROZEN_BLEND_WEIGHT,
        )
        and str(row.get("recent_form_feature_set", ""))
        == ";".join(FROZEN_FEATURE_SET)
        and str(row.get("recent_form_total_release_state", "")).upper()
        == "SHADOW"
    )


def _regular_schedule(schedules: pl.DataFrame) -> pl.DataFrame:
    required = {"season", "week", "game_id", "gameday", "gametime"}
    missing = sorted(required.difference(schedules.columns))
    if missing:
        raise ValueError(
            "forward ledger health schedule missing columns: "
            + ", ".join(missing)
        )
    frame = schedules.filter(pl.col("season") == FORWARD_SEASON)
    if "game_type" in frame.columns:
        frame = frame.filter(pl.col("game_type") == "REG")
    return frame


def audit_candidate_ledger(
    schedules: pl.DataFrame,
    predictions: pl.DataFrame,
    *,
    name: str,
    validator: Callable[[dict[str, object]], bool],
) -> dict[str, object]:
    """Audit one frozen candidate from inception through every opened week."""

    if predictions.is_empty():
        return {
            "name": name,
            "status": "NO_LEDGER",
            "rows": 0,
            "valid_rows": 0,
            "invalid_timing_rows": 0,
            "invalid_spec_rows": 0,
            "duplicate_rows": 0,
            "opened_weeks": [],
            "inception_at": None,
            "eligible_games": 0,
            "captured_eligible_games": 0,
            "capture_coverage": 0.0,
            "missing_game_ids": [],
            "unexpected_game_ids": [],
            "promotion_sample_eligible": False,
        }

    valid_rows: list[dict[str, object]] = []
    invalid_timing = 0
    invalid_spec = 0
    for row in predictions.iter_rows(named=True):
        captured = _parse_utc(row.get("captured_at"))
        kickoff = _parse_utc(row.get("kickoff"))
        if captured is None or kickoff is None or captured >= kickoff:
            invalid_timing += 1
            continue
        if not validator(row):
            invalid_spec += 1
            continue
        valid_rows.append(row)

    if not valid_rows:
        return {
            "name": name,
            "status": "INVALID_LEDGER",
            "rows": predictions.height,
            "valid_rows": 0,
            "invalid_timing_rows": invalid_timing,
            "invalid_spec_rows": invalid_spec,
            "duplicate_rows": 0,
            "opened_weeks": [],
            "inception_at": None,
            "eligible_games": 0,
            "captured_eligible_games": 0,
            "capture_coverage": 0.0,
            "missing_game_ids": [],
            "unexpected_game_ids": [],
            "promotion_sample_eligible": False,
        }

    captured_times = [
        _parse_utc(row["captured_at"])
        for row in valid_rows
    ]
    inception = min(value for value in captured_times if value is not None)
    opened_weeks = sorted(
        {
            (int(row["season"]), int(row["week"]))
            for row in valid_rows
        }
    )

    counts: dict[str, int] = {}
    for row in valid_rows:
        game_id = str(row["game_id"])
        counts[game_id] = counts.get(game_id, 0) + 1
    duplicate_rows = sum(max(0, value - 1) for value in counts.values())
    captured_games = set(counts)

    schedule = _regular_schedule(schedules)
    eligible_games: set[str] = set()
    for season, week in opened_weeks:
        weekly = schedule.filter(
            (pl.col("season") == season) & (pl.col("week") == week)
        )
        for row in weekly.iter_rows(named=True):
            kickoff = _schedule_kickoff(row)
            if kickoff is not None and kickoff > inception:
                eligible_games.add(str(row["game_id"]))

    captured_eligible = captured_games.intersection(eligible_games)
    missing = sorted(eligible_games.difference(captured_games))
    unexpected = sorted(captured_games.difference(eligible_games))
    coverage = (
        len(captured_eligible) / len(eligible_games)
        if eligible_games
        else 0.0
    )
    passed = (
        invalid_timing == 0
        and invalid_spec == 0
        and duplicate_rows == 0
        and not missing
        and not unexpected
        and bool(eligible_games)
    )
    return {
        "name": name,
        "status": "PASS" if passed else "FAIL",
        "rows": predictions.height,
        "valid_rows": len(valid_rows),
        "invalid_timing_rows": invalid_timing,
        "invalid_spec_rows": invalid_spec,
        "duplicate_rows": duplicate_rows,
        "opened_weeks": [
            {"season": season, "week": week}
            for season, week in opened_weeks
        ],
        "inception_at": inception.isoformat(),
        "eligible_games": len(eligible_games),
        "captured_eligible_games": len(captured_eligible),
        "capture_coverage": coverage,
        "missing_game_ids": missing,
        "unexpected_game_ids": unexpected,
        "promotion_sample_eligible": passed,
    }


def build_forward_ledger_health(
    schedules: pl.DataFrame,
    *,
    probability_predictions: pl.DataFrame | None = None,
    qb_total_predictions: pl.DataFrame | None = None,
    recent_form_predictions: pl.DataFrame | None = None,
) -> dict[str, object]:
    """Build candidate-level capture integrity for all active frozen ledgers."""

    probability = (
        probability_predictions
        if probability_predictions is not None
        else load_probability_forward_predictions()
    )
    qb_total = (
        qb_total_predictions
        if qb_total_predictions is not None
        else load_qb_total_forward_predictions()
    )
    recent_form = (
        recent_form_predictions
        if recent_form_predictions is not None
        else load_recent_form_forward_predictions()
    )

    candidates = {
        "probability_home_win": audit_candidate_ledger(
            schedules,
            probability,
            name="probability_home_win",
            validator=_probability_valid,
        ),
        "probability_total_distribution": audit_candidate_ledger(
            schedules,
            probability,
            name="probability_total_distribution",
            validator=_probability_valid,
        ),
        "qb_total": audit_candidate_ledger(
            schedules,
            qb_total,
            name="qb_total",
            validator=_qb_total_valid,
        ),
        "recent_form_total": audit_candidate_ledger(
            schedules,
            recent_form,
            name="recent_form_total",
            validator=_recent_form_valid,
        ),
    }
    all_pass = all(value["status"] == "PASS" for value in candidates.values())
    return {
        "version": 1,
        "status": "PASS" if all_pass else "FAIL",
        "season": FORWARD_SEASON,
        "candidates": candidates,
        "promotion_requires_ledger_health": True,
        "meaning": (
            "Each frozen candidate must have exactly one valid pre-kickoff snapshot "
            "for every eligible game in every week it has opened. Games that kicked "
            "off before a candidate's first persisted snapshot are outside that "
            "candidate's prospective sample and are never backfilled."
        ),
    }


def write_forward_ledger_health(
    schedules: pl.DataFrame,
    *,
    report_path: str | Path = "reports/forward_ledger_health.json",
    docs_path: str | Path = "docs/forward_ledger_health.json",
) -> dict[str, object]:
    report = build_forward_ledger_health(schedules)
    payload = json.dumps(report, indent=2, sort_keys=True, default=str)
    for path in (report_path, docs_path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload, encoding="utf-8")
    return report
