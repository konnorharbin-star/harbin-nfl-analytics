"""Prospective ledger and grading for the frozen 2026 QB-total shadow.

Stage 15 selected one fixed total specification from 2022-2025 development evidence:
QB EPA + CPOE, ridge alpha 0.1, with 75 prior dropbacks. This module never rebuilds
historical predictions. It persists the first eligible pre-kickoff 2026 snapshot and
grades only those rows after final scores are available.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from .contracts import require_columns
from .qb_state import QB_PRIOR_DROPBACKS
from .qb_validated import (
    QB_TOTAL_SHADOW_ENABLED,
    VALIDATED_QB_TOTAL_ALPHA,
    VALIDATED_QB_TOTAL_FEATURE_SET,
)
from .shadow_gate import TargetShadowEvidence, paired_bootstrap_evidence

NFL_SCHEDULE_TIMEZONE = ZoneInfo("America/New_York")
QB_TOTAL_FORWARD_SEASON = 2026
QB_TOTAL_LEDGER_VERSION = 1
QB_TOTAL_SPEC_VERSION = "stage15_fixed_quality_v1"
QB_TOTAL_TRAINING_SEASONS = (2022, 2023, 2024, 2025)
QB_TOTAL_FORWARD_FIELDS = (
    "ledger_version",
    "spec_version",
    "captured_at",
    "kickoff",
    "season",
    "week",
    "game_id",
    "home_team",
    "away_team",
    "baseline_total",
    "qb_total_shadow_total",
    "qb_total_shadow_correction",
    "qb_total_feature_set",
    "qb_total_ridge_alpha",
    "qb_prior_dropbacks",
    "training_seasons",
    "qb_total_release_state",
)


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


def _kickoff(row: dict[str, object]) -> datetime | None:
    day = row.get("gameday")
    time = row.get("gametime")
    if day is None or time in {None, ""}:
        return None
    try:
        local = datetime.fromisoformat(f"{day}T{time}")
    except ValueError:
        return None
    if local.tzinfo is None:
        local = local.replace(tzinfo=NFL_SCHEDULE_TIMEZONE)
    return local.astimezone(UTC)


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _training_signature(seasons: tuple[int, ...]) -> str:
    return ";".join(str(value) for value in seasons)


def _spec_signature(row: dict[str, object]) -> tuple[str, str]:
    return (str(row.get("game_id", "")), str(row.get("spec_version", "")))


def load_qb_total_forward_predictions(
    path: str | Path = "history/qb_total_shadow_predictions_v1.csv",
) -> pl.DataFrame:
    source = Path(path)
    if not source.exists() or not source.stat().st_size:
        return pl.DataFrame()
    return pl.read_csv(source, try_parse_dates=False)


def append_qb_total_forward_predictions(
    projection: pl.DataFrame,
    targets: pl.DataFrame,
    *,
    training_seasons: tuple[int, ...] = QB_TOTAL_TRAINING_SEASONS,
    qb_prior_dropbacks: float = QB_PRIOR_DROPBACKS,
    captured_at: datetime | None = None,
    path: str | Path = "history/qb_total_shadow_predictions_v1.csv",
) -> dict[str, object]:
    """Append the first valid pre-kickoff frozen QB-total prediction per 2026 game."""

    require_columns(
        projection,
        {
            "season",
            "week",
            "game_id",
            "home_team",
            "away_team",
            "baseline_total",
            "qb_total_shadow_total",
            "qb_total_shadow_correction",
            "qb_total_release_state",
        },
        "current_qb_total_projection",
    )
    require_columns(targets, {"game_id", "gameday", "gametime"}, "current_qb_targets")
    if not QB_TOTAL_SHADOW_ENABLED:
        return {"status": "DISABLED", "appended_rows": 0, "total_rows": 0}

    now = captured_at or datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    now = now.astimezone(UTC)

    frozen_training = tuple(int(value) for value in training_seasons)
    expected_training = QB_TOTAL_TRAINING_SEASONS
    source = Path(path)
    source.parent.mkdir(parents=True, exist_ok=True)
    old_rows: list[dict[str, object]] = []
    if source.exists() and source.stat().st_size:
        with source.open(newline="") as handle:
            old_rows = list(csv.DictReader(handle))
    known = {_spec_signature(row) for row in old_rows}
    target_map = {str(row["game_id"]): row for row in targets.iter_rows(named=True)}

    appended: list[dict[str, object]] = []
    skipped_duplicate = 0
    skipped_post_kickoff = 0
    skipped_incomplete = 0
    skipped_wrong_season = 0
    skipped_wrong_training = 0
    for row in projection.iter_rows(named=True):
        if int(row["season"]) != QB_TOTAL_FORWARD_SEASON:
            skipped_wrong_season += 1
            continue
        if frozen_training != expected_training:
            skipped_wrong_training += 1
            continue
        game_id = str(row["game_id"])
        target = target_map.get(game_id)
        kickoff = _kickoff(target) if target is not None else None
        if kickoff is None:
            skipped_incomplete += 1
            continue
        if now >= kickoff:
            skipped_post_kickoff += 1
            continue
        if str(row.get("qb_total_release_state", "")).upper() != "SHADOW":
            skipped_incomplete += 1
            continue
        if not all(
            _finite(row.get(column))
            for column in (
                "baseline_total",
                "qb_total_shadow_total",
                "qb_total_shadow_correction",
            )
        ):
            skipped_incomplete += 1
            continue

        ledger_row: dict[str, object] = {
            "ledger_version": QB_TOTAL_LEDGER_VERSION,
            "spec_version": QB_TOTAL_SPEC_VERSION,
            "captured_at": now.isoformat(),
            "kickoff": kickoff.isoformat(),
            "season": int(row["season"]),
            "week": int(row["week"]),
            "game_id": game_id,
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "baseline_total": float(row["baseline_total"]),
            "qb_total_shadow_total": float(row["qb_total_shadow_total"]),
            "qb_total_shadow_correction": float(row["qb_total_shadow_correction"]),
            "qb_total_feature_set": VALIDATED_QB_TOTAL_FEATURE_SET,
            "qb_total_ridge_alpha": VALIDATED_QB_TOTAL_ALPHA,
            "qb_prior_dropbacks": float(qb_prior_dropbacks),
            "training_seasons": _training_signature(frozen_training),
            "qb_total_release_state": "SHADOW",
        }
        signature = _spec_signature(ledger_row)
        if signature in known:
            skipped_duplicate += 1
            continue
        known.add(signature)
        appended.append(ledger_row)

    combined = [*old_rows, *appended]
    if appended or not source.exists():
        with source.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(QB_TOTAL_FORWARD_FIELDS))
            writer.writeheader()
            writer.writerows(combined)
    return {
        "status": "READY",
        "path": str(source),
        "candidate_rows": projection.height,
        "appended_rows": len(appended),
        "total_rows": len(combined),
        "skipped_duplicate": skipped_duplicate,
        "skipped_post_kickoff": skipped_post_kickoff,
        "skipped_incomplete": skipped_incomplete,
        "skipped_wrong_season": skipped_wrong_season,
        "skipped_wrong_training": skipped_wrong_training,
        "spec_version": QB_TOTAL_SPEC_VERSION,
    }


def _empty_forward_summary(status: str, *, ledger_rows: int = 0) -> dict[str, object]:
    return {
        "status": status,
        "ledger_rows": ledger_rows,
        "graded_games": 0,
        "promotion_eligible": False,
        "spec_version": QB_TOTAL_SPEC_VERSION,
        "meaning": (
            "Prospective 2026 QB-total evidence only. Reconstructed historical predictions "
            "and non-frozen specifications are excluded."
        ),
    }


def grade_qb_total_forward_predictions(
    schedules: pl.DataFrame,
    predictions: pl.DataFrame,
    *,
    minimum_games: int = 128,
    bootstrap_iterations: int = 5000,
    confidence: float = 0.95,
) -> tuple[pl.DataFrame, dict[str, object]]:
    """Grade persisted frozen-spec pre-kickoff QB-total predictions."""

    if predictions.is_empty():
        return pl.DataFrame(), _empty_forward_summary("NO_PREDICTIONS")
    require_columns(
        predictions,
        {
            "spec_version",
            "captured_at",
            "kickoff",
            "season",
            "week",
            "game_id",
            "baseline_total",
            "qb_total_shadow_total",
            "qb_total_feature_set",
            "qb_total_ridge_alpha",
            "qb_prior_dropbacks",
            "training_seasons",
            "qb_total_release_state",
        },
        "qb_total_forward_predictions",
    )
    require_columns(
        schedules,
        {"season", "week", "game_id", "home_score", "away_score"},
        "schedules",
    )

    expected_training = _training_signature(QB_TOTAL_TRAINING_SEASONS)
    valid_rows: list[dict[str, object]] = []
    for row in predictions.iter_rows(named=True):
        captured = _parse_utc(row.get("captured_at"))
        kickoff = _parse_utc(row.get("kickoff"))
        if captured is None or kickoff is None or captured >= kickoff:
            continue
        if int(row["season"]) != QB_TOTAL_FORWARD_SEASON:
            continue
        if str(row["spec_version"]) != QB_TOTAL_SPEC_VERSION:
            continue
        if str(row["qb_total_feature_set"]) != VALIDATED_QB_TOTAL_FEATURE_SET:
            continue
        if float(row["qb_total_ridge_alpha"]) != VALIDATED_QB_TOTAL_ALPHA:
            continue
        if float(row["qb_prior_dropbacks"]) != float(QB_PRIOR_DROPBACKS):
            continue
        if str(row["training_seasons"]) != expected_training:
            continue
        if str(row["qb_total_release_state"]).upper() != "SHADOW":
            continue
        valid_rows.append(row)
    if not valid_rows:
        return pl.DataFrame(), _empty_forward_summary(
            "NO_VALID_PREGAME_PREDICTIONS", ledger_rows=predictions.height
        )

    first_by_game: dict[str, dict[str, object]] = {}
    for row in sorted(valid_rows, key=lambda value: str(value["captured_at"])):
        first_by_game.setdefault(str(row["game_id"]), row)
    first = pl.DataFrame(list(first_by_game.values()))
    finals = schedules.filter(
        pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null()
    ).select(["season", "week", "game_id", "home_score", "away_score"])
    graded = first.join(finals, on=["season", "week", "game_id"], how="inner")
    if graded.is_empty():
        return graded, _empty_forward_summary(
            "NO_GRADED_GAMES", ledger_rows=predictions.height
        )

    graded = graded.with_columns(
        (pl.col("home_score").cast(pl.Float64) + pl.col("away_score").cast(pl.Float64)).alias(
            "actual_total"
        )
    ).with_columns(
        (pl.col("baseline_total").cast(pl.Float64) - pl.col("actual_total"))
        .abs()
        .alias("baseline_abs_error"),
        (pl.col("qb_total_shadow_total").cast(pl.Float64) - pl.col("actual_total"))
        .abs()
        .alias("shadow_abs_error"),
    )

    actual = np.asarray(graded.get_column("actual_total"), dtype=float)
    baseline = np.asarray(graded.get_column("baseline_total"), dtype=float)
    shadow = np.asarray(graded.get_column("qb_total_shadow_total"), dtype=float)
    evidence: TargetShadowEvidence = paired_bootstrap_evidence(
        actual,
        baseline,
        shadow,
        iterations=bootstrap_iterations,
        confidence=confidence,
        minimum_games=minimum_games,
        seed=20261002,
    )
    baseline_error = baseline - actual
    shadow_error = shadow - actual
    baseline_mae = float(np.mean(np.abs(baseline_error)))
    shadow_mae = float(np.mean(np.abs(shadow_error)))
    baseline_rmse = float(np.sqrt(np.mean(np.square(baseline_error))))
    shadow_rmse = float(np.sqrt(np.mean(np.square(shadow_error))))
    summary = {
        "status": evidence.status,
        "ledger_rows": predictions.height,
        "valid_pregame_rows": len(valid_rows),
        "graded_games": graded.height,
        "minimum_games": minimum_games,
        "baseline_total_mae": baseline_mae,
        "qb_total_shadow_mae": shadow_mae,
        "total_mae_improvement": baseline_mae - shadow_mae,
        "baseline_total_rmse": baseline_rmse,
        "qb_total_shadow_rmse": shadow_rmse,
        "total_rmse_improvement": baseline_rmse - shadow_rmse,
        "evidence": evidence.to_dict(),
        "promotion_eligible": evidence.status == "PROMOTION_EVIDENCE",
        "spec_version": QB_TOTAL_SPEC_VERSION,
        "feature_set": VALIDATED_QB_TOTAL_FEATURE_SET,
        "ridge_alpha": VALIDATED_QB_TOTAL_ALPHA,
        "qb_prior_dropbacks": QB_PRIOR_DROPBACKS,
        "training_seasons": list(QB_TOTAL_TRAINING_SEASONS),
        "canonical_score_adjustment_enabled": False,
        "meaning": (
            "Prospective first-snapshot pre-kickoff evidence only. The frozen Stage 15 QB-total "
            "specification is not retuned on 2026 outcomes and does not alter canonical totals."
        ),
    }
    return graded.sort(["season", "week", "game_id"]), summary


def write_qb_total_forward_report(
    schedules: pl.DataFrame,
    *,
    ledger_path: str | Path = "history/qb_total_shadow_predictions_v1.csv",
    report_path: str | Path = "reports/qb_total_forward.json",
    graded_path: str | Path = "reports/qb_total_forward_graded.csv",
    docs_path: str | Path = "docs/qb_total_forward.json",
    minimum_games: int = 128,
) -> dict[str, object]:
    """Grade the prospective ledger and persist machine-readable forward evidence."""

    predictions = load_qb_total_forward_predictions(ledger_path)
    graded, summary = grade_qb_total_forward_predictions(
        schedules,
        predictions,
        minimum_games=minimum_games,
    )
    report_target = Path(report_path)
    docs_target = Path(docs_path)
    report_target.parent.mkdir(parents=True, exist_ok=True)
    docs_target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(summary, indent=2, sort_keys=True, default=str)
    report_target.write_text(payload)
    docs_target.write_text(payload)
    if not graded.is_empty():
        graded_target = Path(graded_path)
        graded_target.parent.mkdir(parents=True, exist_ok=True)
        graded.write_csv(graded_target)
    return summary
