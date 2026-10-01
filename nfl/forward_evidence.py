"""Read-only integrity audit for frozen prospective 2026 shadow ledgers.

The audit never changes a prediction or promotion state. It verifies that the two
frozen total-shadow ledgers are genuinely prospective, internally consistent, and
anchored to the same canonical baseline on overlapping games.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from .contracts import DataContractError, require_columns
from .qb_state import QB_PRIOR_DROPBACKS
from .qb_total_forward import (
    QB_TOTAL_FORWARD_SEASON,
    QB_TOTAL_LEDGER_VERSION,
    QB_TOTAL_SPEC_VERSION,
    QB_TOTAL_TRAINING_SEASONS,
)
from .qb_validated import VALIDATED_QB_TOTAL_ALPHA, VALIDATED_QB_TOTAL_FEATURE_SET
from .recent_form_forward import FORWARD_LEDGER_VERSION
from .recent_form_shadow import (
    FROZEN_BLEND_WEIGHT,
    FROZEN_FEATURE_SET,
    FROZEN_RECENT_ALPHA,
    FROZEN_RIDGE_ALPHA,
)

RECENT_FORM_FORWARD_SEASON = 2026
RECENT_FORM_SPEC_VERSION = "stage8_fixed_recent_form_v1"
FLOAT_TOLERANCE = 1e-9


@dataclass(frozen=True)
class LedgerIntegrity:
    ledger: str
    spec_version: str
    source_path: str | None
    sha256: str | None
    rows: int
    unique_games: int
    earliest_capture: str | None
    latest_capture: str | None
    duplicate_games: int
    prekickoff_violations: int
    wrong_season_rows: int
    frozen_spec_violations: int
    additive_identity_violations: int
    nonfinite_rows: int
    status: str


@dataclass(frozen=True)
class CrossLedgerIntegrity:
    overlapping_games: int
    baseline_disagreements: int
    kickoff_disagreements: int
    team_disagreements: int
    max_baseline_abs_difference: float
    status: str


@dataclass(frozen=True)
class ForwardEvidenceIntegrity:
    recent_form: LedgerIntegrity
    qb_total: LedgerIntegrity
    cross_ledger: CrossLedgerIntegrity
    overall_status: str
    promotion_eligible: bool
    canonical_score_change_enabled: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


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


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _close(left: object, right: object, *, tolerance: float = FLOAT_TOLERANCE) -> bool:
    if not _finite(left) or not _finite(right):
        return False
    return abs(float(left) - float(right)) <= tolerance


def _sha256(path: str | Path | None) -> str | None:
    if path is None:
        return None
    source = Path(path)
    if not source.exists():
        return None
    return hashlib.sha256(source.read_bytes()).hexdigest()


def _capture_range(frame: pl.DataFrame) -> tuple[str | None, str | None]:
    parsed = [
        value
        for value in (_parse_utc(row["captured_at"]) for row in frame.iter_rows(named=True))
        if value is not None
    ]
    if not parsed:
        return None, None
    return min(parsed).isoformat(), max(parsed).isoformat()


def _duplicate_games(frame: pl.DataFrame) -> int:
    counts = frame.group_by("game_id").len().filter(pl.col("len") > 1)
    return counts.height


def audit_recent_form_ledger(
    frame: pl.DataFrame,
    *,
    source_path: str | Path | None = None,
) -> LedgerIntegrity:
    """Verify the frozen recent-form total ledger without mutating it."""

    required = {
        "ledger_version",
        "captured_at",
        "kickoff",
        "season",
        "game_id",
        "baseline_total",
        "recent_form_shadow_total",
        "recent_form_total_adjustment",
        "recent_form_alpha",
        "recent_form_ridge_alpha",
        "recent_form_blend_weight",
        "recent_form_feature_set",
        "recent_form_total_release_state",
    }
    require_columns(frame, required, "recent_form_forward_ledger")
    if frame.is_empty():
        raise DataContractError("recent-form forward ledger is empty")

    prekickoff_violations = 0
    wrong_season_rows = 0
    frozen_spec_violations = 0
    additive_identity_violations = 0
    nonfinite_rows = 0
    expected_features = ";".join(FROZEN_FEATURE_SET)
    for row in frame.iter_rows(named=True):
        captured = _parse_utc(row["captured_at"])
        kickoff = _parse_utc(row["kickoff"])
        if captured is None or kickoff is None or captured >= kickoff:
            prekickoff_violations += 1
        if int(row["season"]) != RECENT_FORM_FORWARD_SEASON:
            wrong_season_rows += 1
        frozen_ok = (
            int(row["ledger_version"]) == FORWARD_LEDGER_VERSION
            and _close(row["recent_form_alpha"], FROZEN_RECENT_ALPHA)
            and _close(row["recent_form_ridge_alpha"], FROZEN_RIDGE_ALPHA)
            and _close(row["recent_form_blend_weight"], FROZEN_BLEND_WEIGHT)
            and str(row["recent_form_feature_set"]) == expected_features
            and str(row["recent_form_total_release_state"]).upper() == "SHADOW"
        )
        if not frozen_ok:
            frozen_spec_violations += 1
        numeric = (
            row["baseline_total"],
            row["recent_form_shadow_total"],
            row["recent_form_total_adjustment"],
        )
        if not all(_finite(value) for value in numeric):
            nonfinite_rows += 1
        elif not _close(
            float(row["baseline_total"]) + float(row["recent_form_total_adjustment"]),
            row["recent_form_shadow_total"],
        ):
            additive_identity_violations += 1

    duplicates = _duplicate_games(frame)
    earliest, latest = _capture_range(frame)
    violations = (
        duplicates
        + prekickoff_violations
        + wrong_season_rows
        + frozen_spec_violations
        + additive_identity_violations
        + nonfinite_rows
    )
    return LedgerIntegrity(
        ledger="recent_form_total",
        spec_version=RECENT_FORM_SPEC_VERSION,
        source_path=str(source_path) if source_path is not None else None,
        sha256=_sha256(source_path),
        rows=frame.height,
        unique_games=frame.get_column("game_id").n_unique(),
        earliest_capture=earliest,
        latest_capture=latest,
        duplicate_games=duplicates,
        prekickoff_violations=pregame_violations if False else prekickoff_violations,
        wrong_season_rows=wrong_season_rows,
        frozen_spec_violations=frozen_spec_violations,
        additive_identity_violations=additive_identity_violations,
        nonfinite_rows=nonfinite_rows,
        status="PASS" if violations == 0 else "FAIL",
    )


def audit_qb_total_ledger(
    frame: pl.DataFrame,
    *,
    source_path: str | Path | None = None,
) -> LedgerIntegrity:
    """Verify the frozen QB-quality total ledger without mutating it."""

    required = {
        "ledger_version",
        "spec_version",
        "captured_at",
        "kickoff",
        "season",
        "game_id",
        "baseline_total",
        "qb_total_shadow_total",
        "qb_total_shadow_correction",
        "qb_total_feature_set",
        "qb_total_ridge_alpha",
        "qb_prior_dropbacks",
        "training_seasons",
        "qb_total_release_state",
    }
    require_columns(frame, required, "qb_total_forward_ledger")
    if frame.is_empty():
        raise DataContractError("QB-total forward ledger is empty")

    prekickoff_violations = 0
    wrong_season_rows = 0
    frozen_spec_violations = 0
    additive_identity_violations = 0
    nonfinite_rows = 0
    expected_training = ";".join(str(value) for value in QB_TOTAL_TRAINING_SEASONS)
    for row in frame.iter_rows(named=True):
        captured = _parse_utc(row["captured_at"])
        kickoff = _parse_utc(row["kickoff"])
        if captured is None or kickoff is None or captured >= kickoff:
            prekickoff_violations += 1
        if int(row["season"]) != QB_TOTAL_FORWARD_SEASON:
            wrong_season_rows += 1
        frozen_ok = (
            int(row["ledger_version"]) == QB_TOTAL_LEDGER_VERSION
            and str(row["spec_version"]) == QB_TOTAL_SPEC_VERSION
            and str(row["qb_total_feature_set"]) == VALIDATED_QB_TOTAL_FEATURE_SET
            and _close(row["qb_total_ridge_alpha"], VALIDATED_QB_TOTAL_ALPHA)
            and _close(row["qb_prior_dropbacks"], QB_PRIOR_DROPBACKS)
            and str(row["training_seasons"]) == expected_training
            and str(row["qb_total_release_state"]).upper() == "SHADOW"
        )
        if not frozen_ok:
            frozen_spec_violations += 1
        numeric = (
            row["baseline_total"],
            row["qb_total_shadow_total"],
            row["qb_total_shadow_correction"],
        )
        if not all(_finite(value) for value in numeric):
            nonfinite_rows += 1
        elif not _close(
            float(row["baseline_total"]) + float(row["qb_total_shadow_correction"]),
            row["qb_total_shadow_total"],
        ):
            additive_identity_violations += 1

    duplicates = _duplicate_games(frame)
    earliest, latest = _capture_range(frame)
    violations = (
        duplicates
        + prekickoff_violations
        + wrong_season_rows
        + frozen_spec_violations
        + additive_identity_violations
        + nonfinite_rows
    )
    return LedgerIntegrity(
        ledger="qb_quality_total",
        spec_version=QB_TOTAL_SPEC_VERSION,
        source_path=str(source_path) if source_path is not None else None,
        sha256=_sha256(source_path),
        rows=frame.height,
        unique_games=frame.get_column("game_id").n_unique(),
        earliest_capture=earliest,
        latest_capture=latest,
        duplicate_games=duplicates,
        prekickoff_violations=prekickoff_violations,
        wrong_season_rows=wrong_season_rows,
        frozen_spec_violations=frozen_spec_violations,
        additive_identity_violations=additive_identity_violations,
        nonfinite_rows=nonfinite_rows,
        status="PASS" if violations == 0 else "FAIL",
    )


def audit_cross_ledger(
    recent_form: pl.DataFrame,
    qb_total: pl.DataFrame,
) -> CrossLedgerIntegrity:
    """Cross-check canonical baseline and game identity across frozen ledgers."""

    require_columns(
        recent_form,
        {"game_id", "kickoff", "home_team", "away_team", "baseline_total"},
        "recent_form_forward_ledger",
    )
    require_columns(
        qb_total,
        {"game_id", "kickoff", "home_team", "away_team", "baseline_total"},
        "qb_total_forward_ledger",
    )
    recent_rows = {
        str(row["game_id"]): row for row in recent_form.iter_rows(named=True)
    }
    qb_rows = {str(row["game_id"]): row for row in qb_total.iter_rows(named=True)}
    overlap = sorted(set(recent_rows) & set(qb_rows))
    baseline_disagreements = 0
    kickoff_disagreements = 0
    team_disagreements = 0
    max_baseline_difference = 0.0
    for game_id in overlap:
        recent = recent_rows[game_id]
        qb = qb_rows[game_id]
        if _finite(recent["baseline_total"]) and _finite(qb["baseline_total"]):
            difference = abs(float(recent["baseline_total"]) - float(qb["baseline_total"]))
            max_baseline_difference = max(max_baseline_difference, difference)
            if difference > FLOAT_TOLERANCE:
                baseline_disagreements += 1
        else:
            baseline_disagreements += 1
        if _parse_utc(recent["kickoff"]) != _parse_utc(qb["kickoff"]):
            kickoff_disagreements += 1
        if (
            str(recent["home_team"]) != str(qb["home_team"])
            or str(recent["away_team"]) != str(qb["away_team"])
        ):
            team_disagreements += 1

    violations = baseline_disagreements + kickoff_disagreements + team_disagreements
    status = "PASS" if overlap and violations == 0 else "FAIL"
    return CrossLedgerIntegrity(
        overlapping_games=len(overlap),
        baseline_disagreements=baseline_disagreements,
        kickoff_disagreements=kickoff_disagreements,
        team_disagreements=team_disagreements,
        max_baseline_abs_difference=max_baseline_difference,
        status=status,
    )


def audit_forward_evidence(
    recent_form: pl.DataFrame,
    qb_total: pl.DataFrame,
    *,
    recent_form_path: str | Path | None = None,
    qb_total_path: str | Path | None = None,
) -> ForwardEvidenceIntegrity:
    """Return one read-only integrity verdict across both frozen forward ledgers."""

    recent = audit_recent_form_ledger(recent_form, source_path=recent_form_path)
    qb = audit_qb_total_ledger(qb_total, source_path=qb_total_path)
    cross = audit_cross_ledger(recent_form, qb_total)
    overall = "PASS" if {recent.status, qb.status, cross.status} == {"PASS"} else "FAIL"
    return ForwardEvidenceIntegrity(
        recent_form=recent,
        qb_total=qb,
        cross_ledger=cross,
        overall_status=overall,
        promotion_eligible=False,
        canonical_score_change_enabled=False,
    )
