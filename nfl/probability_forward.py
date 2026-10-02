"""Prospective 2026 evidence for frozen probability-calibration candidates.

Stage 23 selected a total-dispersion scale of 0.9 and an unregularized one-feature
logistic home-win calibrator using pre-2026 development data. This module freezes those
forms for forward evidence only. It never reconstructs historical forward predictions
and never changes canonical market probabilities or betting decisions.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from .contracts import require_columns
from .probability import GaussianScoreDistribution
from .win_probability import LogisticWinModel

NFL_SCHEDULE_TIMEZONE = ZoneInfo("America/New_York")
PROBABILITY_FORWARD_SEASON = 2026
PROBABILITY_LEDGER_VERSION = 1
PROBABILITY_SPEC_VERSION = "stage23_nested_probability_v1"
PROBABILITY_TRAINING_SEASONS = (2021, 2022, 2023, 2024, 2025)
FROZEN_MARGIN_SCALE = 1.0
FROZEN_TOTAL_SCALE = 0.9
FROZEN_LOGISTIC_ALPHA = 0.0
DEFAULT_MINIMUM_GAMES = 128

PROBABILITY_FORWARD_FIELDS = (
    "ledger_version",
    "spec_version",
    "captured_at",
    "kickoff",
    "season",
    "week",
    "game_id",
    "home_team",
    "away_team",
    "baseline_home_margin",
    "baseline_total",
    "baseline_gaussian_home_win_probability",
    "shadow_logistic_home_win_probability",
    "baseline_total_mean",
    "baseline_total_sigma",
    "shadow_total_mean",
    "shadow_total_sigma",
    "margin_scale",
    "total_scale",
    "logistic_alpha",
    "training_seasons",
    "release_state",
)


@dataclass(frozen=True)
class ProbabilityLossInterval:
    point_improvement: float
    lower: float
    upper: float
    probability_improvement: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class FrozenProbabilityModels:
    baseline_gaussian: GaussianScoreDistribution
    shadow_gaussian: GaussianScoreDistribution
    logistic: LogisticWinModel
    training_games: int


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
    return str(row.get("game_id", "")), str(row.get("spec_version", ""))


def fit_frozen_probability_models(historical: pl.DataFrame) -> FrozenProbabilityModels:
    """Refit fixed Stage 23 forms on all canonical pre-2026 development rows."""

    require_columns(
        historical,
        {
            "season",
            "projected_home_margin",
            "projected_total",
            "actual_home_margin",
            "actual_total",
        },
        "probability_forward_training",
    )
    allowed = historical.filter(pl.col("season").is_in(PROBABILITY_TRAINING_SEASONS))
    if allowed.height != historical.height:
        raise ValueError("probability forward training contains a non-frozen season")
    baseline = GaussianScoreDistribution().fit(historical)
    shadow = GaussianScoreDistribution(
        margin_scale=FROZEN_MARGIN_SCALE,
        total_scale=FROZEN_TOTAL_SCALE,
    ).fit(historical)
    logistic = LogisticWinModel(alpha=FROZEN_LOGISTIC_ALPHA).fit(historical)
    return FrozenProbabilityModels(
        baseline_gaussian=baseline,
        shadow_gaussian=shadow,
        logistic=logistic,
        training_games=historical.height,
    )


def attach_probability_shadow(
    projection: pl.DataFrame,
    historical: pl.DataFrame,
) -> tuple[pl.DataFrame, dict[str, object]]:
    """Attach frozen probability diagnostics without altering canonical score columns."""

    require_columns(
        projection,
        {
            "season",
            "week",
            "game_id",
            "home_team",
            "away_team",
            "baseline_home_margin",
            "baseline_total",
        },
        "current_probability_projection",
    )
    models = fit_frozen_probability_models(historical)
    baseline_gaussian = models.baseline_gaussian
    shadow_gaussian = models.shadow_gaussian
    logistic = models.logistic

    assert baseline_gaussian.total_mean is not None
    assert baseline_gaussian.total_sigma is not None
    assert shadow_gaussian.total_mean is not None
    assert shadow_gaussian.total_sigma is not None

    attached = projection.with_columns(
        pl.col("baseline_home_margin")
        .map_elements(
            baseline_gaussian.home_win_probability,
            return_dtype=pl.Float64,
        )
        .alias("baseline_gaussian_home_win_probability"),
        pl.col("baseline_home_margin")
        .map_elements(
            logistic.predict_probability,
            return_dtype=pl.Float64,
        )
        .alias("shadow_logistic_home_win_probability"),
        (pl.col("baseline_total") + float(baseline_gaussian.total_mean)).alias(
            "baseline_total_mean"
        ),
        pl.lit(float(baseline_gaussian.total_sigma)).alias("baseline_total_sigma"),
        (pl.col("baseline_total") + float(shadow_gaussian.total_mean)).alias(
            "shadow_total_mean"
        ),
        pl.lit(float(shadow_gaussian.total_sigma)).alias("shadow_total_sigma"),
        pl.lit("SHADOW").alias("probability_release_state"),
    )
    meta = {
        "status": "READY",
        "spec_version": PROBABILITY_SPEC_VERSION,
        "training_seasons": list(PROBABILITY_TRAINING_SEASONS),
        "training_games": models.training_games,
        "margin_scale": FROZEN_MARGIN_SCALE,
        "total_scale": FROZEN_TOTAL_SCALE,
        "logistic_alpha": FROZEN_LOGISTIC_ALPHA,
        "release_state": "SHADOW",
        "canonical_probability_change_enabled": False,
    }
    return attached, meta


def load_probability_forward_predictions(
    path: str | Path = "history/probability_shadow_predictions_v1.csv",
) -> pl.DataFrame:
    source = Path(path)
    if not source.exists() or not source.stat().st_size:
        return pl.DataFrame()
    return pl.read_csv(source, try_parse_dates=False)


def append_probability_forward_predictions(
    projection: pl.DataFrame,
    targets: pl.DataFrame,
    *,
    captured_at: datetime | None = None,
    path: str | Path = "history/probability_shadow_predictions_v1.csv",
) -> dict[str, object]:
    """Persist the first eligible frozen probability snapshot for each 2026 game."""

    require_columns(
        projection,
        {
            "season",
            "week",
            "game_id",
            "home_team",
            "away_team",
            "baseline_home_margin",
            "baseline_total",
            "baseline_gaussian_home_win_probability",
            "shadow_logistic_home_win_probability",
            "baseline_total_mean",
            "baseline_total_sigma",
            "shadow_total_mean",
            "shadow_total_sigma",
            "probability_release_state",
        },
        "probability_shadow_projection",
    )
    require_columns(targets, {"game_id", "gameday", "gametime"}, "probability_shadow_targets")

    now = captured_at or datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    now = now.astimezone(UTC)

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
    for row in projection.iter_rows(named=True):
        if int(row["season"]) != PROBABILITY_FORWARD_SEASON:
            skipped_wrong_season += 1
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
        if str(row.get("probability_release_state", "")).upper() != "SHADOW":
            skipped_incomplete += 1
            continue
        numeric_columns = (
            "baseline_home_margin",
            "baseline_total",
            "baseline_gaussian_home_win_probability",
            "shadow_logistic_home_win_probability",
            "baseline_total_mean",
            "baseline_total_sigma",
            "shadow_total_mean",
            "shadow_total_sigma",
        )
        if not all(_finite(row.get(column)) for column in numeric_columns):
            skipped_incomplete += 1
            continue

        ledger_row: dict[str, object] = {
            "ledger_version": PROBABILITY_LEDGER_VERSION,
            "spec_version": PROBABILITY_SPEC_VERSION,
            "captured_at": now.isoformat(),
            "kickoff": kickoff.isoformat(),
            "season": int(row["season"]),
            "week": int(row["week"]),
            "game_id": game_id,
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "baseline_home_margin": float(row["baseline_home_margin"]),
            "baseline_total": float(row["baseline_total"]),
            "baseline_gaussian_home_win_probability": float(
                row["baseline_gaussian_home_win_probability"]
            ),
            "shadow_logistic_home_win_probability": float(
                row["shadow_logistic_home_win_probability"]
            ),
            "baseline_total_mean": float(row["baseline_total_mean"]),
            "baseline_total_sigma": float(row["baseline_total_sigma"]),
            "shadow_total_mean": float(row["shadow_total_mean"]),
            "shadow_total_sigma": float(row["shadow_total_sigma"]),
            "margin_scale": FROZEN_MARGIN_SCALE,
            "total_scale": FROZEN_TOTAL_SCALE,
            "logistic_alpha": FROZEN_LOGISTIC_ALPHA,
            "training_seasons": _training_signature(PROBABILITY_TRAINING_SEASONS),
            "release_state": "SHADOW",
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
            writer = csv.DictWriter(handle, fieldnames=list(PROBABILITY_FORWARD_FIELDS))
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
        "spec_version": PROBABILITY_SPEC_VERSION,
    }


def _loss_interval(
    improvement: np.ndarray,
    *,
    iterations: int,
    confidence: float,
    seed: int,
) -> ProbabilityLossInterval:
    values = np.asarray(improvement, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("probability loss improvement must be finite and non-empty")
    if iterations < 500:
        raise ValueError("iterations must be >= 500")
    rng = np.random.default_rng(seed)
    samples = np.empty(iterations, dtype=float)
    for index in range(iterations):
        draw = rng.integers(0, values.size, size=values.size)
        samples[index] = float(np.mean(values[draw]))
    alpha = (1.0 - confidence) / 2.0
    return ProbabilityLossInterval(
        point_improvement=float(np.mean(values)),
        lower=float(np.quantile(samples, alpha)),
        upper=float(np.quantile(samples, 1.0 - alpha)),
        probability_improvement=float(np.mean(samples > 0.0)),
    )


def _status(
    *,
    games: int,
    minimum_games: int,
    intervals: tuple[ProbabilityLossInterval, ...],
) -> str:
    if games < minimum_games:
        return "SHADOW_INSUFFICIENT_SAMPLE"
    if any(value.point_improvement <= 0.0 for value in intervals):
        return "SHADOW_FAILING"
    if all(value.lower > 0.0 for value in intervals):
        return "PROMOTION_EVIDENCE"
    return "SHADOW_INCONCLUSIVE"


def _binary_losses(actual: np.ndarray, probability: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    predicted = np.clip(np.asarray(probability, dtype=float), 1e-9, 1.0 - 1e-9)
    brier = np.square(predicted - actual)
    log_loss = -(actual * np.log(predicted) + (1.0 - actual) * np.log(1.0 - predicted))
    return brier, log_loss


def _ece(actual: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    score = 0.0
    for index in range(bins):
        upper = edges[index + 1] + (1e-12 if index == bins - 1 else 0.0)
        mask = (probability >= edges[index]) & (probability < upper)
        if np.any(mask):
            score += float(np.mean(mask)) * abs(
                float(np.mean(actual[mask])) - float(np.mean(probability[mask]))
            )
    return float(score)


def _normal_nll(actual: np.ndarray, mean: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (
        0.5 * np.square((actual - mean) / sigma)
        + np.log(sigma)
        + 0.5 * math.log(2.0 * math.pi)
    )


def _empty_summary(status: str, *, ledger_rows: int = 0) -> dict[str, object]:
    return {
        "status": status,
        "ledger_rows": ledger_rows,
        "graded_games": 0,
        "promotion_eligible": False,
        "spec_version": PROBABILITY_SPEC_VERSION,
        "canonical_probability_change_enabled": False,
        "meaning": (
            "Prospective 2026 probability evidence only. Historical predictions are not "
            "reconstructed into this ledger."
        ),
    }


def grade_probability_forward_predictions(
    schedules: pl.DataFrame,
    predictions: pl.DataFrame,
    *,
    minimum_games: int = DEFAULT_MINIMUM_GAMES,
    bootstrap_iterations: int = 5000,
    confidence: float = 0.95,
) -> tuple[pl.DataFrame, dict[str, object]]:
    """Grade persisted first-snapshot probability shadows against final outcomes."""

    if predictions.is_empty():
        return pl.DataFrame(), _empty_summary("NO_PREDICTIONS")
    require_columns(
        predictions,
        {
            "spec_version",
            "captured_at",
            "kickoff",
            "season",
            "week",
            "game_id",
            "baseline_gaussian_home_win_probability",
            "shadow_logistic_home_win_probability",
            "baseline_total_mean",
            "baseline_total_sigma",
            "shadow_total_mean",
            "shadow_total_sigma",
            "margin_scale",
            "total_scale",
            "logistic_alpha",
            "training_seasons",
            "release_state",
        },
        "probability_forward_predictions",
    )
    require_columns(
        schedules,
        {"season", "week", "game_id", "home_score", "away_score"},
        "probability_forward_schedules",
    )

    expected_training = _training_signature(PROBABILITY_TRAINING_SEASONS)
    valid_rows: list[dict[str, object]] = []
    for row in predictions.iter_rows(named=True):
        captured = _parse_utc(row.get("captured_at"))
        kickoff = _parse_utc(row.get("kickoff"))
        if captured is None or kickoff is None or captured >= kickoff:
            continue
        if int(row["season"]) != PROBABILITY_FORWARD_SEASON:
            continue
        if str(row["spec_version"]) != PROBABILITY_SPEC_VERSION:
            continue
        if float(row["margin_scale"]) != FROZEN_MARGIN_SCALE:
            continue
        if float(row["total_scale"]) != FROZEN_TOTAL_SCALE:
            continue
        if float(row["logistic_alpha"]) != FROZEN_LOGISTIC_ALPHA:
            continue
        if str(row["training_seasons"]) != expected_training:
            continue
        if str(row["release_state"]).upper() != "SHADOW":
            continue
        valid_rows.append(row)

    if not valid_rows:
        return pl.DataFrame(), _empty_summary(
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
        return graded, _empty_summary("NO_GRADED_GAMES", ledger_rows=predictions.height)

    graded = graded.with_columns(
        (
            pl.col("home_score").cast(pl.Float64)
            - pl.col("away_score").cast(pl.Float64)
        ).alias("actual_home_margin"),
        (
            pl.col("home_score").cast(pl.Float64)
            + pl.col("away_score").cast(pl.Float64)
        ).alias("actual_total"),
    )

    actual_margin = np.asarray(graded.get_column("actual_home_margin"), dtype=float)
    non_ties = actual_margin != 0
    actual_win = (actual_margin[non_ties] > 0).astype(float)
    baseline_win_p = np.asarray(
        graded.get_column("baseline_gaussian_home_win_probability"), dtype=float
    )[non_ties]
    shadow_win_p = np.asarray(
        graded.get_column("shadow_logistic_home_win_probability"), dtype=float
    )[non_ties]

    baseline_brier, baseline_log = _binary_losses(actual_win, baseline_win_p)
    shadow_brier, shadow_log = _binary_losses(actual_win, shadow_win_p)
    brier_interval = _loss_interval(
        baseline_brier - shadow_brier,
        iterations=bootstrap_iterations,
        confidence=confidence,
        seed=20261003,
    )
    log_interval = _loss_interval(
        baseline_log - shadow_log,
        iterations=bootstrap_iterations,
        confidence=confidence,
        seed=20261004,
    )
    home_status = _status(
        games=int(actual_win.size),
        minimum_games=minimum_games,
        intervals=(brier_interval, log_interval),
    )

    actual_total = np.asarray(graded.get_column("actual_total"), dtype=float)
    baseline_total_mean = np.asarray(graded.get_column("baseline_total_mean"), dtype=float)
    baseline_total_sigma = np.asarray(graded.get_column("baseline_total_sigma"), dtype=float)
    shadow_total_mean = np.asarray(graded.get_column("shadow_total_mean"), dtype=float)
    shadow_total_sigma = np.asarray(graded.get_column("shadow_total_sigma"), dtype=float)
    baseline_total_nll = _normal_nll(actual_total, baseline_total_mean, baseline_total_sigma)
    shadow_total_nll = _normal_nll(actual_total, shadow_total_mean, shadow_total_sigma)
    total_interval = _loss_interval(
        baseline_total_nll - shadow_total_nll,
        iterations=bootstrap_iterations,
        confidence=confidence,
        seed=20261005,
    )
    total_status = _status(
        games=graded.height,
        minimum_games=minimum_games,
        intervals=(total_interval,),
    )

    graded = graded.with_columns(
        pl.Series("baseline_brier_loss", np.full(graded.height, np.nan)),
        pl.Series("shadow_brier_loss", np.full(graded.height, np.nan)),
        pl.Series("baseline_total_nll", baseline_total_nll),
        pl.Series("shadow_total_nll", shadow_total_nll),
    )
    if np.any(non_ties):
        baseline_values = np.full(graded.height, np.nan)
        shadow_values = np.full(graded.height, np.nan)
        baseline_values[non_ties] = baseline_brier
        shadow_values[non_ties] = shadow_brier
        graded = graded.with_columns(
            pl.Series("baseline_brier_loss", baseline_values),
            pl.Series("shadow_brier_loss", shadow_values),
        )

    summary = {
        "status": "READY",
        "ledger_rows": predictions.height,
        "valid_pregame_rows": len(valid_rows),
        "graded_games": graded.height,
        "non_tied_games": int(actual_win.size),
        "minimum_games": minimum_games,
        "spec_version": PROBABILITY_SPEC_VERSION,
        "training_seasons": list(PROBABILITY_TRAINING_SEASONS),
        "home_win": {
            "status": home_status,
            "baseline_brier": float(np.mean(baseline_brier)),
            "shadow_brier": float(np.mean(shadow_brier)),
            "brier_improvement": brier_interval.to_dict(),
            "baseline_log_loss": float(np.mean(baseline_log)),
            "shadow_log_loss": float(np.mean(shadow_log)),
            "log_loss_improvement": log_interval.to_dict(),
            "baseline_ece": _ece(actual_win, baseline_win_p),
            "shadow_ece": _ece(actual_win, shadow_win_p),
            "promotion_eligible": home_status == "PROMOTION_EVIDENCE",
        },
        "total_distribution": {
            "status": total_status,
            "baseline_nll": float(np.mean(baseline_total_nll)),
            "shadow_nll": float(np.mean(shadow_total_nll)),
            "nll_improvement": total_interval.to_dict(),
            "promotion_eligible": total_status == "PROMOTION_EVIDENCE",
        },
        "margin_scale": FROZEN_MARGIN_SCALE,
        "total_scale": FROZEN_TOTAL_SCALE,
        "logistic_alpha": FROZEN_LOGISTIC_ALPHA,
        "promotion_eligible": (
            home_status == "PROMOTION_EVIDENCE"
            or total_status == "PROMOTION_EVIDENCE"
        ),
        "canonical_probability_change_enabled": False,
        "meaning": (
            "First-snapshot pre-kickoff 2026 evidence only. Stage 23 forms are frozen "
            "and are not retuned on 2026 outcomes. Canonical probabilities stay unchanged."
        ),
    }
    return graded.sort(["season", "week", "game_id"]), summary


def write_probability_forward_report(
    schedules: pl.DataFrame,
    *,
    ledger_path: str | Path = "history/probability_shadow_predictions_v1.csv",
    report_path: str | Path = "reports/probability_forward.json",
    graded_path: str | Path = "reports/probability_forward_graded.csv",
    docs_path: str | Path = "docs/probability_forward.json",
    minimum_games: int = DEFAULT_MINIMUM_GAMES,
) -> dict[str, object]:
    predictions = load_probability_forward_predictions(ledger_path)
    graded, summary = grade_probability_forward_predictions(
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
