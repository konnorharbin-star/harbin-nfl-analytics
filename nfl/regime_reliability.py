"""Regime-specific NFL probability and edge reliability.

The fair-score model remains sportsbook-independent. This module only evaluates the
downstream model probability after a no-vig market probability exists. Segment
definitions are fixed in code before outcomes are inspected; they are not discovered
by searching historical ROI.

Reliability is deliberately conservative:
- only verified two-way entry quotes may produce an operational registry;
- validation and later holdout seasons must both support a segment;
- empirical win rates are Beta-Binomial shrunk toward the model probability;
- calibration and incremental value versus the no-vig market must both pass;
- missing or insufficient segment evidence fails closed for operational decisions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np
import polars as pl
from scipy.stats import beta as beta_distribution

from .contracts import DataContractError, require_columns

_EPS = 1e-9
PRIOR_STRENGTH = 40.0
CALIBRATION_GAP_LIMIT = 0.06
BRIER_TOLERANCE = 0.005
LOG_LOSS_TOLERANCE = 0.01
MIN_REALIZED_EDGE = 0.0
MIN_EDGE_RETENTION = 0.10
SEVERE_NEGATIVE_ROI = -0.10
MIN_CLV_COVERAGE_FOR_GATE = 0.50

MIN_GAMES_BY_DIMENSION: dict[str, int] = {
    "market": 100,
    "side": 45,
    "role": 45,
    "confidence": 30,
    "edge": 30,
    "margin_environment": 35,
    "total_environment": 35,
    "season_phase": 30,
}

REQUIRED_DIMENSIONS_BY_MARKET: dict[str, tuple[str, ...]] = {
    "moneyline": (
        "market",
        "side",
        "role",
        "confidence",
        "edge",
        "margin_environment",
        "season_phase",
    ),
    "spread": (
        "market",
        "side",
        "role",
        "confidence",
        "edge",
        "margin_environment",
        "season_phase",
    ),
    "total": (
        "market",
        "side",
        "confidence",
        "edge",
        "total_environment",
        "season_phase",
    ),
}

BASE_REQUIRED = {
    "season",
    "market_type",
    "side",
    "model_probability",
    "no_vig_probability",
    "probability_edge",
    "result",
}


@dataclass(frozen=True)
class SegmentMetrics:
    games: int
    wins: int
    losses: int
    pushes: int
    mean_model_probability: float | None
    mean_market_probability: float | None
    mean_model_edge: float | None
    raw_win_rate: float | None
    shrunk_win_rate: float | None
    raw_calibration_gap: float | None
    shrunk_calibration_gap: float | None
    posterior_low_80: float | None
    posterior_high_80: float | None
    realized_edge_shrunk: float | None
    edge_retention_ratio: float | None
    model_brier: float | None
    market_brier: float | None
    model_log_loss: float | None
    market_log_loss: float | None
    roi: float | None
    clv_samples: int
    clv_coverage: float
    average_clv: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _number(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _binary_result(value: object) -> float | None:
    result = str(value or "").strip().lower()
    if result == "win":
        return 1.0
    if result == "loss":
        return 0.0
    return None


def _confidence_bucket(probability: float) -> str:
    value = float(probability)
    if value < 0.50:
        return "below_50"
    if value < 0.55:
        return "50_55"
    if value < 0.58:
        return "55_58"
    if value <= 0.62:
        return "58_62"
    if value < 0.67:
        return "62_67"
    return "67_plus"


def _edge_bucket(edge: float) -> str:
    value = float(edge)
    if value < 0.02:
        return "below_02"
    if value < 0.04:
        return "02_04"
    if value < 0.06:
        return "04_06"
    if value < 0.10:
        return "06_10"
    return "10_plus"


def _margin_environment(projected_home_margin: float) -> str:
    magnitude = abs(float(projected_home_margin))
    if magnitude < 3.0:
        return "close"
    if magnitude < 7.0:
        return "moderate"
    return "large"


def _total_environment(projected_total: float) -> str:
    total = float(projected_total)
    if total < 42.0:
        return "low"
    if total <= 48.0:
        return "middle"
    return "high"


def _season_phase(week: int | None) -> str:
    if week is None:
        return "unknown"
    if week <= 4:
        return "early"
    if week <= 12:
        return "middle"
    return "late"


def _role(
    *,
    market_type: str,
    side: str,
    line: float | None,
    no_vig_probability: float,
) -> str | None:
    if market_type == "moneyline":
        if abs(no_vig_probability - 0.5) <= 1e-9:
            return "pickem"
        return "favorite" if no_vig_probability > 0.5 else "underdog"
    if market_type != "spread":
        return None
    if line is None or abs(line) <= 1e-9:
        return "pickem"
    # The historical/live row line is already from the selected side's perspective.
    del side
    return "favorite" if line < 0 else "underdog"


def candidate_segment_keys(
    *,
    market_type: str,
    side: str,
    model_probability: float,
    no_vig_probability: float,
    probability_edge: float,
    projected_home_margin: float,
    projected_total: float,
    line: float | None = None,
    week: int | None = None,
) -> dict[str, str]:
    """Return the fixed reliability segment for each applicable dimension."""

    market = str(market_type).strip().lower()
    side_name = str(side).strip().lower()
    if market not in REQUIRED_DIMENSIONS_BY_MARKET:
        raise ValueError(f"unsupported market_type: {market_type}")

    keys: dict[str, str] = {
        "market": f"market:{market}",
        "side": f"side:{market}:{side_name}",
        "confidence": f"confidence:{_confidence_bucket(model_probability)}",
        "edge": f"edge:{_edge_bucket(probability_edge)}",
        "season_phase": f"season_phase:{_season_phase(week)}",
    }
    role = _role(
        market_type=market,
        side=side_name,
        line=line,
        no_vig_probability=no_vig_probability,
    )
    if role is not None:
        keys["role"] = f"role:{market}:{role}"

    if market in {"moneyline", "spread"}:
        keys["margin_environment"] = (
            f"margin_environment:{_margin_environment(projected_home_margin)}"
        )
    if market == "total":
        keys["total_environment"] = (
            f"total_environment:{_total_environment(projected_total)}"
        )
    return keys


def _verified_sample(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    require_columns(frame, BASE_REQUIRED, "regime_reliability_bets")
    sample = frame
    if "entry_price_verified" in sample.columns:
        sample = sample.filter(
            pl.col("entry_price_verified")
            .cast(pl.Boolean, strict=False)
            .fill_null(False)
        )
    if "entry_quote_verified" in sample.columns:
        sample = sample.filter(
            pl.col("entry_quote_verified")
            .cast(pl.Boolean, strict=False)
            .fill_null(False)
        )
    return sample


def _segment_rows(frame: pl.DataFrame) -> dict[str, list[dict[str, object]]]:
    output: dict[str, list[dict[str, object]]] = {}
    for row in frame.iter_rows(named=True):
        market = str(row.get("market_type") or "").strip().lower()
        if market not in REQUIRED_DIMENSIONS_BY_MARKET:
            continue
        model_probability = _number(row.get("model_probability"))
        market_probability = _number(row.get("no_vig_probability"))
        edge = _number(row.get("probability_edge"))
        margin = _number(
            row.get("projected_home_margin", row.get("model_margin_home"))
        )
        total = _number(row.get("projected_total", row.get("model_total")))
        if (
            model_probability is None
            or market_probability is None
            or edge is None
            or margin is None
            or total is None
        ):
            continue
        line = _number(row.get("line", row.get("quant_price")))
        week_value = row.get("week")
        try:
            week = int(week_value) if week_value is not None else None
        except (TypeError, ValueError):
            week = None
        keys = candidate_segment_keys(
            market_type=market,
            side=str(row.get("side") or row.get("quant_side") or ""),
            model_probability=model_probability,
            no_vig_probability=market_probability,
            probability_edge=edge,
            projected_home_margin=margin,
            projected_total=total,
            line=line,
            week=week,
        )
        for key in keys.values():
            output.setdefault(key, []).append(row)
    return output


def _metrics(rows: list[dict[str, object]]) -> SegmentMetrics:
    if not rows:
        return SegmentMetrics(
            games=0,
            wins=0,
            losses=0,
            pushes=0,
            mean_model_probability=None,
            mean_market_probability=None,
            mean_model_edge=None,
            raw_win_rate=None,
            shrunk_win_rate=None,
            raw_calibration_gap=None,
            shrunk_calibration_gap=None,
            posterior_low_80=None,
            posterior_high_80=None,
            realized_edge_shrunk=None,
            edge_retention_ratio=None,
            model_brier=None,
            market_brier=None,
            model_log_loss=None,
            market_log_loss=None,
            roi=None,
            clv_samples=0,
            clv_coverage=0.0,
            average_clv=None,
        )

    outcomes: list[float] = []
    model_probabilities: list[float] = []
    market_probabilities: list[float] = []
    profits: list[float] = []
    clv: list[float] = []
    pushes = 0

    for row in rows:
        outcome = _binary_result(row.get("result"))
        if outcome is None:
            if str(row.get("result") or "").strip().lower() == "push":
                pushes += 1
            continue
        model_probability = _number(row.get("model_probability"))
        market_probability = _number(row.get("no_vig_probability"))
        if model_probability is None or market_probability is None:
            continue
        if not (
            0.0 < model_probability < 1.0
            and 0.0 < market_probability < 1.0
        ):
            continue
        outcomes.append(outcome)
        model_probabilities.append(model_probability)
        market_probabilities.append(market_probability)
        profit = _number(row.get("net_units"))
        if profit is not None:
            profits.append(profit)
        clv_value = _number(row.get("clv_proxy"))
        if clv_value is not None:
            clv.append(clv_value)

    games = len(outcomes)
    if games == 0:
        return SegmentMetrics(
            games=0,
            wins=0,
            losses=0,
            pushes=pushes,
            mean_model_probability=None,
            mean_market_probability=None,
            mean_model_edge=None,
            raw_win_rate=None,
            shrunk_win_rate=None,
            raw_calibration_gap=None,
            shrunk_calibration_gap=None,
            posterior_low_80=None,
            posterior_high_80=None,
            realized_edge_shrunk=None,
            edge_retention_ratio=None,
            model_brier=None,
            market_brier=None,
            model_log_loss=None,
            market_log_loss=None,
            roi=None,
            clv_samples=0,
            clv_coverage=0.0,
            average_clv=None,
        )

    actual = np.asarray(outcomes, dtype=float)
    model = np.clip(
        np.asarray(model_probabilities, dtype=float),
        _EPS,
        1.0 - _EPS,
    )
    market = np.clip(
        np.asarray(market_probabilities, dtype=float),
        _EPS,
        1.0 - _EPS,
    )
    wins = int(np.sum(actual))
    losses = games - wins
    mean_model = float(np.mean(model))
    mean_market = float(np.mean(market))
    mean_edge = mean_model - mean_market
    raw_rate = wins / games

    prior_alpha = max(_EPS, PRIOR_STRENGTH * mean_model)
    prior_beta = max(_EPS, PRIOR_STRENGTH * (1.0 - mean_model))
    posterior_alpha = prior_alpha + wins
    posterior_beta = prior_beta + losses
    shrunk_rate = posterior_alpha / (posterior_alpha + posterior_beta)
    posterior_low = float(
        beta_distribution.ppf(0.10, posterior_alpha, posterior_beta)
    )
    posterior_high = float(
        beta_distribution.ppf(0.90, posterior_alpha, posterior_beta)
    )
    realized_edge = shrunk_rate - mean_market
    retention = (
        realized_edge / mean_edge
        if abs(mean_edge) > 1e-9
        else None
    )

    model_brier = float(np.mean(np.square(model - actual)))
    market_brier = float(np.mean(np.square(market - actual)))
    model_log = float(
        -np.mean(
            actual * np.log(model)
            + (1.0 - actual) * np.log(1.0 - model)
        )
    )
    market_log = float(
        -np.mean(
            actual * np.log(market)
            + (1.0 - actual) * np.log(1.0 - market)
        )
    )
    roi = float(np.mean(profits)) if profits else None
    clv_samples = len(clv)
    clv_coverage = clv_samples / games if games else 0.0
    average_clv = float(np.mean(clv)) if clv else None

    return SegmentMetrics(
        games=games,
        wins=wins,
        losses=losses,
        pushes=pushes,
        mean_model_probability=mean_model,
        mean_market_probability=mean_market,
        mean_model_edge=mean_edge,
        raw_win_rate=raw_rate,
        shrunk_win_rate=shrunk_rate,
        raw_calibration_gap=raw_rate - mean_model,
        shrunk_calibration_gap=shrunk_rate - mean_model,
        posterior_low_80=posterior_low,
        posterior_high_80=posterior_high,
        realized_edge_shrunk=realized_edge,
        edge_retention_ratio=retention,
        model_brier=model_brier,
        market_brier=market_brier,
        model_log_loss=model_log,
        market_log_loss=market_log,
        roi=roi,
        clv_samples=clv_samples,
        clv_coverage=clv_coverage,
        average_clv=average_clv,
    )


def _dimension(key: str) -> str:
    return key.split(":", 1)[0]


def _season_pass(metrics: SegmentMetrics, *, dimension: str) -> tuple[bool, list[str]]:
    minimum = MIN_GAMES_BY_DIMENSION[dimension]
    reasons: list[str] = []
    if metrics.games < minimum:
        reasons.append(f"sample {metrics.games} < {minimum}")
        return False, reasons

    gap = metrics.shrunk_calibration_gap
    if gap is None or abs(gap) > CALIBRATION_GAP_LIMIT:
        reasons.append(
            "shrunk calibration gap exceeds "
            f"{CALIBRATION_GAP_LIMIT:.3f}"
        )
    if (
        metrics.model_brier is None
        or metrics.market_brier is None
        or metrics.model_brier > metrics.market_brier + BRIER_TOLERANCE
    ):
        reasons.append("model Brier does not match the no-vig market")
    if (
        metrics.model_log_loss is None
        or metrics.market_log_loss is None
        or metrics.model_log_loss > metrics.market_log_loss + LOG_LOSS_TOLERANCE
    ):
        reasons.append("model log loss does not match the no-vig market")

    if (
        metrics.mean_model_edge is not None
        and metrics.mean_model_edge > 0.005
    ):
        if (
            metrics.realized_edge_shrunk is None
            or metrics.realized_edge_shrunk <= MIN_REALIZED_EDGE
        ):
            reasons.append("positive modeled edge did not realize above market")
        retention = metrics.edge_retention_ratio
        if retention is None or retention < MIN_EDGE_RETENTION:
            reasons.append(
                "realized edge retained less than "
                f"{MIN_EDGE_RETENTION:.0%} of modeled edge"
            )

    if metrics.roi is not None and metrics.roi < SEVERE_NEGATIVE_ROI:
        reasons.append("segment ROI is severely negative")
    if (
        metrics.clv_coverage >= MIN_CLV_COVERAGE_FOR_GATE
        and metrics.average_clv is not None
        and metrics.average_clv < 0
    ):
        reasons.append("segment closing-line value is negative")

    return not reasons, reasons


def _segment_report(
    *,
    key: str,
    validation_rows: list[dict[str, object]],
    holdout_rows: list[dict[str, object]],
) -> dict[str, object]:
    dimension = _dimension(key)
    validation = _metrics(validation_rows)
    holdout = _metrics(holdout_rows)
    validation_pass, validation_reasons = _season_pass(
        validation,
        dimension=dimension,
    )
    holdout_pass, holdout_reasons = _season_pass(
        holdout,
        dimension=dimension,
    )
    minimum = MIN_GAMES_BY_DIMENSION[dimension]
    sufficient = (
        validation.games >= minimum
        and holdout.games >= minimum
    )
    if not sufficient:
        status = "INSUFFICIENT"
    elif validation_pass and holdout_pass:
        status = "RELIABLE"
    else:
        status = "UNRELIABLE"

    return {
        "key": key,
        "dimension": dimension,
        "minimum_games_per_season": minimum,
        "status": status,
        "validation_pass": validation_pass,
        "holdout_pass": holdout_pass,
        "validation_failures": validation_reasons,
        "holdout_failures": holdout_reasons,
        "validation": validation.to_dict(),
        "holdout": holdout.to_dict(),
    }


def _empty_report(reason: str) -> dict[str, object]:
    registry = {
        "version": 1,
        "status": "INSUFFICIENT_DATA",
        "operational_ready": False,
        "fail_closed": True,
        "reason": reason,
        "required_dimensions_by_market": {
            market: list(dimensions)
            for market, dimensions in REQUIRED_DIMENSIONS_BY_MARKET.items()
        },
        "segments": {},
        "market_status": {
            market: "INSUFFICIENT"
            for market in REQUIRED_DIMENSIONS_BY_MARKET
        },
    }
    return {
        "version": 1,
        "status": "INSUFFICIENT_DATA",
        "reason": reason,
        "verified_rows": 0,
        "validation_season": None,
        "holdout_season": None,
        "segments": {},
        "summary": {
            "reliable_segments": 0,
            "unreliable_segments": 0,
            "insufficient_segments": 0,
            "reliable_markets": 0,
        },
        "operational_registry": registry,
    }


def build_regime_reliability_report(
    bets: pl.DataFrame,
    *,
    validation_season: int | None = None,
    holdout_season: int | None = None,
) -> dict[str, object]:
    """Evaluate predeclared probability/edge regimes on consecutive seasons."""

    if bets.is_empty():
        return _empty_report("no historical market rows")
    try:
        sample = _verified_sample(bets)
    except DataContractError as exc:
        return _empty_report(str(exc))
    if sample.is_empty():
        return _empty_report("no verified two-way entry quotes")

    seasons = sorted(
        int(value)
        for value in (
            sample.get_column("season")
            .cast(pl.Int64, strict=False)
            .drop_nulls()
            .unique()
            .to_list()
        )
    )
    if validation_season is None or holdout_season is None:
        if len(seasons) < 2:
            return _empty_report("fewer than two verified seasons")
        validation_season = seasons[-2]
        holdout_season = seasons[-1]
    if validation_season >= holdout_season:
        raise ValueError("validation_season must be earlier than holdout_season")

    validation = sample.filter(
        pl.col("season").cast(pl.Int64, strict=False) == validation_season
    )
    holdout = sample.filter(
        pl.col("season").cast(pl.Int64, strict=False) == holdout_season
    )
    if validation.is_empty() or holdout.is_empty():
        return _empty_report(
            "validation and holdout seasons both require verified rows"
        )

    validation_segments = _segment_rows(validation)
    holdout_segments = _segment_rows(holdout)
    all_keys = sorted(
        set(validation_segments) | set(holdout_segments)
    )
    segments = {
        key: _segment_report(
            key=key,
            validation_rows=validation_segments.get(key, []),
            holdout_rows=holdout_segments.get(key, []),
        )
        for key in all_keys
    }

    market_status: dict[str, str] = {}
    for market in REQUIRED_DIMENSIONS_BY_MARKET:
        entry = segments.get(f"market:{market}")
        market_status[market] = (
            str(entry.get("status"))
            if isinstance(entry, dict)
            else "INSUFFICIENT"
        )

    reliable = sum(
        entry["status"] == "RELIABLE"
        for entry in segments.values()
    )
    unreliable = sum(
        entry["status"] == "UNRELIABLE"
        for entry in segments.values()
    )
    insufficient = sum(
        entry["status"] == "INSUFFICIENT"
        for entry in segments.values()
    )
    reliable_markets = sum(
        value == "RELIABLE" for value in market_status.values()
    )
    operational_ready = reliable_markets >= 2

    registry = {
        "version": 1,
        "status": "READY",
        "validation_season": validation_season,
        "holdout_season": holdout_season,
        "verified_rows": sample.height,
        "operational_ready": operational_ready,
        "fail_closed": True,
        "prior_strength": PRIOR_STRENGTH,
        "calibration_gap_limit": CALIBRATION_GAP_LIMIT,
        "brier_tolerance": BRIER_TOLERANCE,
        "log_loss_tolerance": LOG_LOSS_TOLERANCE,
        "minimum_games_by_dimension": MIN_GAMES_BY_DIMENSION,
        "required_dimensions_by_market": {
            market: list(dimensions)
            for market, dimensions in REQUIRED_DIMENSIONS_BY_MARKET.items()
        },
        "market_status": market_status,
        "segments": segments,
    }
    return {
        "version": 1,
        "status": "READY",
        "verified_rows": sample.height,
        "validation_season": validation_season,
        "holdout_season": holdout_season,
        "definitions_frozen_before_outcomes": True,
        "multiple_testing_search_disabled": True,
        "sample_shrinkage": (
            "Beta-Binomial posterior centered on mean model probability "
            f"with prior strength {PRIOR_STRENGTH:g}"
        ),
        "segments": segments,
        "summary": {
            "reliable_segments": reliable,
            "unreliable_segments": unreliable,
            "insufficient_segments": insufficient,
            "reliable_markets": reliable_markets,
            "market_status": market_status,
            "operational_ready": operational_ready,
        },
        "operational_registry": registry,
        "meaning": (
            "A segment is operationally reliable only when both the validation "
            "season and the later untouched holdout clear fixed calibration, "
            "incremental-value-versus-market, and severe economic-adversity checks."
        ),
    }


def assess_candidate_regime_reliability(
    registry: dict[str, object] | None,
    *,
    market_type: str,
    side: str,
    model_probability: float,
    no_vig_probability: float,
    probability_edge: float,
    projected_home_margin: float,
    projected_total: float,
    line: float | None = None,
    week: int | None = None,
) -> dict[str, object]:
    """Return candidate-specific regime reliability; missing evidence fails closed."""

    if not isinstance(registry, dict):
        return {
            "ready": False,
            "status": "BLOCKED",
            "reason": "regime reliability registry missing",
            "matched_segments": {},
            "blocked_segments": [],
            "missing_segments": [],
        }
    if not bool(registry.get("fail_closed", True)):
        return {
            "ready": True,
            "status": "NOT_ENFORCED",
            "reason": "regime reliability enforcement disabled",
            "matched_segments": {},
            "blocked_segments": [],
            "missing_segments": [],
        }

    market = str(market_type).strip().lower()
    required_map = registry.get("required_dimensions_by_market")
    if not isinstance(required_map, dict):
        required_map = {
            key: list(value)
            for key, value in REQUIRED_DIMENSIONS_BY_MARKET.items()
        }
    required = required_map.get(market)
    if not isinstance(required, list):
        required = list(REQUIRED_DIMENSIONS_BY_MARKET.get(market, ()))
    segments = registry.get("segments")
    if not isinstance(segments, dict):
        segments = {}

    keys = candidate_segment_keys(
        market_type=market,
        side=side,
        model_probability=model_probability,
        no_vig_probability=no_vig_probability,
        probability_edge=probability_edge,
        projected_home_margin=projected_home_margin,
        projected_total=projected_total,
        line=line,
        week=week,
    )

    matched: dict[str, dict[str, object]] = {}
    blocked: list[str] = []
    missing: list[str] = []
    for dimension in required:
        key = keys.get(str(dimension))
        if key is None:
            missing.append(str(dimension))
            continue
        entry = segments.get(key)
        if not isinstance(entry, dict):
            missing.append(key)
            continue
        status = str(entry.get("status") or "INSUFFICIENT").upper()
        matched[str(dimension)] = {
            "key": key,
            "status": status,
            "validation_games": (
                entry.get("validation", {}).get("games")
                if isinstance(entry.get("validation"), dict)
                else None
            ),
            "holdout_games": (
                entry.get("holdout", {}).get("games")
                if isinstance(entry.get("holdout"), dict)
                else None
            ),
        }
        if status != "RELIABLE":
            blocked.append(key)

    ready = not blocked and not missing
    reason = (
        "all required historical regimes are reliable"
        if ready
        else "candidate falls in unvalidated or unreliable historical regimes"
    )
    return {
        "ready": ready,
        "status": "RELIABLE" if ready else "BLOCKED",
        "reason": reason,
        "matched_segments": matched,
        "blocked_segments": blocked,
        "missing_segments": missing,
    }


def compact_registry(report: dict[str, object]) -> dict[str, object]:
    """Extract the persisted operational registry from a full audit report."""

    value = report.get("operational_registry")
    return value if isinstance(value, dict) else _empty_report(
        "regime reliability report missing registry"
    )["operational_registry"]
