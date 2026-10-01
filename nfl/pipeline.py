"""Canonical NFL operational pipeline modeled after the CFB production path."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from .context import build_current_context
from .current import run_current_projection, unplayed_regular_games
from .data import NFLDataClient
from .decision_ledger import append_portfolio_decisions
from .espn_market import ESPNMarketClient
from .free_market_backtest import build_archive_projection_dataset
from .health import write_health
from .line_history import append_market_snapshots
from .market_intel import build_market_intelligence
from .model_card import write_model_card
from .monitoring import write_live_monitoring
from .policy import load_policy
from .portfolio import apply_portfolio_controls
from .probability import evaluate_probability_holdout
from .proof import write_evidence_report
from .release_gate import write_release_gate
from .reporting import write_canonical_report


def _probability_meta(historical: pl.DataFrame, season: int) -> dict[str, object]:
    past = historical.filter(pl.col("season") < season)
    validation_season = season - 2
    holdout_season = season - 1
    try:
        evaluation = evaluate_probability_holdout(
            past,
            validation_season=validation_season,
            holdout_season=holdout_season,
        )
    except Exception as exc:
        return {
            "status": "BLOCKED",
            "validation_season": validation_season,
            "holdout_season": holdout_season,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    metrics = evaluation.holdout_calibrated
    return {
        "status": "READY",
        "validation_season": evaluation.validation_season,
        "holdout_season": evaluation.holdout_season,
        "training_games": evaluation.training_games,
        "holdout_games": evaluation.holdout_games,
        "margin_scale": evaluation.margin_scale,
        "total_scale": evaluation.total_scale,
        "margin_nll": metrics.margin_nll,
        "total_nll": metrics.total_nll,
        "home_win_brier": metrics.home_win_brier,
        "margin_50_coverage": metrics.margin_50_coverage,
        "margin_80_coverage": metrics.margin_80_coverage,
        "total_50_coverage": metrics.total_50_coverage,
        "total_80_coverage": metrics.total_80_coverage,
        "candidate_pass": evaluation.candidate_pass,
    }


def _qb_coverage(projection: pl.DataFrame) -> float:
    if projection.is_empty():
        return 0.0
    required = {"home_qb_proxy_id", "away_qb_proxy_id"}
    if not required.issubset(projection.columns):
        return 0.0
    covered = projection.filter(
        pl.col("home_qb_proxy_id").is_not_null()
        & pl.col("away_qb_proxy_id").is_not_null()
    ).height
    return covered / projection.height


def _context_sources(
    source: NFLDataClient,
    season: int,
    *,
    refresh: bool,
) -> tuple[dict[str, pl.DataFrame], dict[str, str]]:
    frames: dict[str, pl.DataFrame] = {}
    status: dict[str, str] = {}
    loaders = {
        "injuries": source.load_injuries,
        "depth_charts": source.load_depth_charts,
        "rosters_weekly": source.load_rosters_weekly,
    }
    for name, loader in loaders.items():
        try:
            frames[name] = loader([season], refresh=refresh)
            status[name] = "OK"
        except Exception as exc:
            frames[name] = pl.DataFrame()
            status[name] = f"ERROR: {type(exc).__name__}: {exc}"
    return frames, status


def _merge_context_meta(
    context_meta: dict[str, object],
    projection: pl.DataFrame,
) -> dict[str, object]:
    qb = _qb_coverage(projection)
    components = context_meta.get("components")
    if not isinstance(components, dict):
        components = {}
    combined = {
        "quarterback": qb,
        "injuries_personnel": float(components.get("injuries_personnel", 0.0) or 0.0),
        "weather_stadium": float(components.get("weather_stadium", 0.0) or 0.0),
        "rest_travel": float(components.get("rest_travel", 0.0) or 0.0),
    }
    coverage = sum(combined.values()) / len(combined)
    output = dict(context_meta)
    output.update(
        {
            "status": "READY" if min(combined.values()) >= 0.90 else "PARTIAL",
            "coverage": coverage,
            "qb_coverage": qb,
            "components": combined,
            "score_adjustment_enabled": False,
        }
    )
    return output


def run_operational_pipeline(
    season: int,
    week: int | None = None,
    *,
    history_seasons: int = 4,
    refresh: bool = False,
    capture_lines: bool = True,
    persist_decisions: bool = True,
    client: NFLDataClient | None = None,
) -> tuple[pl.DataFrame, dict[str, object]]:
    """Run the CFB-style NFL operational path without bypassing any release gate."""

    if history_seasons < 3:
        raise ValueError("history_seasons must be >= 3")
    source = client or NFLDataClient()
    run_at = datetime.now(UTC)
    projection, projection_audit = run_current_projection(
        season,
        week,
        client=source,
        refresh=refresh,
    )
    target_week = projection_audit.week
    history_start = season - history_seasons
    schedules = source.load_schedules(
        list(range(history_start - 1, season + 1)),
        refresh=refresh,
    )
    targets = unplayed_regular_games(schedules, season, target_week)
    markets = ESPNMarketClient().current_markets(targets, week=target_week)

    historical = build_archive_projection_dataset(
        schedules,
        start_season=history_start,
        end_season=season,
    ).filter(
        (pl.col("season") < season)
        | ((pl.col("season") == season) & (pl.col("week") < target_week))
    )

    policy = load_policy()
    candidates, market_meta = build_market_intelligence(
        projection,
        targets,
        markets,
        historical,
        policy=policy,
    )

    context_frames, context_source_status = _context_sources(
        source,
        season,
        refresh=refresh,
    )
    try:
        context_frame, raw_context_meta = build_current_context(
            targets,
            season=season,
            week=target_week,
            injuries=context_frames["injuries"],
            depth_charts=context_frames["depth_charts"],
            rosters=context_frames["rosters_weekly"],
            as_of=run_at,
        )
    except Exception as exc:
        context_frame = pl.DataFrame()
        raw_context_meta = {
            "status": "BLOCKED",
            "coverage": 0.0,
            "components": {},
            "weather_errors": [],
            "reason": f"{type(exc).__name__}: {exc}",
            "score_adjustment_enabled": False,
        }
    context = _merge_context_meta(raw_context_meta, projection)
    if not candidates.is_empty() and not context_frame.is_empty():
        candidates = candidates.join(context_frame, on="game_id", how="left")

    evidence = write_evidence_report()
    probability = _probability_meta(historical, season)
    context_errors = [
        value for value in context_source_status.values() if value.startswith("ERROR:")
    ]
    data_quality = {
        "status": "WARN" if context_errors else "OK",
        "projection_games": projection.height,
        "target_games": targets.height,
        "market_rows": candidates.height,
        "context_source_errors": context_errors,
    }
    meta: dict[str, object] = {
        "generated_at": run_at.isoformat(),
        "season": season,
        "week": target_week,
        "projection_audit": projection_audit.to_dict(),
        "market_coverage": {
            "games": market_meta.get("games", projection.height),
            "moneyline": market_meta.get("moneyline", 0),
            "spread": market_meta.get("spread", 0),
            "total": market_meta.get("total", 0),
        },
        "market_intelligence": market_meta,
        "probability": probability,
        "current_context": context,
        "data_quality": data_quality,
        "sources": {
            "football": "nflverse",
            "current_market": "ESPN public endpoints",
            "historical_market": "nflverse public archive",
            **context_source_status,
            "weather": "Open-Meteo",
        },
    }

    monitor = write_live_monitoring(candidates, meta)
    gate = write_release_gate(meta, monitor, data_quality)
    allocated, portfolio = apply_portfolio_controls(
        candidates,
        policy=policy,
        release_gate=gate,
    )
    line_capture = (
        append_market_snapshots(markets, targets) if capture_lines else {"status": "disabled"}
    )
    ledger = (
        append_portfolio_decisions(allocated)
        if persist_decisions
        else {"status": "disabled"}
    )
    health = write_health(meta, monitor, gate)
    model_card = write_model_card(meta, gate, evidence)
    report = write_canonical_report(
        allocated,
        meta=meta,
        portfolio=portfolio,
        monitoring=monitor,
        release_gate=gate,
        health=health,
        evidence=evidence,
        model_card=model_card,
        ledger=ledger,
        line_capture=line_capture,
    )
    return allocated, report
