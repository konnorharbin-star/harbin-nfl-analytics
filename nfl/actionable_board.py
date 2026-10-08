"""Human-only NFL research board.

Raw positive EV is an UNVALIDATED research estimate. A row is never presented
as a betting recommendation unless current personnel/QB context, holdout
reliability, edge shrinkage and quote provenance all pass. No execution action,
bookmaker order or bankroll operation is implemented here.
"""
from __future__ import annotations

import math
import polars as pl
from .contracts import require_columns

COLUMNS = (
    "season", "week", "game_id", "away_team", "home_team",
    "model_margin_home", "model_total", "quant_market", "quant_side",
    "quant_book", "quant_price", "quant_odds", "quant_probability",
    "quant_edge", "quant_ev", "quant_quote_at", "recommendation_status",
    "production_signal", "research_signal", "execution_ready",
    "portfolio_stake_units", "context_veto", "probability_reliability_veto",
)
EVIDENCE = (
    "regime_reliability_ready", "regime_reliability_status",
    "probability_reliability_ready", "context_injuries_personnel_fresh",
    "context_freshness_veto", "qb_certainty_veto", "qb_context_ready",
    "market_execution_verified", "market_quote_timestamp_verified",
    "market_quote_sanity_ok", "edge_discovery_tier",
    "edge_discovery_reason", "edge_shrunk_ev",
    "edge_market_shrinkage_status", "market_disagreement_severity",
    "market_dispersion_high",
)
EXTRA = (
    "predicted_home_score", "predicted_away_score", "projected_winner",
    "betting_action", "quote_quality", "evidence_status",
    "evidence_blockers", "conservative_ev",
)

def _true(row: dict[str, object], field: str) -> bool:
    return row.get(field) is True

def _positive(value: object) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(number) and number > 0

def _evaluate(row: dict[str, object]) -> tuple[str, str, str, float | None]:
    if not row.get("quant_market"):
        return ("NO_VERIFIED_MARKET", "NO_MARKET", "no model market offer", None)
    quote = all(
        row.get(field) not in (None, "")
        for field in ("quant_book", "quant_odds", "quant_quote_at")
    ) and all(_true(row, field) for field in (
        "execution_ready", "market_execution_verified",
        "market_quote_timestamp_verified", "market_quote_sanity_ok",
    ))
    quote_label = "VERIFIED_QUOTE_RESEARCH_ONLY" if quote else "NO_VERIFIED_QUOTE"
    reasons = []
    if not quote:
        reasons.append("QUOTE_NOT_EXECUTION_VERIFIED")
    if not _true(row, "context_injuries_personnel_fresh") or _true(row, "context_freshness_veto"):
        reasons.append("STALE_OR_UNKNOWN_INJURY_CONTEXT")
    if not _true(row, "qb_context_ready") or _true(row, "qb_certainty_veto"):
        reasons.append("QB_STARTER_STATE_UNCERTAIN")
    if _true(row, "context_veto"):
        reasons.append("CONTEXT_VETO")
    if not _true(row, "probability_reliability_ready") or _true(
        row, "probability_reliability_veto"
    ):
        reasons.append("PROBABILITY_UNRELIABLE")
    if not _true(row, "regime_reliability_ready") or str(
        row.get("regime_reliability_status") or ""
    ).upper() != "RELIABLE":
        reasons.append("HISTORICAL_REGIME_NOT_VALIDATED")
    if str(row.get("edge_market_shrinkage_status") or "") == "MARKET_ONLY_PREFERRED":
        reasons.append("HOLDOUT_PREFERS_MARKET")
    if not _positive(row.get("edge_shrunk_ev")):
        reasons.append("NO_POSITIVE_HOLDOUT_SHRUNK_EV")
    if row.get("edge_discovery_tier") != "SUPPORTED_RESEARCH":
        reasons.append("EDGE_EVIDENCE_NOT_SUPPORTED")
    if _true(row, "market_dispersion_high") or str(
        row.get("market_disagreement_severity") or ""
    ).upper() == "HIGH":
        reasons.append("UNUSUAL_MARKET_DISAGREEMENT")
    if row.get("recommendation_status") != "ACTIVE":
        reasons.append("RECOMMENDATION_NOT_ACTIVE")

    if reasons:
        return ("PASS", quote_label, ";".join(dict.fromkeys(reasons)), None)
    # Conservative EV only shown when every independent research check passes.
    # Positive shrunk EV is not proof of profitability and cannot place a wager.
    return (
        "REVIEW_ONLY", quote_label,
        "ALL_RESEARCH_GATES_PASSED_NOT_AN_EXECUTED_BET",
        float(row["edge_shrunk_ev"]),
    )

def build_actionable_board(current: pl.DataFrame) -> pl.DataFrame:
    """Preserve every projection; suppress misleading unsupported raw EV picks."""
    require_columns(
        current,
        {"season", "week", "game_id", "home_team", "away_team",
         "model_margin_home", "model_total"},
        "nfl_current_actionable_board",
    )
    data = current
    for name in (*COLUMNS, *EVIDENCE):
        if name not in data.columns:
            data = data.with_columns(pl.lit(None).alias(name))

    assessments = [_evaluate(row) for row in data.iter_rows(named=True)]
    return (
        data.with_columns(
            ((pl.col("model_total") + pl.col("model_margin_home")) / 2)
            .alias("predicted_home_score"),
            ((pl.col("model_total") - pl.col("model_margin_home")) / 2)
            .alias("predicted_away_score"),
            pl.when(pl.col("model_margin_home") > 0)
            .then(pl.col("home_team"))
            .when(pl.col("model_margin_home") < 0)
            .then(pl.col("away_team"))
            .otherwise(pl.lit("TIE_PROJECTION"))
            .alias("projected_winner"),
            pl.Series("betting_action", [row[0] for row in assessments], dtype=pl.String),
            pl.Series("quote_quality", [row[1] for row in assessments], dtype=pl.String),
            pl.Series("evidence_blockers", [row[2] for row in assessments], dtype=pl.String),
            pl.Series(
                "conservative_ev", [row[3] for row in assessments], dtype=pl.Float64
            ),
            pl.Series(
                "evidence_status",
                ["SUPPORTED_RESEARCH_NOT_PRODUCTION" if row[0] == "REVIEW_ONLY"
                 else "BLOCKED_OR_MISSING" for row in assessments], dtype=pl.String,
            ),
        )
        .select(*COLUMNS, *EVIDENCE, *EXTRA)
        .sort("season", "week", "game_id")
    )

def write_actionable_board(
    current: pl.DataFrame, path: str = "outputs/actionable_betting_board.csv"
) -> pl.DataFrame:
    from pathlib import Path
    board = build_actionable_board(current)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    board.write_csv(destination)
    return board
