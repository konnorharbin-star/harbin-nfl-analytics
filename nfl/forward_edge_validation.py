"""NFL forward market-edge evaluation on immutable first pre-kickoff candidates.

Never recompute model probabilities after kickoff. Never use closing quotes to
select a cohort, reprice a changed spread or total, or promote a betting policy.
This is not evidence of actual fills or real-money sportsbook profit.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from .book_identity import canonical_book_identity
from .contracts import require_columns
from .grading import _grade_value
from .market import american_implied_probability, american_to_decimal

LEDGER_VERSION = 1
SPEC_VERSION = "nfl_frozen_market_edge_forward_v1"
SEASON = 2026
MARKETS = ("moneyline", "spread", "total")
MIN_BOOTSTRAP_ROWS = 100
MIN_BOOTSTRAP_GAMES = 80
MIN_BOOTSTRAP_WEEKS = 8
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20261008
MAX_QUOTE_AGE_MINUTES = 30
CLOSE_WINDOW_MINUTES = 90
EPS = 1e-6

LEDGER_FIELDS = (
    "ledger_version", "spec_version", "captured_at", "kickoff", "season",
    "week", "game_id", "home_team", "away_team",
    "market", "side", "book", "line", "odds", "quote_at",
    "model_probability", "no_vig_probability", "raw_edge", "raw_ev",
    "source_verified", "source_timestamp_verified", "quote_sanity_ok",
    "execution_ready_at_capture", "release_action_at_capture",
    "portfolio_stake_units_at_capture", "edge_discovery_tier_at_capture",
    "edge_timing_action_at_capture", "model_family_at_capture",
    "snapshot_status",
)

GRADED_FIELDS = (
    *LEDGER_FIELDS,
    "observation_status", "home_score", "away_score", "result",
    "simulated_net_units", "model_brier", "market_brier",
    "model_log_loss", "market_log_loss", "realized_minus_market_probability",
    "raw_ev_realization_gap", "near_kickoff_status",
    "near_kickoff_captured_at", "near_kickoff_line", "near_kickoff_odds",
    "near_kickoff_line_change", "near_kickoff_price_change_pp",
)


def _parse(value: object) -> datetime | None:
    if isinstance(value, datetime):
        dt = value
    elif value is None or value == "":
        return None
    else:
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    return dt.astimezone(UTC) if dt.tzinfo is not None else None


def _num(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _truth(value: object) -> bool:
    return value is True or str(value or "").strip().lower() == "true"


def _odds(value: object) -> int | None:
    number = _num(value)
    if number is None or int(number) != number:
        return None
    integer = int(number)
    return integer if integer >= 100 or integer <= -100 else None


def _valid_market(market: str, side: str, line: object, odds: object) -> bool:
    if _odds(odds) is None:
        return False
    if market == "moneyline":
        return side in {"home", "away"} and line in {None, ""}
    if market == "spread":
        return side in {"home", "away"} and _num(line) is not None
    if market == "total":
        return side in {"over", "under"} and _num(line) is not None
    return False


def _quote_status(row: Mapping[str, object]) -> str:
    captured = _parse(row.get("captured_at"))
    kickoff = _parse(row.get("kickoff"))
    quote = _parse(row.get("quote_at"))
    if captured is None or kickoff is None or captured >= kickoff:
        return "INVALID_CAPTURE_TIME"
    if not (
        _truth(row.get("source_verified"))
        and _truth(row.get("source_timestamp_verified"))
        and _truth(row.get("quote_sanity_ok"))
    ):
        return "UNVERIFIED_BOOK_OR_MARKET_PRICE"
    if not canonical_book_identity(row.get("book")):
        return "MISSING_BOOK"
    if not _valid_market(
        str(row.get("market") or ""), str(row.get("side") or ""),
        row.get("line"), row.get("odds"),
    ):
        return "INVALID_OFFER"
    if (
        quote is None or quote > captured or quote >= kickoff
        or (captured - quote).total_seconds() < 0
        or (captured - quote) > timedelta(minutes=MAX_QUOTE_AGE_MINUTES)
    ):
        return "STALE_FUTURE_OR_UNTIMED_OFFER"
    p = _num(row.get("model_probability"))
    m = _num(row.get("no_vig_probability"))
    edge = _num(row.get("raw_edge"))
    ev = _num(row.get("raw_ev"))
    odds = _odds(row.get("odds"))
    if (
        p is None or m is None or edge is None or ev is None or odds is None
        or not (0 < p < 1 and 0 < m < 1)
    ):
        return "INVALID_FROZEN_PROBABILITY"
    if (
        abs((p - m) - edge) > 1e-5
        or abs((p * american_to_decimal(odds) - 1) - ev) > 1e-5
    ):
        return "UNRECONCILED_PROBABILITY_OR_EV"
    return "VALID_POINT_IN_TIME_RESEARCH_QUOTE"


def _as_ledger(row: Mapping[str, object], captured: datetime):
    values = {
        "ledger_version": LEDGER_VERSION, "spec_version": SPEC_VERSION,
        "captured_at": captured.isoformat(),
        "kickoff": row.get("kickoff"),
        "season": row.get("season"), "week": row.get("week"),
        "game_id": row.get("game_id"),
        "home_team": row.get("home_team"), "away_team": row.get("away_team"),
        "market": row.get("quant_market"), "side": row.get("quant_side"),
        "book": row.get("quant_book"), "line": row.get("quant_price"),
        "odds": row.get("quant_odds"), "quote_at": row.get("quant_quote_at"),
        "model_probability": row.get("quant_probability"),
        "no_vig_probability": row.get("quant_market_probability"),
        "raw_edge": row.get("quant_edge"), "raw_ev": row.get("quant_ev"),
        "source_verified": row.get("market_execution_verified"),
        "source_timestamp_verified": row.get("market_quote_timestamp_verified"),
        "quote_sanity_ok": row.get("market_quote_sanity_ok"),
        "execution_ready_at_capture": row.get("execution_ready"),
        "release_action_at_capture": row.get("portfolio_action"),
        "portfolio_stake_units_at_capture": row.get("portfolio_stake_units"),
        "edge_discovery_tier_at_capture": row.get("edge_discovery_tier"),
        "edge_timing_action_at_capture": row.get("edge_timing_action"),
        "model_family_at_capture": row.get("probability_model_family"),
    }
    values["snapshot_status"] = _quote_status(values)
    return values


def load_forward_candidates(
    path: str | Path = "history/edge_forward_candidates_v1.csv",
) -> pl.DataFrame:
    source = Path(path)
    if not source.exists() or not source.stat().st_size:
        return pl.DataFrame()
    return pl.read_csv(source, try_parse_dates=False, infer_schema_length=2000)


def append_forward_candidates(
    current: pl.DataFrame,
    *,
    captured_at: datetime | None = None,
    path: str | Path = "history/edge_forward_candidates_v1.csv",
) -> dict[str, object]:
    """Write exactly the first observed candidate per 2026 game+market.

    First ineligible/blocked candidate is retained, not replaced by a later
    price that happens to look better. Existing entries are never updated.
    """
    now = captured_at or datetime.now(UTC)
    if now.tzinfo is None:
        raise ValueError("forward capture requires timezone-aware timestamp")
    now = now.astimezone(UTC)
    required = {"season", "week", "game_id", "kickoff", "quant_market"}
    if not current.is_empty():
        require_columns(current, required, "edge_forward_current_candidates")
    dest = Path(path)
    old = []
    if dest.exists() and dest.stat().st_size:
        with dest.open(newline="", encoding="utf-8") as handle:
            old = list(csv.DictReader(handle))
    known: set[tuple[str, str]] = set()
    for row in old:
        key = (str(row.get("game_id")), str(row.get("market")))
        if key in known:
            raise ValueError("duplicate frozen forward game-market key: " + str(key))
        known.add(key)
    appended = []
    counts = Counter()
    # Deterministic within-run tie-breaking: no outcome or future observations.
    for source in sorted(
        current.to_dicts(),
        key=lambda row: (str(row.get("game_id")), str(row.get("quant_market"))),
    ):
        if int(source["season"]) != SEASON:
            counts["wrong_season"] += 1
            continue
        kickoff = _parse(source.get("kickoff"))
        if kickoff is None or now >= kickoff:
            counts["post_kickoff_or_invalid_kickoff"] += 1
            continue
        game = str(source.get("game_id") or "")
        market = str(source.get("quant_market") or "")
        if not game or market not in MARKETS:
            counts["invalid_game_market"] += 1
            continue
        key = (game, market)
        if key in known:
            counts["previously_frozen"] += 1
            continue
        row = _as_ledger(source, now)
        appended.append(row)
        known.add(key)
        counts[row["snapshot_status"]] += 1
    if appended:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(LEDGER_FIELDS))
            writer.writeheader()
            writer.writerows(old + appended)
    return {
        "status": "APPEND_ONLY_FIRST_SNAPSHOT",
        "schema_version": LEDGER_VERSION, "appended_rows": len(appended),
        "total_rows": len(old) + len(appended),
        "capture_status_counts": dict(sorted(counts.items())),
        "research_only": True, "bets_authorized": False,
        "historical_backfill_allowed": False,
    }


def _close_observation(
    entry: Mapping[str, object], snapshots: pl.DataFrame,
):
    """Return last same-book observation in final 90 minutes, never 'close'."""
    if snapshots.is_empty():
        return {}
    required = {
        "game_id", "market_type", "book", "captured_at", "kickoff",
        "first_side", "first_line", "first_american_odds",
        "second_side", "second_line", "second_american_odds",
    }
    if not required.issubset(snapshots.columns):
        return {}
    decision = _parse(entry.get("captured_at"))
    kickoff = _parse(entry.get("kickoff"))
    book = canonical_book_identity(entry.get("book"))
    if decision is None or kickoff is None or not book:
        return {}
    earliest = kickoff - timedelta(minutes=CLOSE_WINDOW_MINUTES)
    selected = None
    for row in snapshots.iter_rows(named=True):
        if (
            str(row.get("game_id")) != str(entry.get("game_id"))
            or str(row.get("market_type")) != str(entry.get("market"))
            or canonical_book_identity(row.get("book")) != book
        ):
            continue
        when = _parse(row.get("captured_at"))
        row_kickoff = _parse(row.get("kickoff"))
        if (
            when is None or row_kickoff is None or row_kickoff != kickoff
            or not (decision < when < kickoff) or when < earliest
        ):
            continue
        for prefix in ("first", "second"):
            if str(row.get(f"{prefix}_side")) != str(entry.get("side")):
                continue
            line = row.get(f"{prefix}_line")
            odds = _odds(row.get(f"{prefix}_american_odds"))
            if not _valid_market(
                str(entry.get("market")), str(entry.get("side")), line, odds
            ):
                continue
            if selected is None or when > selected[0]:
                selected = when, _num(line), odds
    if selected is None:
        return {}
    when, new_line, new_odds = selected
    initial_line = _num(entry.get("line"))
    old_odds = _odds(entry.get("odds"))
    if old_odds is None:
        return {}
    if str(entry.get("market")) != "moneyline" and new_line != initial_line:
        status = "HANDICAP_CHANGED_PRICE_UTILITY_UNPRICED"
    else:
        status = "SAME_HANDICAP_PRICE_OBSERVED"
    movement = None
    market = str(entry.get("market"))
    side = str(entry.get("side"))
    if new_line is not None and initial_line is not None:
        if market == "spread":
            movement = initial_line - new_line
        elif market == "total":
            movement = new_line - initial_line if side == "over" else initial_line - new_line
    return {
        "near_kickoff_status": status,
        "near_kickoff_captured_at": when.isoformat(),
        "near_kickoff_line": new_line,
        "near_kickoff_odds": new_odds,
        "near_kickoff_line_change": movement,
        "near_kickoff_price_change_pp": (
            (american_implied_probability(new_odds)
             - american_implied_probability(old_odds)) * 100
            if status == "SAME_HANDICAP_PRICE_OBSERVED" else None
        ),
    }


def _scored(entry: Mapping[str, object], final: Mapping[str, object]) -> dict[str, object]:
    scored = dict(entry)
    scored.update({field: None for field in GRADED_FIELDS if field not in scored})
    scored["observation_status"] = "PENDING_RESULT"
    scored["near_kickoff_status"] = "NOT_OBSERVED"
    home = _num(final.get("home_score"))
    away = _num(final.get("away_score"))
    if home is None or away is None:
        return scored
    scored["home_score"], scored["away_score"] = home, away
    if str(final.get("home_team")) != str(entry.get("home_team")) or str(
        final.get("away_team")
    ) != str(entry.get("away_team")):
        scored["observation_status"] = "TEAM_IDENTITY_MISMATCH"
        return scored
    if str(final.get("season")) != str(entry.get("season")):
        scored["observation_status"] = "SEASON_MISMATCH"
        return scored
    if str(entry.get("snapshot_status")) != "VALID_POINT_IN_TIME_RESEARCH_QUOTE":
        scored["observation_status"] = "INVALID_FROZEN_QUOTE"
        return scored
    if _quote_status(entry) != "VALID_POINT_IN_TIME_RESEARCH_QUOTE":
        scored["observation_status"] = "INVALID_FROZEN_QUOTE"
        return scored
    market = str(entry.get("market"))
    side = str(entry.get("side"))
    line = _num(entry.get("line"))
    odds = _odds(entry.get("odds"))
    assert odds is not None
    margin, total = home - away, home + away
    value = _grade_value(market, side, line, margin, total)
    result = "win" if value > 1e-12 else "loss" if value < -1e-12 else "push"
    scored["result"] = result
    scored["simulated_net_units"] = (
        american_to_decimal(odds) - 1 if result == "win"
        else -1.0 if result == "loss" else 0.0
    )
    if result == "push":
        scored["observation_status"] = "GRADED_PUSH_UNSCORED_PROBABILITY"
    else:
        actual = 1.0 if result == "win" else 0.0
        p = float(entry["model_probability"])
        m = float(entry["no_vig_probability"])
        scored["model_brier"] = (p - actual) ** 2
        scored["market_brier"] = (m - actual) ** 2
        scored["model_log_loss"] = -math.log(max(EPS, p if actual else 1 - p))
        scored["market_log_loss"] = -math.log(max(EPS, m if actual else 1 - m))
        scored["realized_minus_market_probability"] = actual - m
        scored["observation_status"] = "GRADED_PROBABILITY"
    scored["raw_ev_realization_gap"] = (
        float(entry["raw_ev"]) - scored["simulated_net_units"]
    )
    return scored


def _ci(rows: list[dict[str, object]], key: str):
    values = [r for r in rows if _num(r.get(key)) is not None]
    if (
        len(values) < MIN_BOOTSTRAP_ROWS
        or len({r["game_id"] for r in values}) < MIN_BOOTSTRAP_GAMES
        or len({(r["season"], r["week"]) for r in values}) < MIN_BOOTSTRAP_WEEKS
    ):
        return None
    blocks = defaultdict(list)
    for row in values:
        blocks[(row["season"], row["week"])].append(float(row[key]))
    summary = [(sum(v), len(v)) for _, v in sorted(blocks.items())]
    sums = np.asarray([x[0] for x in summary], dtype=float)
    sizes = np.asarray([x[1] for x in summary], dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    picks = rng.integers(0, len(summary), size=(BOOTSTRAP_REPS, len(summary)))
    means = sums[picks].sum(axis=1) / sizes[picks].sum(axis=1)
    return [float(z) for z in np.quantile(means, [0.025, 0.975])]


def _summary(rows: list[dict[str, object]]):
    decided = [r for r in rows if r["observation_status"] == "GRADED_PROBABILITY"]
    graded = [r for r in rows if str(r["observation_status"]).startswith("GRADED_")]
    for row in decided:
        row["_paired_brier_lift"] = row["market_brier"] - row["model_brier"]
        row["_paired_log_lift"] = row["market_log_loss"] - row["model_log_loss"]
    results = {
        "frozen": len(rows), "graded": len(graded),
        "decided": len(decided),
        "distinct_games": len({r["game_id"] for r in decided}),
        "distinct_kickoff_weeks": len({(r["season"], r["week"]) for r in decided}),
        "pushes": sum(r["result"] == "push" for r in graded),
        "mean_model_brier": (
            float(np.mean([r["model_brier"] for r in decided])) if decided else None
        ),
        "mean_market_brier": (
            float(np.mean([r["market_brier"] for r in decided])) if decided else None
        ),
        "mean_model_log_loss": (
            float(np.mean([r["model_log_loss"] for r in decided])) if decided else None
        ),
        "mean_market_log_loss": (
            float(np.mean([r["market_log_loss"] for r in decided])) if decided else None
        ),
        "mean_raw_ev": (
            float(np.mean([r["raw_ev"] for r in graded])) if graded else None
        ),
        "hypothetical_archive_free_quote_roi": (
            float(np.mean([r["simulated_net_units"] for r in graded]))
            if graded else None
        ),
        "mean_realized_minus_market_probability": (
            float(np.mean([r["realized_minus_market_probability"] for r in decided]))
            if decided else None
        ),
        "mean_raw_ev_realization_gap": (
            float(np.mean([r["raw_ev_realization_gap"] for r in graded]))
            if graded else None
        ),
        "market_relative_brier_lift_95_ci": _ci(decided, "_paired_brier_lift"),
        "market_relative_log_loss_lift_95_ci": _ci(decided, "_paired_log_lift"),
        "hypothetical_roi_95_ci": _ci(graded, "simulated_net_units"),
        "quote_status": dict(sorted(Counter(r["snapshot_status"] for r in rows).items())),
        "grading_status": dict(
            sorted(Counter(r["observation_status"] for r in rows).items())
        ),
        "near_kickoff_same_line": sum(
            r.get("near_kickoff_status") == "SAME_HANDICAP_PRICE_OBSERVED"
            for r in graded
        ),
        "near_kickoff_changed_handicap": sum(
            r.get("near_kickoff_status") == "HANDICAP_CHANGED_PRICE_UTILITY_UNPRICED"
            for r in graded
        ),
    }
    results["brier_lift_vs_market"] = (
        results["mean_market_brier"] - results["mean_model_brier"]
        if decided else None
    )
    results["log_loss_lift_vs_market"] = (
        results["mean_market_log_loss"] - results["mean_model_log_loss"]
        if decided else None
    )
    return results


def grade_forward_candidates(
    ledger: pl.DataFrame,
    schedules: pl.DataFrame,
    *,
    snapshots: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, dict[str, object]]:
    """Grade only persisted, pre-kickoff 2026 candidates; never backfill."""
    report = {
        "schema_version": LEDGER_VERSION, "spec_version": SPEC_VERSION,
        "status": "PENDING_FORWARD", "research_only": True,
        "staking_authorized": False, "production_model_change_enabled": False,
        "forward_results_used_to_refit": False,
        "frozen_candidate_rows": 0, "by_market": {},
        "summary": _summary([]),
        "minimum_paired_cases_for_uncertainty": MIN_BOOTSTRAP_ROWS,
        "minimum_distinct_games": MIN_BOOTSTRAP_GAMES,
        "minimum_kickoff_weeks": MIN_BOOTSTRAP_WEEKS,
        "price_provenance": (
            "Observed sportsbook feed capture; source assertion only, not a guaranteed "
            "executable fill or independent confirmation of sportsbook publication"
        ),
        "late_quote_provenance": (
            "Last observed same-book quote in final 90 minutes; not official close"
        ),
        "side_selection_bias": "max raw EV per game/market before result; no causal lift",
        "evidence_required": (
            "new independent kickoff weeks and same-book frozen quotes; "
            "no policy promotion on forward cases without separate review"
        ),
    }
    if ledger.is_empty():
        return pl.DataFrame(), report
    require_columns(ledger, set(LEDGER_FIELDS), "frozen_edge_forward_ledger")
    require_columns(
        schedules, {"game_id", "season", "home_team", "away_team",
                    "home_score", "away_score"}, "nfl_forward_final_scores",
    )
    history = snapshots if snapshots is not None else pl.DataFrame()
    seen = set()
    finals = {}
    for final in schedules.iter_rows(named=True):
        if (
            _num(final.get("home_score")) is not None
            and _num(final.get("away_score")) is not None
        ):
            finals[str(final["game_id"])] = final
    rows = []
    duplicates = 0
    for entry in ledger.iter_rows(named=True):
        key = (str(entry.get("game_id")), str(entry.get("market")))
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        if str(entry.get("spec_version")) != SPEC_VERSION or str(
            entry.get("season")
        ) != str(SEASON):
            row = dict(entry)
            row.update({field: None for field in GRADED_FIELDS if field not in row})
            row["observation_status"] = "INVALID_SPEC_OR_SEASON"
        elif _parse(entry.get("captured_at")) is None or _parse(
            entry.get("kickoff")
        ) is None or _parse(entry.get("captured_at")) >= _parse(entry.get("kickoff")):
            row = dict(entry)
            row.update({field: None for field in GRADED_FIELDS if field not in row})
            row["observation_status"] = "INVALID_FROZEN_TIME"
        else:
            final = finals.get(str(entry.get("game_id")))
            if final is None:
                row = dict(entry)
                row.update({field: None for field in GRADED_FIELDS if field not in row})
                row["observation_status"] = "PENDING_RESULT"
            else:
                row = _scored(entry, final)
                if row["observation_status"].startswith("GRADED_"):
                    row.update(_close_observation(entry, history))
        rows.append(row)
    grouped = {}
    for market in MARKETS:
        grouped[market] = _summary([row for row in rows if row.get("market") == market])
    overall = _summary(rows)
    mature = overall["decided"]
    report.update({
        "status": (
            "INVALID_LEDGER_DUPLICATES_FAIL_CLOSED" if duplicates
            else "PENDING_FORWARD" if mature < MIN_BOOTSTRAP_ROWS
            else "RESEARCH_ONLY_SAMPLE_MATURED_NOT_APPROVED"
        ),
        "frozen_candidate_rows": ledger.height, "first_unique_rows": len(rows),
        "duplicate_frozen_keys": duplicates,
        "summary": overall,
        "by_market": grouped,
        "promotion_eligible": False,
        "heldout_2025_used_for_threshold_tuning": False,
    })
    return pl.DataFrame(rows), report


def write_forward_validation(
    graded: pl.DataFrame,
    report: Mapping[str, object],
    *,
    report_dir: str | Path = "reports",
    docs_dir: str | Path = "docs",
    outputs_dir: str | Path = "outputs",
):
    rows = graded.to_dicts() if not graded.is_empty() else []
    for root in (Path(report_dir), Path(docs_dir), Path(outputs_dir)):
        root.mkdir(parents=True, exist_ok=True)
        (root / "forward_edge_validation.json").write_text(
            json.dumps(dict(report), indent=2, sort_keys=True, default=str)
        )
        with (root / "forward_edge_graded.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(GRADED_FIELDS), extrasaction="ignore"
            )
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field) for field in GRADED_FIELDS})
