"""Research-only, bookmaker-neutral NFL pregame matchup preference board.

This reads original first-seen, price-free Elo and QB-proxy forecasts. It
does not refit predictions, use sportsbook prices, place wagers, or certify
calibration/profit. Predictions and bettable market edges are distinct.
"""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

ELO_PATH = Path("history/moneyline_elo_v1_2026.jsonl")
QB_PATH = Path("history/moneyline_weekly_qb_challenger_v2_2026.jsonl")
WINDOW = timedelta(days=10)


def _clock(value: str) -> datetime:
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("Timestamp must have a timezone")
    return dt.astimezone(UTC)


def _prob(row: dict, key: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Missing numeric probability: " + key)
    out = float(value)
    if not (0 <= out <= 1):
        raise ValueError("Probability outside [0,1]: " + key)
    return out


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def build_board(elo_rows: list[dict], qb_rows: list[dict], *, now: datetime) -> dict:
    """Build a research preference for every upcoming matched, original game.

    No outcome is loaded. Model agreement is descriptive, NOT independent
    evidence of accuracy: the two football models may share source inputs.
    """
    if now.tzinfo is None:
        raise ValueError("The as-of reference must be timezone-aware")
    at = now.astimezone(UTC)
    qbs = {r.get("game_id"): r for r in qb_rows}
    picks = []
    rejected = []
    seen = set()
    for e in elo_rows:
        game_id = str(e.get("game_id") or "")
        if not game_id or game_id in seen:
            rejected.append({"game_id": game_id, "reason": "DUPLICATE_OR_EMPTY_ELO_EVENT"})
            continue
        seen.add(game_id)
        q = qbs.get(game_id)
        if q is None:
            rejected.append({"game_id": game_id, "reason": "MISSING_QB_RESEARCH_FORECAST"})
            continue
        try:
            kickoff = _clock(e["kickoff_utc"])
            t_elo = _clock(e["first_observed_at_utc"])
            t_qb = _clock(q["first_observed_at_utc"])
        except (KeyError, ValueError, TypeError):
            rejected.append({"game_id": game_id, "reason": "TIMESTAMP_MISSING_INVALID"})
            continue
        if not (at < kickoff <= at + WINDOW):
            continue
        if not (t_elo <= at and t_qb <= at and t_elo < kickoff and t_qb < kickoff):
            rejected.append({"game_id": game_id, "reason": "NOT_AUTHENTIC_PREKICKOFF_FORECAST"})
            continue
        if (
            e.get("home_team") != q.get("home") or
            e.get("away_team") != q.get("away") or
            e.get("kickoff_utc") != q.get("kickoff_utc") or
            e.get("season") != q.get("season") or
            e.get("week") != q.get("week")
        ):
            rejected.append({"game_id": game_id, "reason": "FORECAST_IDENTITY_MISMATCH"})
            continue
        if e.get("bookmaker_account_acceptance") is not False or q.get("wager") is not False:
            rejected.append({"game_id": game_id, "reason": "FORECAST_CONTAINS_EXECUTION_CLAIM"})
            continue
        try:
            ph_e = _prob(e, "p_home_win")
            pa_e = _prob(e, "p_away_win")
            pt_e = _prob(e, "p_tie_push")
            ph_q = _prob(q, "p_home_win")
            pa_q = _prob(q, "p_away_win")
            pt_q = _prob(q, "p_tie_push")
        except ValueError:
            rejected.append({"game_id": game_id, "reason": "INVALID_FORECAST_PROBABILITIES"})
            continue
        if abs(ph_e + pa_e + pt_e - 1) > 1e-5 or abs(ph_q + pa_q + pt_q - 1) > 1e-5:
            rejected.append({"game_id": game_id, "reason": "FORECAST_PROBABILITIES_DO_NOT_SUM_TO_ONE"})
            continue
        side_e = "home" if ph_e > pa_e else "away" if pa_e > ph_e else None
        side_q = "home" if ph_q > pa_q else "away" if pa_q > ph_q else None
        agreement = side_e is not None and side_e == side_q
        selected = (e["home_team"] if side_e == "home" else e["away_team"]) if agreement else None
        e_selected = ph_e if side_e == "home" else pa_e
        q_selected = ph_q if side_q == "home" else pa_q
        picks.append({
            "game_id": game_id, "season": e["season"], "week": e["week"],
            "kickoff_utc": kickoff.isoformat(),
            "away_team": e["away_team"], "home_team": e["home_team"],
            "elo_home_win": round(ph_e, 6), "qb_proxy_home_win": round(ph_q, 6),
            "elo_away_win": round(pa_e, 6), "qb_proxy_away_win": round(pa_q, 6),
            "elo_pick": e["home_team"] if side_e == "home" else e["away_team"] if side_e == "away" else None,
            "qb_proxy_pick": q["home"] if side_q == "home" else q["away"] if side_q == "away" else None,
            "consensus_pick": selected,
            "descriptive_min_directional_probability": round(min(e_selected, q_selected), 6) if agreement else None,
            "research_tier": "AGREEMENT_PREFERENCE_UNVALIDATED" if agreement else "DISAGREEMENT_NO_DIRECTIONAL_PREFERENCE",
            "elo_observed_utc": t_elo.isoformat(),
            "qb_observed_utc": t_qb.isoformat(),
            "qb_proxy_status": q.get("QB_proxy_status", "UNKNOWN"),
            "prediction_independent_of_sportsbook_prices": True,
            "forecast_validated_profitable": False,
            "p_win_as_bet": None, "p_push_as_bet": None,
            "p_loss_as_bet": None, "p_void_as_bet": None,
            "market_quote": None, "estimated_ev": None, "wager_authorized": False,
            "action": "PASS_RESEARCH_ONLY",
            "missing_evidence": [
                "VALIDATED_CHRONOLOGICAL_PREDICTIVE_ACCURACY",
                "EXPECTED_STARTING_QB_AND_INJURY_RECONCILIATION",
                "MATCHED_MARKET_LINE_AND_SPORTSBOOK_PRICE",
                "INDEPENDENT_FORWARD_MARKET_RELATIVE_ADVANTAGE",
            ],
        })
    picks.sort(key=lambda r: (
        r["consensus_pick"] is None,
        -(r["descriptive_min_directional_probability"] or 0),
        r["game_id"],
    ))
    return {
        "spec": "NFL_BOOK_NEUTRAL_PREMATCH_DIRECTION_V1",
        "as_of_utc": at.isoformat(),
        "status": "DIRECTIONAL_RESEARCH_ONLY_NO_BETS",
        "forecast_is_frozen_source_only": True,
        "this_report_is_not_a_first_seen_forecast": True,
        "independent_model_validation_claimed": False,
        "profitability_or_edge_claimed": False,
        "all_books_optional_for_direction": True,
        "no_price_is_an_approved_bet": True,
        "games": picks,
        "rejected": rejected,
        "top_directional_research_game": next((p["game_id"] for p in picks if p["consensus_pick"]), None),
        "approved_bets": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elo", type=Path, default=ELO_PATH)
    parser.add_argument("--qb", type=Path, default=QB_PATH)
    parser.add_argument("--out", type=Path, default=Path("reports/book_neutral_direction_board.json"))
    parser.add_argument("--as-of", help="UTC observation time; omit for now")
    args = parser.parse_args()
    now = _clock(args.as_of) if args.as_of else datetime.now(UTC)
    report = build_board(read_jsonl(args.elo), read_jsonl(args.qb), now=now)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "games": len(report["games"]),
        "preferred_matchup": report["top_directional_research_game"],
        "approved_bets": 0,
        "output": str(args.out),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
