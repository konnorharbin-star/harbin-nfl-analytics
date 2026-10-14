"""Actual-line receiving-yards diagnostics; never certifies an executable edge."""

from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np

from nfl.prop_probability import over_probability
from scripts.nfl_receiving_usage_edge_test import examples, fit_errors, vector
from scripts.usage_prop_research import _usage_forecast

MARKET = "receiving_yards"
WINDOW = 8  # Already selected on 2024; do not optimize against priced outcomes.


def decimal_odds(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("missing/non-numeric odds")
    if not math.isfinite(value) or abs(value) < 100:
        raise ValueError("invalid American odds")
    return 1 + (value / 100 if value > 0 else 100 / abs(value))


def import_histories(payload, *, bookmaker="draftkings"):
    """Accept provider responses, requiring one named book and actual outcomes.

    Source-reported historical close is a diagnostic proxy. No bookmaker origin
    timestamp, entry availability or settlement rule is inferred from this API.
    """
    if not isinstance(payload, dict):
        raise ValueError("history must be an object")
    responses = payload.get("responses", [payload])
    if not isinstance(responses, list):
        raise ValueError("responses must be a list")
    quotes, audit, seen, duplicate_keys = [], Counter(), set(), set()
    for response in responses:
        if not isinstance(response, dict):
            raise ValueError("response must be an object")
        data = response.get("data", response)
        if not isinstance(data, dict) or not isinstance(data.get("games", []), list):
            raise ValueError("malformed provider history")
        if data.get("market") != "player_reception_yds" or data.get("bookmaker") != bookmaker:
            audit["wrong_market_or_missing_single_book"] += 1
            continue
        if data.get("truncated"):
            audit["truncated_player_history"] += 1
        player_id = data.get("player_id")
        if not isinstance(player_id, str) or not player_id.startswith("00-"):
            audit["invalid_player_id"] += 1
            continue
        for quote in data.get("games", []):
            try:
                gid = quote["game_id"]
                if not isinstance(gid, str) or not gid.startswith("2025_"):
                    audit["outside_frozen_diagnostic_year"] += 1
                    continue
                line, actual = quote["line"], quote["actual"]
                if any(
                    isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                    for v in (line, actual)
                ):
                    raise ValueError("missing/nonfinite line or outcome")
                if line < 0 or line % 1 != 0.5 or actual < 0 or actual % 1:
                    raise ValueError("unsupported integer-line push or invalid outcome")
                over = decimal_odds(quote["over_odds"])
                under = decimal_odds(quote["under_odds"])
                key = (gid, player_id)
                if key in seen:
                    duplicate_keys.add(key)
                    raise ValueError("duplicate contract")
                seen.add(key)
                quotes.append(
                    {
                        "game_id": gid,
                        "player_id": player_id,
                        "line": line,
                        "actual": actual,
                        "book": bookmaker,
                        "over_decimal": over,
                        "under_decimal": under,
                        "market_over_proxy": (1 / over) / (1 / over + 1 / under),
                        "quote_origin_utc": None,
                        "line_stage": "PROVIDER_REPORTED_HISTORICAL_CLOSE",
                    }
                )
            except (KeyError, ValueError, TypeError):
                audit["invalid_or_duplicate_contract"] += 1
    quotes = [q for q in quotes if (q["game_id"], q["player_id"]) not in duplicate_keys]
    audit["previously_accepted_duplicate_contracts_removed"] = len(duplicate_keys)
    return quotes, dict(audit)


def fit_fixed_candidate(games, stats):
    histories = defaultdict(list)
    for row in stats:
        histories[(row["season"], row["team"], row["player_id"])].append(row)
    training, unknown = examples(games, stats, 2024)
    train = []
    for row, actual in training:
        if row["market"] != MARKET:
            continue
        usage = _usage_forecast(row, histories, WINDOW)
        if usage is not None:
            train.append((dict(row, usage_mean=usage), actual))
    if len(train) < 160:
        raise ValueError("insufficient 2024 training")
    base_errors = fit_errors(train, "challenger_mean")
    usage_errors = fit_errors(train, "usage_mean")
    testing, missing_test = examples(games, stats, 2025)
    forecasts = {}
    for row, actual in testing:
        if row["market"] != MARKET:
            continue
        usage = _usage_forecast(row, histories, WINDOW)
        if usage is not None:
            forecasts[(row["game_id"], row["player_id"])] = {
                **row,
                "actual": actual,
                "usage_mean": usage,
                "base_errors": vector(base_errors, row),
                "usage_errors": vector(usage_errors, row),
            }
    return forecasts, {
        "training_year": 2024,
        "evaluation_year": 2025,
        "usage_window_fixed": WINDOW,
        "training_groups": len(train),
        "missing_training_outcomes": unknown,
        "missing_diagnostic_outcomes": missing_test,
        "evaluation_year_previously_inspected": True,
    }


def score_quotes(forecasts, quotes, audit=None):
    counts = Counter(audit or {})
    records = []
    for quote in quotes:
        row = forecasts.get((quote["game_id"], quote["player_id"]))
        if row is None:
            counts["no_matching_past_only_forecast"] += 1
            continue
        if row["actual"] != quote["actual"]:
            counts["provider_nflverse_outcome_conflict"] += 1
            continue
        line = quote["line"]
        pb = over_probability(row["challenger_mean"], line, MARKET, row["base_errors"])
        pu = over_probability(row["usage_mean"], line, MARKET, row["usage_errors"])
        pm = quote["market_over_proxy"]
        outcome = int(row["actual"] > line)
        candidates = [
            (pu * quote["over_decimal"] - 1, "over", quote["over_decimal"], pu),
            ((1 - pu) * quote["under_decimal"] - 1, "under", quote["under_decimal"], 1 - pu),
        ]
        ev, side, odds, win_probability = max(candidates, key=lambda c: (c[0], c[1]))
        won = outcome if side == "over" else 1 - outcome
        records.append(
            {
                **quote,
                "player_name": row.get("player_name"),
                "baseline_over": pb,
                "usage_over": pu,
                "observed_over": outcome,
                "brier_usage": (pu - outcome) ** 2,
                "brier_baseline": (pb - outcome) ** 2,
                "brier_market_proxy": (pm - outcome) ** 2,
                "brier_usage_minus_market": (pu - outcome) ** 2 - (pm - outcome) ** 2,
                "side": side,
                "raw_ev_proxy": ev,
                "selected_probability": win_probability,
                "nominal_return_proxy": odds - 1 if won else -1,
                "won": bool(won),
                "betting_authorized": False,
            }
        )
    # One candidate per game, chosen WITHOUT its outcome or realized return.
    chosen = {}
    for record in records:
        if record["raw_ev_proxy"] <= 0:
            continue
        gid = record["game_id"]
        key = (-record["raw_ev_proxy"], record["player_id"], record["side"])
        if gid not in chosen or key < chosen[gid][0]:
            chosen[gid] = (key, record)
    selected = [chosen[gid][1] for gid in sorted(chosen)]

    def clustered_interval(field):
        groups = defaultdict(list)
        for record in records:
            groups[record["game_id"]].append(record[field])
        if len(groups) < 20:
            return None
        values = list(groups.values())
        rng = np.random.default_rng(20261014)
        samples = []
        for _ in range(4000):
            picked = [values[i] for i in rng.integers(0, len(values), size=len(values))]
            samples.append(sum(map(sum, picked)) / sum(map(len, picked)))
        return np.quantile(samples, [0.00625, 0.99375]).tolist()  # Four-family adjustment.

    summary = {
        field: sum(r[field] for r in records) / len(records) if records else None
        for field in (
            "brier_usage",
            "brier_baseline",
            "brier_market_proxy",
            "brier_usage_minus_market",
        )
    }
    calibration = {}
    for name, field in (
        ("usage", "usage_over"),
        ("baseline", "baseline_over"),
        ("market_proxy", "market_over_proxy"),
    ):
        probabilities = [r[field] for r in records]
        outcomes = [r["observed_over"] for r in records]
        bins = defaultdict(list)
        for p, y in zip(probabilities, outcomes, strict=True):
            bins[min(9, int(p * 10))].append((p, y))
        log_loss = sum(
            -math.log(max(1e-12, p if y else 1 - p))
            for p, y in zip(probabilities, outcomes, strict=True)
        )
        ece = sum(
            abs(sum(p for p, y in group) - sum(y for p, y in group)) for group in bins.values()
        )
        calibration[name] = {
            "log_loss": log_loss / len(records) if records else None,
            "ece_fixed_10_bins": ece / len(records) if records else None,
        }
    return {
        "status": "ACTUAL_LINE_DIAGNOSTIC_NOT_EDGE_PROOF"
        if records
        else "NO_MATCHED_MARKET_EVIDENCE",
        "matched_contracts": len(records),
        "distinct_games": len({r["game_id"] for r in records}),
        "audit": dict(counts),
        "probability_scores": summary,
        "calibration_diagnostics": calibration,
        "game_cluster_brier_difference_interval_98_75": clustered_interval(
            "brier_usage_minus_market"
        ),
        "selection_rule": "ONE_POSITIVE_MAX_RAW_EV_CONTRACT_PER_GAME_TIES_BY_PLAYER_ID_SIDE",
        "research_selected_games": len(selected),
        "nominal_return_proxy_mean": sum(r["nominal_return_proxy"] for r in selected)
        / len(selected)
        if selected
        else None,
        "records": records,
        "research_selections": selected,
        "verified_executable_decisions": 0,
        "clv": None,
        "actual_roi": None,
        "original_price_timestamps_verified": False,
        "settlement_rules_verified": False,
        "future_forward_record_required": True,
        "validated_edges": 0,
        "promotion_earned": False,
        "betting_authorized": False,
    }
