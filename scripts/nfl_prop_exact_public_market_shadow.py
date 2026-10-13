"""Strict prospective join of frozen NFL player CDFs to public two-sided prop quotes.

Research only. A public quote observation is NOT proof of an accepted U.S.
BetMGM account offer. Empirical player CDFs condition on a recorded official
stat row, NOT sportsbook participation, no-action or void settlement. Never
calculate a model price edge, a ticket EV, or an authorized bet here.
"""
from __future__ import annotations

import argparse
from datetime import timedelta, datetime, timezone
import json
import math
from pathlib import Path

from scripts.nfl_player_prop_full_distribution import (
    VERSION as FROZEN_VERSION,
    exact_line_probabilities,
    hash_object,
    _utc,
)

VERSION = "NFL_PROP_EXACT_PUBLIC_MARKET_CDF_SHADOW_V1"
MAX_ORIGIN_AGE = timedelta(minutes=15)
INPUT = Path("research/nfl_prop_market_quote_candidates_2026.json")
FORECASTS = Path("history/player_prop_full_distributions_2026.jsonl")
CATALOG = Path("history/player_prop_2024_residual_catalog_v1.json")
JOURNAL = Path("history/nfl_prop_exact_market_shadow_v1_2026.jsonl")
REPORT = Path("reports/nfl_prop_exact_market_shadow_v1_2026.json")


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(row) for row in path.read_text(encoding="utf8").splitlines()
            if row.strip()]


def decimal_price(value):
    if isinstance(value, bool) or not isinstance(value, int) or abs(value) < 100:
        raise ValueError("American price must be an integer with magnitude >=100")
    return 1 + (value / 100 if value > 0 else 100 / abs(value))


def check_line(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError("numeric full-game player-stat line required")
    line = float(value)
    if not math.isfinite(line) or line < 0 or abs(2*line-round(2*line)) > 1e-8:
        raise ValueError("prop threshold must be nonnegative integer or half-point")
    if line == 0:
        raise ValueError("zero whole-point line unsupported by frozen CDF")
    return line


def quote_id(q):
    return str(q["quote_id"])


def create_rows(quotes, forecasts, catalog, *, now):
    """Freeze contemporaneously before kickoff; never reconstruct backfilled bets.

    Quote provenance is supplied by the input provider, not authenticated
    sportsbook execution. Numeric gap/EV withheld until settlement denominator
    and calibration are properly identified and independently tested.
    """
    at = _utc(now)
    if not isinstance(quotes, list):
        raise ValueError("quotes must be a list")
    frozen = {}
    for row in forecasts:
        k = (str(row["game_id"]), str(row["player_id"]), str(row["market"]))
        if k in frozen:
            raise ValueError("ambiguous duplicate first-seen frozen player CDF")
        frozen[k] = row
    entries, seen = [], set()
    for q in quotes:
        if not isinstance(q, dict):
            raise ValueError("quote must be a dictionary")
        required = ("quote_id", "game_id", "player_id", "team", "market",
                    "kickoff_utc", "period", "bookmaker", "book_region",
                    "snapshot_id", "source_url", "source_observed_utc",
                    "source_quote_origin_utc", "sides", "line")
        if any(key not in q for key in required) or any(
                q[key] in (None, "") for key in required
                if key != "source_quote_origin_utc"):
            raise ValueError("incomplete original prop quote observation")
        if not isinstance(q["quote_id"], str) or not q["quote_id"].strip():
            raise ValueError("invalid quote identifier")
        if quote_id(q) in seen:
            raise ValueError("duplicate quote_id in input")
        seen.add(quote_id(q))
        if not str(q["source_url"]).startswith("https://"):
            raise ValueError("HTTPS provider source URL required")
        if q["period"] != "FULL_GAME":
            raise ValueError("not a full-game prop instrument")
        if not isinstance(q["bookmaker"], str) or not q["bookmaker"].strip():
            raise ValueError("bookmaker identity required")
        if not isinstance(q["book_region"], str) or not q["book_region"].strip():
            raise ValueError("book attribution region required")
        if not isinstance(q["snapshot_id"], str) or not q["snapshot_id"].strip():
            raise ValueError("source snapshot identity required")
        if q.get("account_price_accepted") is True or q.get("actual_bet_placed") is True:
            raise ValueError("this observational pipeline must not assert execution")
        observed = _utc(q["source_observed_utc"])
        kickoff = _utc(q["kickoff_utc"])
        if not observed <= at < kickoff:
            raise ValueError("quote observation cannot be joined after kickoff or before capture")
        k = (str(q["game_id"]), str(q["player_id"]), str(q["market"]))
        forecast = frozen.get(k)
        if forecast is None:
            raise ValueError("no exact frozen game/player/market GSIS ID match")
        if (str(q["team"]) != str(forecast["team"])
                or _utc(forecast["kickoff_utc"]) != kickoff):
            raise ValueError("market event identity disagrees with original frozen forecast")
        freeze = _utc(forecast["first_observed_at_utc"])
        if not freeze <= observed < kickoff:
            raise ValueError("original player CDF did not exist by quote capture")
        if q.get("source_origin_timestamp_claimed_verified") is True and q["source_quote_origin_utc"] is None:
            raise ValueError("cannot claim verified origin with missing time")
        line = check_line(q["line"])
        sides = q["sides"]
        if (not isinstance(sides, list) or len(sides) != 2 or
                any(not isinstance(side, dict) for side in sides) or
                {side.get("selection") for side in sides} != {"OVER", "UNDER"}):
            raise ValueError("same-snapshot two-sided OVER/UNDER prices required")
        price_by_side = {}
        for side in sides:
            if check_line(side.get("line")) != line:
                raise ValueError("over/under price thresholds are not the same contract")
            if any(side.get(name) != q[name] for name in (
                    "snapshot_id", "bookmaker", "source_url",
                    "source_observed_utc", "source_quote_origin_utc")):
                raise ValueError("bookmaker side snapshots have mismatched provenance")
            price_by_side[side["selection"]] = decimal_price(side.get("american_odds"))
        raw_over = 1 / price_by_side["OVER"]
        raw_under = 1 / price_by_side["UNDER"]
        denominator = raw_over + raw_under
        if not 1 < denominator <= 1.35:
            raise ValueError("suspicious two-sided bookmaker hold")
        origin_raw = q["source_quote_origin_utc"]
        origin = _utc(origin_raw) if origin_raw is not None else None
        if origin is not None and not origin <= observed:
            raise ValueError("quote origin timestamp cannot follow observation")
        if origin is None:
            provenance = "CAPTURE_ONLY_ORIGIN_UNKNOWN"
        elif origin < freeze:
            provenance = "ORIGIN_PREDATES_FIRST_FROZEN_FORECAST"
        elif observed-origin > MAX_ORIGIN_AGE:
            provenance = "SOURCE_ORIGIN_STALE_AT_CAPTURE"
        else:
            provenance = "CLAIMED_RECENT_ORIGIN_UNAUTHENTICATED"
        # The current CDF is conditioned on an official player-stat row.
        # It does not establish the unconditional sportsbook target.
        conditional = exact_line_probabilities(
            forecast, catalog, line, observed_utc=observed.isoformat()
        )
        entry = {
            "model": VERSION, "quote_id": q["quote_id"],
            "game_id": k[0], "player_id": k[1], "market": k[2],
            "team": forecast["team"], "period": "FULL_GAME",
            "line": line, "kickoff_utc": kickoff.isoformat(),
            "bookmaker_attributed": q["bookmaker"],
            "book_region_attributed": q["book_region"],
            "snapshot_id": q["snapshot_id"], "source_url": q["source_url"],
            "source_observed_utc": observed.isoformat(),
            "source_quote_origin_utc": origin.isoformat() if origin else None,
            "source_origin_timestamp_claimed_verified": bool(q.get(
                "source_origin_timestamp_claimed_verified", False)),
            "origin_provenance": provenance,
            "source_origin_independently_authenticated": False,
            "source_price_over_american": next(s["american_odds"] for s in sides if s["selection"] == "OVER"),
            "source_price_under_american": next(s["american_odds"] for s in sides if s["selection"] == "UNDER"),
            "source_pair_hold": round(denominator-1, 8),
            "source_pair_over_no_vig_conditional_on_nonpush": round(raw_over/denominator, 9),
            "original_forecast_utc": freeze.isoformat(),
            "original_forecast_sha256": hash_object(forecast),
            "original_2024_catalog_sha256": catalog["sha256"],
            "p_model_over_given_official_stat_row": conditional[
                "p_over_conditional_on_official_stat_row"],
            "p_model_under_given_official_stat_row": conditional[
                "p_under_conditional_on_official_stat_row"],
            "p_model_push_given_official_stat_row": conditional[
                "p_push_conditional_on_official_stat_row"],
            "conditional_player_stat_row_is_not_book_settlement": True,
            "player_active_verified": False, "participation_void_verified": False,
            "account_accepted_price_verified": False,
            "unconditional_market_pwin_push_loss_void": None,
            "model_minus_book_comparable_probability_delta": None,
            "estimated_ev": None, "conservative_ev": None,
            "paper_bet": False, "real_bet": False,
            "decision": "PASS", "status": "RESEARCH_ONLY",
        }
        entry["first_join_registered_utc"] = at.isoformat()
        entry["original_row_sha256"] = hash_object(entry)
        entries.append(entry)
    return entries


def append_first_seen(entries, path):
    """Append only; refuse silently changing earlier same quote identifiers."""
    path = Path(path)
    prior = read_jsonl(path)
    known = {}
    for row in prior:
        if row.get("original_row_sha256") != hash_object(
                {k:v for k,v in row.items() if k != "original_row_sha256"}):
            raise ValueError("frozen quote-model research row integrity failure")
        key = quote_id(row)
        if key in known:
            raise ValueError("duplicate historical quote identifier")
        known[key] = row
    new = []
    for row in entries:
        key = quote_id(row)
        if key in known:
            # Preserve historical first-seen instead of letting a re-run
            # change the original research record.
            old = known[key]
            for k in ("game_id", "player_id", "market", "line", "snapshot_id",
                      "source_observed_utc", "source_price_over_american",
                      "source_price_under_american"):
                if old[k] != row[k]:
                    raise ValueError("refusing to rewrite existing quote event")
            continue
        known[key] = row
        new.append(row)
    if new:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf8") as out:
            for row in new:
                out.write(json.dumps(row, sort_keys=True, allow_nan=False)+"\n")
    return {"existing": len(prior), "new": len(new), "cumulative": len(prior)+len(new)}


def run(*, quotes_path=INPUT, forecasts_path=FORECASTS,
        catalog_path=CATALOG, journal_path=JOURNAL, report_path=REPORT,
        now=None):
    at = _utc(now or datetime.now(timezone.utc))
    body = json.loads(Path(quotes_path).read_text(encoding="utf8"))
    if not isinstance(body, dict) or not isinstance(body.get("records"), list):
        raise ValueError("research input must be an object with records array")
    quotes = body["records"]
    # Empty inputs are NOT fabricated into accepted bookmaker observations.
    if quotes:
        forecasts = read_jsonl(forecasts_path)
        catalog = json.loads(Path(catalog_path).read_text(encoding="utf8"))
        entries = create_rows(quotes, forecasts, catalog, now=at)
    else:
        entries = []
    saved = append_first_seen(entries, journal_path)
    report = {
        "model": VERSION, "observed_utc": at.isoformat(),
        "candidate_public_quote_observations": len(quotes),
        "first_seen_quote_model_joins": saved,
        "recent_claimed_origin_rows": sum(
            e["origin_provenance"] == "CLAIMED_RECENT_ORIGIN_UNAUTHENTICATED"
            for e in entries),
        "independently_verified_accepted_US_BetMGM_prices": 0,
        "book_market_comparable_unconditional_probabilities": 0,
        "prospective_settled_bets": 0, "true_ev": None,
        "calibrated_edge_found": False, "decision": "PASS",
        "status": "RESEARCH_ONLY",
    }
    dest = Path(report_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, sort_keys=True,
                               allow_nan=False)+"\n", encoding="utf8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--forecasts", type=Path, default=FORECASTS)
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--journal", type=Path, default=JOURNAL)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    print(json.dumps(run(quotes_path=args.input,
                         forecasts_path=args.forecasts,
                         catalog_path=args.catalog,
                         journal_path=args.journal,
                         report_path=args.report), sort_keys=True))


if __name__ == "__main__":
    main()
