"""Fail-closed research-only match of v2 NFL player props to BetMGM editorial prices.

This is NOT a scraper, executable quote, calibrated fair-value model or bet
recommendation. Editorial prices are never sportsbook-origin timestamps.
Never interpolate synthetic grid thresholds to an offered line.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone, timedelta
import json
import math
from pathlib import Path
from urllib.parse import urlparse

VERSION = "FIRST_PARTY_EDITORIAL_PROP_EXACT_LINE_RESEARCH_V1"
FROZEN_MODEL = "PROP_PARTICIPATION_CONDITIONAL_V2_RESEARCH"
ALLOWED_MARKETS = frozenset(("receptions", "receiving_yards", "rushing_yards", "passing_yards"))


def utc(value):
    if not isinstance(value, str) or not value:
        raise ValueError("required original UTC timestamp missing")
    t = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if t.tzinfo is None or t.utcoffset() != timedelta(0):
        raise ValueError("timestamp must be explicitly UTC")
    return t.astimezone(timezone.utc)


def finite_float(value):
    if isinstance(value, bool):
        raise ValueError("boolean is not a number")
    x = float(value)
    if not math.isfinite(x):
        raise ValueError("non-finite value")
    return x


def american_to_implied(value):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("American odds must be integer, not unquoted placeholders")
    if -100 < value < 100:
        raise ValueError("American odds outside valid range")
    return 100 / (100 + value) if value >= 100 else (-value) / (100 - value)


def frozen_index(rows, *, cutoff):
    result = {}
    for row in rows:
        if row.get("version") != FROZEN_MODEL or str(row.get("recommended_bet", "")).lower() != "false":
            raise ValueError("source must be immutable research-only v2 forecasts")
        if row.get("bookmaker") or row.get("american_odds") or row.get("model_ev"):
            raise ValueError("v2 probability source must not claim bookmaker prices or wager EV")
        if row.get("market") not in ALLOWED_MARKETS:
            raise ValueError("unknown frozen player market")
        first = utc(row["first_observed_at_utc"])
        kickoff = utc(row["kickoff_utc"])
        if not first < kickoff:
            raise ValueError("original forecast at/after game kickoff")
        line = finite_float(row["synthetic_line"])
        if abs((line % 1) - .5) > 1e-9:
            raise ValueError("frozen probability not at half-point stat threshold")
        over = finite_float(row["p_over"])
        under = finite_float(row["p_under"])
        if min(over, under) < 0 or max(over, under) > 1 or not math.isclose(over + under, 1, abs_tol=1e-5):
            raise ValueError("invalid over/under probability simplex")
        key = (str(row["game_id"]), str(row["player_id"]), row["market"], line)
        if key in result:
            raise ValueError("duplicate original frozen exact contract")
        if first <= cutoff:
            result[key] = row
    return result


def research_match(frozen_rows, editorial, *, observed_utc):
    """Diagnostic only, and ALWAYS PASS. Inputs are independently sourced."""
    observed = utc(observed_utc)
    originals = frozen_index(frozen_rows, cutoff=observed)
    snapshots = editorial.get("quotes")
    if not isinstance(snapshots, list):
        raise ValueError("quotes list required")
    seen = set()
    out = []
    for item in snapshots:
        event = str(item["event_id"])
        player_id = str(item["player_id"])
        market = str(item["market"])
        line = finite_float(item["line"])
        if not event or not player_id or market not in ALLOWED_MARKETS:
            raise ValueError("complete player, event, market identity required")
        if abs(line % 1 - .5) > 1e-9:
            raise ValueError("whole-number/push prop is unsupported")
        if item.get("period") != "FULL_GAME" or item.get("bookmaker") != "BetMGM":
            raise ValueError("unsupported period or bookmaker")
        if item.get("source_kind") != "FIRST_PARTY_EDITORIAL":
            raise ValueError("only sourced first-party editorial research accepted")
        url = str(item.get("source_url", ""))
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname not in ("sports.betmgm.com", "www.betmgm.com"):
            raise ValueError("first-party BetMGM HTTPS source URL required")
        if item.get("book_origin_quote_utc") is not None or item.get("account_offer_verified") is not False:
            raise ValueError("editorial source cannot impersonate verified book-origin or account quote")
        kickoff = utc(item["kickoff_utc"])
        if observed >= kickoff:
            raise ValueError("editorial capture not before kickoff")
        published = utc(item["source_publication_utc"]) if item.get("source_publication_utc") else None
        if published is not None and published > observed:
            raise ValueError("source was published after research observation")
        key = (event, player_id, market, line)
        if key in seen:
            raise ValueError("duplicate sourced exact contract")
        seen.add(key)
        original = originals.get(key)
        blockers = ["ORIGINAL_SPORTSBOOK_OFFER_TIME_UNKNOWN",
                    "ACCOUNT_EXECUTION_UNVERIFIED",
                    "PLAYER_PARTICIPATION_AND_VOID_UNVERIFIED",
                    "MODEL_NOT_MARKET_VALIDATED"]
        if published is None:
            blockers.append("EDITORIAL_PUBLICATION_UTC_UNVERIFIED")
        elif observed - published > timedelta(minutes=15):
            blockers.append("EDITORIAL_PUBLICATION_OLDER_THAN_15_MINUTES")
        if original is None:
            blockers.append("NO_ORIGINAL_FROZEN_PROBABILITY_AT_EXACT_THRESHOLD_AND_PLAYER")
        else:
            # A source name or asserted GSIS crosswalk is NOT independently
            # verified here; provenance and exact identity are mandatory.
            if item.get("player_id_crosswalk_verified") is not True or not item.get("player_id_crosswalk_source_url"):
                blockers.append("GSIS_ID_CROSSWALK_UNVERIFIED")
                original = None
            elif str(item.get("player_name", "")).casefold().strip() != str(original.get("player_name", "")).casefold().strip():
                blockers.append("SOURCE_PLAYER_NAME_DIFFERS_FROM_FROZEN_PLAYER_ID")
                original = None
        if original is not None:
            if utc(original["kickoff_utc"]) != kickoff:
                raise ValueError("kickoff differs between frozen model and source")
            if published is not None and utc(original["first_observed_at_utc"]) > published:
                blockers.append("FORECAST_CREATED_AFTER_EDITORIAL_PUBLICATION")
        forecast_at_publication = (
            original is not None and published is not None and
            utc(original["first_observed_at_utc"]) <= published
        )
        if original is not None and not forecast_at_publication:
            blockers.append("NO_VALID_ORIGINAL_FORECAST_BEFORE_TIMED_EDITORIAL")
        po = pu = model = gap = None
        try:
            if item.get("over_american") is None or item.get("under_american") is None:
                raise ValueError("TWO_SIDED_PRICE_MISSING")
            over = american_to_implied(item["over_american"])
            under = american_to_implied(item["under_american"])
            po = round(over / (over + under), 8)
            pu = round(under / (over + under), 8)
            if forecast_at_publication:
                model = round(finite_float(original["p_over"]), 8)
                gap = round(model - po, 8)
        except (ValueError, TypeError) as error:
            if str(error) == "TWO_SIDED_PRICE_MISSING":
                blockers.append("TWO_SIDED_PRICE_MISSING")
            else:
                raise
        out.append({
            "event_id": event, "player_id": player_id, "market": market,
            "line": line, "period": "FULL_GAME", "source_url": url,
            "source_publication_utc": item.get("source_publication_utc"),
            "collector_observed_utc": observed.isoformat(),
            "original_forecast_utc": original["first_observed_at_utc"] if original else None,
            "exact_original_model_match": original is not None,
            "time_eligible_comparison": forecast_at_publication,
            "source_identity_crosswalk_independently_authenticated": False,
            "two_sided_editorial_quote": po is not None,
            "no_vig_conditional_over_price_reference": po,
            "no_vig_conditional_under_price_reference": pu,
            "unpriced_participation_conditional_model_p_over": model,
            "difference_is_not_true_ev": gap,
            "blockers": blockers,
            "source_original_sportsbook_offer_time_known": False,
            "account_executable_price_verified": False,
            "model_vs_market_profit_validated": False,
            "estimated_true_ev": None, "decision": "PASS",
        })
    return {
        "version": VERSION, "status": "FIRST_PARTY_EDITORIAL_RESEARCH_ONLY",
        "decision": "PASS", "betting_recommendations": [],
        "research_contracts": len(out),
        "exact_frozen_contract_matches": sum(x["exact_original_model_match"] for x in out),
        "exact_matches_with_two_editorial_prices": sum(x["time_eligible_comparison"] and x["two_sided_editorial_quote"] for x in out),
        "original_book_offer_update_times_verified": 0,
        "account_executable_betmgm_offers": 0,
        "profitable_edges_verified": 0, "contracts": out,
        "warning": "Editorial price timestamps and claimed player-ID crosswalk are NOT independently authenticated by this script. No cross-line interpolation, true EV, paper ROI, or recommended bets.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frozen", type=Path,
                   default=Path("history/prop_participation_shadow_v2_2026.csv"))
    p.add_argument("--editorial", type=Path, required=True)
    p.add_argument("--observed-utc", required=True)
    p.add_argument("--report", type=Path,
                   default=Path("reports/first_party_editorial_prop_bridge.json"))
    a = p.parse_args()
    with a.frozen.open(newline="", encoding="utf8") as fp:
        frozen = list(csv.DictReader(fp))
    result = research_match(
        frozen, json.loads(a.editorial.read_text(encoding="utf8")),
        observed_utc=a.observed_utc,
    )
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf8")
    print(json.dumps({k: result[k] for k in (
        "decision", "research_contracts", "exact_frozen_contract_matches",
        "exact_matches_with_two_editorial_prices", "profitable_edges_verified",
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
