"""Web-first odds research against immutable pregame NFL/CFB forecasts.

An analyst searches/reads public webpages and commits immutable observations.
Do not infer sportsbook price-update timestamps from article/game timestamps.
No network, API key, scrape, sportsbook access, wager or live staking here.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlparse

from scripts.forward_price_validation import decimal, first_commit_time, utc
from scripts.walters_key_market_gate import _model, compare, half_point

SPEC = "walters_web_research_observation_v1"
SAFE_NAME = re.compile(r"^[a-zA-Z0-9_-]{3,110}$")
BOOK_SOURCES = {
    "FanDuel": {"fanduel.com", "www.fanduel.com"},
    "bet365": {"foxsports.com", "www.foxsports.com", "bet365.com", "www.bet365.com"},
}
MAX_PUBLICATION_LAG = timedelta(days=7)


def _url(value):
    if not isinstance(value, str):
        raise ValueError("NO_SOURCE_URL")
    parsed = urlparse(value)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username
            or parsed.password or not parsed.path or parsed.fragment):
        raise ValueError("INVALID_DIRECT_HTTPS_URL")
    return parsed.hostname.lower()


def _one(f, q, doc, *, now):
    """Validate an observed website quote, never upgrade it to executable."""
    if q.get("game_id") != f.get("game_id") or any(
        q.get(k) != f.get(k) for k in
        ("home_team", "away_team", "kickoff")
    ):
        raise ValueError("GAME_OR_KICKOFF_MISMATCH")
    if doc.get("sport") != f.get("sport"):
        raise ValueError("SPORT_MISMATCH")
    book = doc.get("book")
    if book not in BOOK_SOURCES:
        raise ValueError("BOOK_NOT_IN_VETTED_WEB_PUBLISHERS")
    domain = _url(doc.get("source_url"))
    if domain not in BOOK_SOURCES[book]:
        raise ValueError("SITE_NOT_ASSOCIATED_WITH_NAMED_BOOK")
    if doc.get("publisher_is_official_book") is not True:
        raise ValueError("BOOK_PUBLISHER_NOT_IDENTIFIED")
    if doc.get("verified_executable_sportsbook_quote") is not False:
        raise ValueError("MUST_BE_RESEARCH_ONLY")
    if doc.get("page_price_updated_at") is not None:
        # A web article's publication time is never a price-update timestamp.
        raise ValueError("NO_INDEPENDENT_PRICE_UPDATE_SOURCE")
    observed = utc(doc["observed_at"])
    published = utc(f["captured_at"])
    kickoff = utc(f["kickoff"])
    if not published <= observed < kickoff or observed > now:
        raise ValueError("OUTSIDE_PREGAME_FORECAST_WINDOW")
    if kickoff - observed > MAX_PUBLICATION_LAG:
        raise ValueError("WEB_OBSERVATION_BEYOND_SEVEN_DAYS")
    if (not isinstance(q.get("quote_context"), str)
            or len(q["quote_context"].strip()) < 12):
        raise ValueError("MISSING_HUMAN_READ_WEB_EVIDENCE")
    home_line, away_line = half_point(q["home_spread"]), half_point(
        q["away_spread"]
    )
    if home_line != -away_line:
        raise ValueError("SIDES_DIFFERENT_HANDICAP")
    for side in ("home", "away"):
        price = q[f"{side}_american_odds"]
        if isinstance(price, bool):
            raise ValueError("INVALID_PRICE")
        odds = float(price)
        if odds != int(odds):
            raise ValueError("NON_INTEGER_AMERICAN_ODDS")
        decimal(price)
    quoted = {
        "quote_id": f["game_id"] + "_web_" + book.lower(),
        "game_id": f["game_id"],
        "sport": f["sport"],
        "home_team": f["home_team"],
        "away_team": f["away_team"],
        "kickoff": f["kickoff"],
        "book": book,
        "source_url": doc["source_url"],
        "home_spread": home_line,
        "away_spread": away_line,
        "home_american_odds": int(q["home_american_odds"]),
        "away_american_odds": int(q["away_american_odds"]),
    }
    view = compare(f, quoted)
    view.update({
        "web_observed_at": observed.isoformat(),
        "publisher": doc["publisher"],
        "source_kind": doc["source_kind"],
        "source_price_timestamp_verified": False,
        "book_execution_independently_verified": False,
        "book_access_verified": False,
        "injury_and_personnel_context_verified": False,
        "weather_context_verified": False,
        "status": "PUBLIC_WEB_RESEARCH_ONLY_NO_EXECUTABLE_EDGE",
        "betting_authorized": False,
        "wager_authorized": False,
        "quote_context": q["quote_context"],
    })
    return view


def audit(forecast_root, web_root, *, now, commit_lookup=first_commit_time):
    """Git-first publication chronology and source-read evidence guardrail."""
    if now.tzinfo is None:
        raise ValueError("UTC-aware research audit clock required")
    forecasts, bad_forecasts = {}, {}
    for path in sorted(Path(forecast_root).glob("*.json")):
        try:
            f = json.loads(path.read_text(encoding="utf-8"))
            _model(f)
            original_commit = commit_lookup(path)
            if (f["game_id"] != path.stem or f["game_id"] in forecasts
                    or original_commit is None
                    or not utc(f["captured_at"]) <= original_commit < utc(
                        f["kickoff"]
                    )):
                raise ValueError("FORECAST_NOT_ORIGINALLY_PUBLISHED_PREGAME")
            forecasts[f["game_id"]] = (f, original_commit)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            bad_forecasts[path.name] = str(exc)
    records, blocked, seen = [], {}, set()
    pages = sorted(Path(web_root).glob("*.json"))
    for path in pages:
        reasons = []
        try:
            if not SAFE_NAME.fullmatch(path.stem):
                raise ValueError("UNSAFE_SOURCE_FILE_NAME")
            doc = json.loads(path.read_text(encoding="utf-8"))
            if doc.get("spec") != SPEC or doc.get("observation_id") != path.stem:
                raise ValueError("SOURCE_SPEC_OR_FILE_ID_MISMATCH")
            if doc.get("source_kind") != "public_webpage_read":
                raise ValueError("NOT_READ_FROM_WEB_PAGE")
            timestamp = utc(doc["observed_at"])
            original_commit = commit_lookup(path)
            if original_commit is None or not timestamp <= original_commit <= now:
                raise ValueError("WEB_CAPTURE_NOT_ORIGINALLY_COMMITTED_AFTER_READ")
            if not isinstance(doc.get("quotes"), list) or not doc["quotes"]:
                raise ValueError("NO_QUOTED_SIDES")
            for q in doc["quotes"]:
                name = q.get("game_id")
                if name not in forecasts:
                    reasons.append(str(name) + ":MISSING_PRIOR_MODEL_FORECAST")
                    continue
                f, model_commit = forecasts[name]
                if model_commit >= timestamp:
                    reasons.append(str(name) + ":MODEL_NOT_PUBLISHED_BEFORE_WEB_READ")
                    continue
                fingerprint = (name, str(doc.get("book")), timestamp.isoformat())
                if fingerprint in seen:
                    reasons.append(str(name) + ":DUPLICATE_SOURCE_BOOK_OBSERVATION")
                    continue
                seen.add(fingerprint)
                try:
                    view = _one(f, q, doc, now=now)
                    view["observation_id"] = doc["observation_id"]
                    view["original_capture_git_commit_verified"] = True
                    view["model_git_commit_precedes_web_capture"] = True
                    records.append(view)
                except (ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
                    reasons.append(str(name) + ":" + str(exc))
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            reasons.append(str(exc))
        if reasons:
            blocked[path.name] = reasons
    records.sort(key=lambda x: (x["game_id"], x["web_observed_at"], x["book"]))
    return {
        "spec": "walters_web_research_audit_v1",
        "status": "RESEARCH_ONLY_NO_VERIFIED_ECONOMIC_EDGE",
        "method": "human_or_assistant_reads_public_web_search_results_then_git_freezes_price",
        "frozen_original_forecasts": len(forecasts),
        "bad_forecasts": bad_forecasts,
        "web_evidence_documents": len(pages),
        "accepted_paired_web_spreads": len(records),
        "blocked_observations": blocked,
        "research_comparisons": records,
        "coverage_by_book": dict(Counter(x["book"] for x in records)),
        "webpage_read_does_not_prove_original_book_quote_update_time": True,
        "webpage_read_does_not_prove_executable_sportsbook_odds": True,
        "webpage_read_does_not_automatically_refresh": True,
        "no_paid_api_required": True,
        "historical_profitable_edge_proven": False,
        "closing_line_value_verified": False,
        "betting_authorized": False,
        "actual_wagers": 0,
    }


def main():
    from datetime import UTC, datetime
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--forecasts", type=Path,
        default=Path("history/walters_key_forward_v1/forecasts"),
    )
    parser.add_argument(
        "--web-root", type=Path,
        default=Path("history/walters_key_forward_v1/web_observations"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("reports/walters_web_research.json"),
    )
    args = parser.parse_args()
    result = audit(args.forecasts, args.web_root, now=datetime.now(UTC))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "frozen_forecasts": result["frozen_original_forecasts"],
        "paired_web_spreads": result["accepted_paired_web_spreads"],
        "source_errors": len(result["blocked_observations"]),
        "betting_authorized": False,
    }))


if __name__ == "__main__":
    main()
