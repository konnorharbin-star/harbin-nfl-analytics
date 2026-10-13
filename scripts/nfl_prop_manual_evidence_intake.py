"""Manual-only, non-scraping, ex-ante NFL player-prop observation intake.

Transcribe a permitted human-observed two-sided price into the strict v4.26
research quote schema. This is NOT an automated odds fetcher and does NOT
authenticate a BetMGM U.S. offer, player participation or an executable bet.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import unicodedata
from urllib.parse import urlsplit

from scripts.nfl_player_prop_full_distribution import _utc, hash_object
from scripts.nfl_prop_exact_public_market_shadow import (
    create_rows, read_jsonl, decimal_price, check_line
)

VERSION = "NFL_PROP_MANUAL_EVIDENCE_INTAKE_V1"
RAW = Path("research/nfl_prop_manual_evidence_2026.json")
OUT = Path("research/nfl_prop_market_quote_candidates_2026.json")
FORECASTS = Path("history/player_prop_full_distributions_2026.jsonl")
CATALOG = Path("history/player_prop_2024_residual_catalog_v1.json")
REPORT = Path("reports/nfl_prop_manual_evidence_intake_2026.json")
MAX_CAPTURE_AGE = timedelta(minutes=10)
MARKETS = {"receiving_yards", "rushing_yards", "passing_yards", "receptions"}
SOURCE_KINDS = {"PUBLIC_MANUALLY_VIEWED", "USER_VERIFIED_BOOK_SCREEN"}
DISALLOWED_SOURCE_HOSTS = {"edgelabs.bet"}  # Published terms prohibit data republication.


def _name_parts(raw):
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("player name is required")
    normalized = unicodedata.normalize("NFKD", raw).casefold()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized).strip()
    parts = normalized.split()
    if len(parts) < 2:
        raise ValueError("need given name or initial and exact surname")
    return parts


def name_matches(manual, frozen):
    """No fuzzy surname matches; initial-only candidates must be unique."""
    m, f = _name_parts(manual), _name_parts(frozen)
    return (m[-1] == f[-1] and
            (m[0] == f[0] or (len(m[0]) == 1 and f[0].startswith(m[0]))
             or (len(f[0]) == 1 and m[0].startswith(f[0]))))


def _validate_url(value):
    if not isinstance(value, str):
        raise ValueError("source URL is required")
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("source must have HTTPS URL")
    host = parsed.hostname.lower()
    if any(host == d or host.endswith("." + d) for d in DISALLOWED_SOURCE_HOSTS):
        raise ValueError("source terms forbid reuse of this market data")
    return value


def _resolve(entry, forecasts):
    k = (str(entry["game_id"]), str(entry["team"]), str(entry["market"]))
    matches = [
        f for f in forecasts
        if (str(f.get("game_id")), str(f.get("team")), str(f.get("market"))) == k
        and name_matches(entry["player_display_name"], str(f.get("player_name", "")))
    ]
    if entry.get("player_id") is not None:
        matches = [f for f in matches if str(f.get("player_id")) == str(entry["player_id"])]
    if len(matches) != 1:
        raise ValueError("player identity unresolved or ambiguous; supply verified GSIS ID")
    return matches[0]


def prepare(raw_entries, forecasts, *, now):
    """Construct strictly time-qualified public-attributed research candidates.

    A raw record cannot be retroactively accepted >10 minutes after capture,
    nor after kickoff. The original quote-update time stays genuinely NULL
    if the upstream source did not disclose it.
    """
    at = _utc(now)
    if not isinstance(raw_entries, list):
        raise ValueError("manual observations must be a list")
    output, ids = [], set()
    for row in raw_entries:
        if not isinstance(row, dict):
            raise ValueError("manual observation must be a dictionary")
        required = ("quote_id", "source_url", "source_kind",
                    "source_observed_utc", "source_quote_origin_utc",
                    "snapshot_id", "game_id", "team", "market", "period",
                    "player_display_name", "bookmaker", "book_region",
                    "line", "american_odds_over", "american_odds_under")
        if any(key not in row for key in required):
            raise ValueError("missing a required manual source observation field")
        if (not isinstance(row["quote_id"], str) or
                not row["quote_id"].strip() or row["quote_id"] in ids):
            raise ValueError("missing or duplicate manual quote ID")
        ids.add(row["quote_id"])
        _validate_url(row["source_url"])
        if row["source_kind"] not in SOURCE_KINDS:
            raise ValueError("only explicitly manual public/book observations allowed")
        if not isinstance(row.get("source_reuse_permitted"), bool) or not row["source_reuse_permitted"]:
            raise ValueError("source reuse permission must be explicitly confirmed")
        if row["period"] != "FULL_GAME" or row["market"] not in MARKETS:
            raise ValueError("unsupported full-game player-stat market")
        if not isinstance(row["snapshot_id"], str) or not row["snapshot_id"].strip():
            raise ValueError("source snapshot ID required")
        if not isinstance(row["bookmaker"], str) or not row["bookmaker"].strip():
            raise ValueError("bookmaker attribution required")
        if not isinstance(row["book_region"], str) or not row["book_region"].strip():
            raise ValueError("source book region required")
        if row.get("account_price_accepted") is True or row.get("actual_bet_placed") is True:
            raise ValueError("manual research intake never records execution")
        observed = _utc(row["source_observed_utc"])
        if not timedelta(0) <= at - observed <= MAX_CAPTURE_AGE:
            raise ValueError("manual source capture older than ten minutes or future")
        f = _resolve(row, forecasts)
        kickoff = _utc(f["kickoff_utc"])
        freeze = _utc(f["first_observed_at_utc"])
        if not freeze <= observed <= at < kickoff:
            raise ValueError("original player forecast or quote capture not pregame")
        line = check_line(row["line"])
        for key in ("american_odds_over", "american_odds_under"):
            decimal_price(row[key])
        if row["source_quote_origin_utc"] is not None:
            origin = _utc(row["source_quote_origin_utc"])
            if origin > observed:
                raise ValueError("source quote origin follows human observation")
        # No external data access in this module: manual observation data only.
        sides = []
        for selection, price in (
            ("OVER", row["american_odds_over"]),
            ("UNDER", row["american_odds_under"]),
        ):
            sides.append({
                "selection": selection, "line": line, "american_odds": price,
                "bookmaker": row["bookmaker"], "snapshot_id": row["snapshot_id"],
                "source_url": row["source_url"],
                "source_observed_utc": observed.isoformat(),
                "source_quote_origin_utc": (
                    _utc(row["source_quote_origin_utc"]).isoformat()
                    if row["source_quote_origin_utc"] is not None else None
                )
            })
        result = {
            "quote_id": row["quote_id"], "game_id": str(f["game_id"]),
            "player_id": str(f["player_id"]), "team": str(f["team"]),
            "market": row["market"], "kickoff_utc": kickoff.isoformat(),
            "period": "FULL_GAME", "bookmaker": row["bookmaker"],
            "book_region": row["book_region"], "snapshot_id": row["snapshot_id"],
            "source_url": row["source_url"],
            "source_observed_utc": observed.isoformat(),
            "source_quote_origin_utc": (
                _utc(row["source_quote_origin_utc"]).isoformat()
                if row["source_quote_origin_utc"] is not None else None
            ),
            "source_origin_timestamp_claimed_verified": False,
            "account_price_accepted": False, "actual_bet_placed": False,
            "line": line, "sides": sides,
            "source_kind": row["source_kind"], "capture_method": "MANUAL",
            "source_reuse_permitted_attested": True,
            "source_publication_utc": row.get("source_publication_utc"),
            "player_display_name_transcribed": row["player_display_name"],
            "original_frozen_forecast_sha256": hash_object(f),
            "validated_candidate_is_executable": False,
        }
        result["manual_candidate_sha256"] = hash_object(result)
        output.append(result)
    return output


def append_candidates(candidates, path):
    """Preserve original manual source and refuse changed earlier observations."""
    path = Path(path)
    prior = json.loads(path.read_text(encoding="utf8")) if path.exists() else {"records":[]}
    if not isinstance(prior, dict) or not isinstance(prior.get("records"), list):
        raise ValueError("candidate evidence registry format invalid")
    old = {}
    for r in prior["records"]:
        key = r["quote_id"]
        if key in old:
            raise ValueError("duplicate existing source quote ID")
        if r.get("manual_candidate_sha256") != hash_object(
                {k:v for k,v in r.items() if k!="manual_candidate_sha256"}):
            raise ValueError("manual first-seen quote evidence hash modified")
        old[key] = r
    fresh = []
    for r in candidates:
        key = r["quote_id"]
        if key in old:
            if r != old[key]:
                # Input can be re-entered later, but must not silently rewrite
                # a frozen book price, timestamp, or line.
                raise ValueError("attempt to rewrite first-seen manual quote")
            continue
        old[key] = r
        fresh.append(r)
    if fresh:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"records": list(old.values())},
                                  sort_keys=True, indent=2, allow_nan=False)+"\n",
                       encoding="utf8")
        tmp.replace(path)
    return {"previous":len(prior["records"]),"new":len(fresh),
            "total":len(old)}


def run(*, input_path=RAW, output_path=OUT, forecasts_path=FORECASTS,
        catalog_path=CATALOG, report_path=REPORT, now=None):
    at=_utc(now or datetime.now(timezone.utc))
    raw=json.loads(Path(input_path).read_text(encoding="utf8"))
    if not isinstance(raw,dict) or not isinstance(raw.get("records"),list):
        raise ValueError("manual intake input requires records list")
    entries=raw["records"]
    if entries:
        forecasts=read_jsonl(forecasts_path)
        catalog=json.loads(Path(catalog_path).read_text(encoding="utf8"))
        candidates=prepare(entries,forecasts,now=at)
        # Validate EVERY candidate using the existing prior model/catalog.
        create_rows(candidates,forecasts,catalog,now=at)
    else:
        candidates=[]
    result=append_candidates(candidates,output_path)
    report={
        "version":VERSION,"observed_utc":at.isoformat(),
        "human_observation_rows_in_input":len(entries),
        "strict_first_seen_candidates":result,
        "live_odds_fetched":False,"automated_scraping":False,
        "validated_executable_BetMGM_account_prices":0,
        "reliable_profitable_edge":False,"true_ev":None,
        "decision":"PASS","status":"RESEARCH_ONLY",
    }
    report_path=Path(report_path); report_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n",
                           encoding="utf8")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=RAW)
    parser.add_argument("--output",type=Path,default=OUT)
    args=parser.parse_args()
    print(json.dumps(run(input_path=args.input,output_path=args.output),sort_keys=True))


if __name__=="__main__":
    main()
