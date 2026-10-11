"""Audit a public NFL bettor's pregame picks and hypothetical follower prices.

This does not reconstruct a private model, access Pikkit, scrape X, or certify a
screenshot. Flat one-unit paper settlement only when individual pre-kickoff
sources, full-game exact contracts, and verified postgame scores are present.
No wagers, advisory or positive-EV claims. Profiles/graphs are not picks.
"""
from __future__ import annotations

from collections import defaultdict
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from scripts.professional_method_audit import _utc, decimal_odds

VERSION = "PUBLIC_BETTOR_EXACT_PICK_AUDIT_V1"
MARKETS = {
    "GAME_SPREAD": {"HOME", "AWAY"},
    "GAME_TOTAL": {"OVER", "UNDER"},
    "GAME_MONEYLINE": {"HOME", "AWAY"},
}
SOURCE_TYPES = {"PUBLIC_X_POST", "PUBLIC_PIKKIT_INDIVIDUAL_POST",
                "PUBLIC_AUTHOR_ARTICLE"}
FIELDS = ("record_id", "bettor", "source_url", "source_type",
          "posted_utc", "captured_utc", "kickoff_utc",
          "event_id", "league", "market", "period", "participant",
          "selection", "line", "american_odds", "bookmaker", "region")


def _utc_string(s):
    return _utc(s).isoformat()


def validate_pick(pick):
    if not isinstance(pick, dict) or any(pick.get(k) is None or
            pick[k] == "" for k in FIELDS):
        raise ValueError("missing individual pick provenance")
    if pick["league"] != "NFL" or pick["source_type"] not in SOURCE_TYPES:
        raise ValueError("unsupported league or source")
    if pick["market"] not in MARKETS or pick["selection"] not in MARKETS[pick["market"]]:
        raise ValueError("unsupported market or selection")
    if pick["period"] != "FULL_GAME" or pick["participant"] != "GAME":
        raise ValueError("only full-game individual NFL game bets supported")
    if not str(pick["source_url"]).startswith("https://"):
        raise ValueError("source URL must be https")
    if not _utc(pick["posted_utc"]) <= _utc(pick["captured_utc"]) < _utc(pick["kickoff_utc"]):
        raise ValueError("public pick is not timestamped and captured before kickoff")
    decimal_odds(pick["american_odds"])
    if pick["market"] == "GAME_MONEYLINE":
        if pick["line"] != "NA":
            raise ValueError("moneyline uses line NA")
    else:
        try:
            v = float(pick["line"])
        except (ValueError, TypeError):
            raise ValueError("unparseable contract line")
        if not math.isfinite(v):
            raise ValueError("nonfinite line")
        if pick["market"] == "GAME_SPREAD" and pick.get("line_convention") != "HOME_SIGNED":
            raise ValueError("spread must use HOME_SIGNED handicap")
    return pick


def _validate_follower(pick, price):
    """Hypothetical after-post observed public follower price; never an executed bet."""
    if price is None:
        return None
    if not isinstance(price, dict):
        raise ValueError("follower quote not a dict")
    fields = ("event_id", "market", "period", "participant", "selection",
              "line", "bookmaker", "region", "source_url", "quote_utc",
              "captured_utc", "american_odds")
    if any(price.get(k) is None or price[k] == "" for k in fields):
        raise ValueError("missing follower price quote")
    if any(str(pick[k]) != str(price[k]) for k in
           ("event_id", "market", "period", "participant", "selection", "line")):
        raise ValueError("follower price is a different betting contract")
    if not str(price["source_url"]).startswith("https://"):
        raise ValueError("follower price lacks actual source URL")
    if not _utc(pick["posted_utc"]) <= _utc(price["quote_utc"]) <= _utc(price["captured_utc"]) < _utc(pick["kickoff_utc"]):
        raise ValueError("follower quote is stale, unobserved or post-kickoff")
    decimal_odds(price["american_odds"])
    return price


def _grade(pick, result):
    if result is None:
        return "PENDING"
    if str(result.get("event_id")) != str(pick["event_id"]):
        raise ValueError("settlement event mismatch")
    if _utc(result["final_utc"]) <= _utc(pick["kickoff_utc"]):
        raise ValueError("settlement precedes kickoff")
    home, away = (float(result[k]) for k in ("home_score","away_score"))
    if any(not math.isfinite(v) or v < 0 or v != int(v) for v in (home, away)):
        raise ValueError("invalid NFL final score")
    if pick["market"] == "GAME_SPREAD":
        # HOME_SIGNED is a signed handicap for HOME. AWAY is its inverse.
        diff = (home-away) + float(pick["line"])
        if pick["selection"] == "AWAY": diff = -diff
    elif pick["market"] == "GAME_TOTAL":
        diff = home + away - float(pick["line"])
        if pick["selection"] == "UNDER": diff = -diff
    else:
        diff = home - away
        if pick["selection"] == "AWAY": diff = -diff
    return "W" if diff > 0 else "L" if diff < 0 else "P"


def _flat_unit_return(outcome, american_odds):
    """Hypothetical units per $1 risked, not a Pikkit synced or placed bet."""
    if outcome == "W":
        return decimal_odds(american_odds) - 1
    if outcome == "L":
        return -1.0
    if outcome == "P":
        return 0.0
    return None


def _summary(rows, field):
    groups=defaultdict(list)
    for row in rows:
        groups[row[field]].append(row)
    out={}
    for name, cohort in sorted(groups.items()):
        scored=[r for r in cohort if r["grade"] in {"W","L","P"}]
        units=sum(r["poster_price_flat_one_unit_pnl"] for r in scored)
        follower=[r for r in cohort if r["follower_public_quote_flat_one_unit_pnl"] is not None]
        out[name]={
            "individual_picks":len(cohort),
            "graded":len(scored),
            "wins":sum(r["grade"]=="W" for r in scored),
            "losses":sum(r["grade"]=="L" for r in scored),
            "pushes":sum(r["grade"]=="P" for r in scored),
            "poster_price_hypothetical_one_unit_profit":round(units,6),
            "poster_price_hypothetical_roi":round(units/len(scored),6) if scored else None,
            "follower_quotes_with_settlement":len(follower),
            "follower_public_quote_hypothetical_one_unit_profit":
                round(sum(r["follower_public_quote_flat_one_unit_pnl"] for r in follower),6),
            "follower_public_quote_hypothetical_roi":
                round(sum(r["follower_public_quote_flat_one_unit_pnl"] for r in follower)/len(follower),6) if follower else None,
        }
    return out


def audit(picks, finals=(), follower_quotes=None):
    if not isinstance(picks,list) or not isinstance(finals,(tuple,list)):
        raise ValueError("picks and finals must be lists")
    if follower_quotes is None:follower_quotes={}
    if not isinstance(follower_quotes,dict):
        raise ValueError("follower quotes must be keyed by record ID")
    indexed={}
    for g in finals:
        if not isinstance(g,dict) or not g.get("event_id"):
            raise ValueError("invalid settlement")
        if g["event_id"] in indexed:
            raise ValueError("duplicate event settlement")
        indexed[g["event_id"]]=g
    seen=set()
    outputs=[]
    for p in picks:
        validate_pick(p)
        if p["record_id"] in seen:
            raise ValueError("duplicate public pick record")
        seen.add(p["record_id"])
        follower=_validate_follower(p,follower_quotes.get(p["record_id"]))
        grade=_grade(p,indexed.get(p["event_id"]))
        poster=_flat_unit_return(grade,p["american_odds"])
        copied=_flat_unit_return(grade,follower["american_odds"]) if follower else None
        outputs.append({
            "record_id":p["record_id"],"bettor":p["bettor"],"market":p["market"],
            "event_id":p["event_id"],"selection":p["selection"],"line":p["line"],
            "source_url":p["source_url"],"posted_utc":p["posted_utc"],
            "captured_utc":p["captured_utc"],"grade":grade,
            "poster_price_flat_one_unit_pnl":poster,
            "follower_public_quote_flat_one_unit_pnl":copied,
            "follower_quote_time":follower["quote_utc"] if follower else None,
            "actual_wager_verified":False,"actual_pikkit_units_verified":False,
            "model_edge_validated":False,
        })
    if any(k not in seen for k in follower_quotes):
        raise ValueError("orphaned follower quote")
    result={
        "version":VERSION,"status":"PAPER_RESEARCH_ONLY" if outputs else "NO_INDIVIDUAL_PICKS_AVAILABLE",
        "decision":"PASS","actual_bets_verified":False,"pikkit_profits_confirmed":False,
        "edge_validated":False,"stake_convention":"FLAT_1_UNIT_HYPOTHETICAL_RISK_NOT_ACTUAL_STAKES",
        "synced_pikkit_access":False,
        "by_bettor":_summary(outputs,"bettor"),
        "by_market":_summary(outputs,"market"),
        "individual_picks":outputs,
        "cannot_infer_strategy_from_profile_screenshot":True,
        "blockers":[
            "No authenticated synced account-level Pikkit bet feed available through this script",
            "Publicly posted odds need not be available to followers when observed",
            "Profit at article/poster odds is hypothetical, not proof of account returns",
            "A public bettor record alone is not an independent probabilistic BetMGM pricing edge",
        ],
    }
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True,
                        help="JSON object with picks, results, follower_quotes")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    data=json.loads(args.input.read_text(encoding="utf8"))
    result=audit(data.get("picks",[]),data.get("results",[]),
                 data.get("follower_quotes",{}))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n",encoding="utf8")
    print(f"{result['status']}; {len(result['individual_picks'])} individual sourced picks; no wagers or confirmed profitable edge.")


if __name__=="__main__":
    main()
