"""Official NFL postgame grading of frozen editorial picks; research only."""
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlparse
from scripts.public_bettor_tracking import read_picks, review, as_utc, theoretical_units

def audit(picks, evidence):
    if evidence.get("schema") != "public_expert_official_result_snapshot_v1":
        raise ValueError("unknown evidence schema")
    if evidence.get("source_type") != "OFFICIAL_NFL_FINAL_AND_PLAYER_GAME_LOG":
        raise ValueError("wrong evidence source")
    if as_utc(evidence["verified_utc"]) <= as_utc(evidence["kickoff_utc"]):
        raise ValueError("not postgame")
    if as_utc(evidence["verified_utc"]) > datetime.now(timezone.utc):
        raise ValueError("future evidence")
    if urlparse(evidence["official_game_url"]).hostname != "www.nfl.com":
        raise ValueError("official NFL URL required")
    game = dict(gameId=evidence["event_id"], homeTeam=evidence["home_team"],
                awayTeam=evidence["away_team"], homeScore=evidence["home_score"],
                awayScore=evidence["away_score"])
    stats = {}
    for p in evidence["player_stats"]:
        key = (p["participant"].casefold(), p["market"])
        if key in stats or urlparse(p["source_url"]).hostname != "www.nfl.com":
            raise ValueError("duplicate or unverified player source")
        if not isinstance(p["value"], int) or p["value"] < 0:
            raise ValueError("invalid stat")
        stats[key] = p
    report = review(picks, [game])
    source = {p["record_id"]: p for p in picks}
    for r in report["picks"]:
        p = source[r["record_id"]]
        if p["event_id"] != evidence["event_id"]:
            continue
        if as_utc(p["kickoff_utc"]) != as_utc(evidence["kickoff_utc"]):
            raise ValueError("game kickoff mismatch")
        if r["grade"] == "NEEDS_PLAYER_BOX":
            s = stats.get((p["participant"].casefold(), p["market"]))
            if s is None:
                continue
            difference = s["value"] - float(p["line"])
            if p["selection"].lower() == "under":
                difference = -difference
            r["grade"] = "W" if difference > 0 else "L" if difference < 0 else "P"
            r["article_price_hypothetical_units"] = theoretical_units(r["grade"], p["american_odds"])
            r["observed_player_stat"] = s["value"]
            r["official_player_source"] = s["source_url"]
        if r["grade"] in ("W", "L", "P"):
            r["official_game_source"] = evidence["official_game_url"]
            r["official_verified_utc"] = evidence["verified_utc"]
    by_author = {}
    for r in report["picks"]:
        a = by_author.setdefault(r["analyst"], {"sample": 0, "wins": 0, "losses": 0,
                           "pushes": 0, "pending": 0, "needs_player_box": 0,
                           "hypothetical_units": 0.0})
        a["sample"] += 1
        a[{"W":"wins","L":"losses","P":"pushes","PENDING":"pending",
           "NEEDS_PLAYER_BOX":"needs_player_box"}[r["grade"]]] += 1
        a["hypothetical_units"] += r["article_price_hypothetical_units"] or 0
    report["author_summary"] = by_author
    report["status"] = "OFFICIAL_POSTGAME_EDITORIAL_HYPOTHETICAL"
    report["decision"] = "PASS"
    return report

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--picks", default="research/public_expert_picks_2026.csv")
    p.add_argument("--official", default="research/public_expert_official_results_2026.json")
    p.add_argument("--output", default="reports/public_bettor_official_audit.json")
    a = p.parse_args()
    result = audit(read_picks(a.picks), json.loads(Path(a.official).read_text()))
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["author_summary"], indent=2))

if __name__ == "__main__":
    main()
