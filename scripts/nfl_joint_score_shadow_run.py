"""Run the non-production NFL joint score shadow without sportsbook inputs."""
from __future__ import annotations
import argparse, json
from datetime import UTC, datetime
from pathlib import Path
from nfl.joint_score_shadow import score_shadow, utc

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise ValueError("SOURCE_FILE_MISSING_OR_NOT_INDEPENDENTLY_VERIFIED: "+str(path))
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frozen",type=Path,default=Path("history/score_forward_2026.jsonl"))
    p.add_argument("--verified-paired-history",type=Path,required=True)
    p.add_argument("--game-id",required=True)
    p.add_argument("--as-of",help="UTC as-of time; defaults to current UTC")
    p.add_argument("--draws",type=int,default=10000)
    p.add_argument("--seed",type=int,default=20261012)
    p.add_argument("--total-line",type=float)
    p.add_argument("--home-spread",type=float)
    p.add_argument("--out",type=Path,default=Path("reports/joint_score_shadow.json"))
    a=p.parse_args()
    originals=[r for r in load_jsonl(a.frozen) if r.get("event_id")==a.game_id]
    if len(originals)!=1:
        raise ValueError("REQUIRES_EXACTLY_ONE_FIRST_SEEN_SCORE_FORECAST")
    at=utc(a.as_of) if a.as_of else datetime.now(UTC)
    try:
        result=score_shadow(
            originals[0],load_jsonl(a.verified_paired_history),at=at,
            seed=a.seed,draws=a.draws,total_line=a.total_line,home_spread=a.home_spread)
    except ValueError as e:
        print(json.dumps({"status":"BLOCKED_RESEARCH_ONLY","reason":str(e),
                          "game_id":a.game_id,"action":"PASS_RESEARCH_ONLY",
                          "wager_authorized":False,"estimated_ev":None}))
        return
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"status":result["status"],"action":result["action"],
                      "out":str(a.out),"training_games":result["training_games"]}))
if __name__=="__main__":
    main()
