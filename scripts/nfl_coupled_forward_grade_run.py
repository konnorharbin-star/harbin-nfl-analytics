"""Read-only forward score/prop research grading from a frozen JSON artifact."""
import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
import nflreadpy as nfl
from nfl.coupled_forward_grading import grade_snapshot

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--snapshot",required=True,type=Path)
    p.add_argument("--out",type=Path,default=Path("reports/coupled_forward_grade.json"))
    a=p.parse_args()
    snapshot=json.loads(a.snapshot.read_text())
    season=int(snapshot["season"])
    report=grade_snapshot(snapshot,nfl.load_schedules(season).to_dicts(),
                          nfl.load_player_stats([season]).to_dicts(),
                          at=datetime.now(UTC))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"graded":report["counts"]["graded"],
                      "not_mature":report["counts"]["not_mature"],
                      "metrics":report["metrics"],
                      "promotion_ready":report["promotion_ready"],
                      "output":str(a.out)}))
if __name__=="__main__":
    main()
