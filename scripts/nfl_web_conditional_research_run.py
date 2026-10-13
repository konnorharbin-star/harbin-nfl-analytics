"""Run web-search-supported conditional research off original frozen inputs."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json
from nfl.coupled_drive_player_shadow import simulate_joint
from nfl.web_conditioned_model import generate

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--forecast-scan",type=Path,required=True)
    p.add_argument("--replay-inputs",type=Path,required=True)
    p.add_argument("--web-report",type=Path,required=True)
    p.add_argument("--out",type=Path,default=Path("reports/nfl_web_conditional_scenarios.json"))
    args=p.parse_args()
    original_bytes=args.forecast_scan.read_bytes()
    web_bytes=args.web_report.read_bytes()
    replay_bytes=args.replay_inputs.read_bytes()
    original=json.loads(original_bytes)
    web=json.loads(web_bytes)
    frozen=json.loads(replay_bytes)
    source_sha=hashlib.sha256(original_bytes).hexdigest()
    if web.get("paired_model_artifact_sha256")!=source_sha:
        raise ValueError("WEB_CONTEXT_NOT_LINKED_TO_ORIGINAL_FORECAST")
    try:
        result=generate(original,frozen,web,now=datetime.now(timezone.utc),
                        simulate=simulate_joint)
    except ValueError as exc:
        result={"spec":"NFL_WEB_CONDITIONED_PLAYER_RESEARCH_V1",
                "status":"CONDITIONAL_SOURCE_BLOCKED","reason":str(exc),
                "scenarios":[],"original_forecast_modified":False,
                "estimated_ev":None,"wager_authorized":False,
                "action":"PASS_RESEARCH_ONLY"}
    result["paired_model_sha256"]=source_sha
    result["frozen_replay_sha256"]=hashlib.sha256(replay_bytes).hexdigest()
    result["dated_web_context_sha256"]=hashlib.sha256(web_bytes).hexdigest()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    data=json.dumps(result,indent=2,sort_keys=True)+"\n"
    args.out.write_text(data,encoding="utf-8")
    print(json.dumps({"status":result["status"],
      "scenario_count":len(result["scenarios"]),
      "blocked_count":len(result.get("blocked",[])),
      "output":str(args.out),"wager_authorized":False,
      "research_sha256":hashlib.sha256(data.encode()).hexdigest()}))

if __name__=="__main__":
    main()
