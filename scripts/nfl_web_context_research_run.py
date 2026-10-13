"""Run a read-only public web-context check alongside a frozen NFL scan."""
import argparse,hashlib,json
from datetime import UTC,datetime
from pathlib import Path
import nflreadpy as nfl
from nfl.web_context_research import collect_web_context

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--forecast-scan",type=Path,required=True)
    p.add_argument("--out",type=Path,default=Path("reports/nfl_web_context_research.json"))
    a=p.parse_args()
    raw=a.forecast_scan.read_bytes()
    source=json.loads(raw)
    if source.get("spec")!="NFL_COUPLED_DRIVE_PLAYER_SHADOW_SCAN_V1":
        raise ValueError("UNSUPPORTED_ORIGINAL_SCAN")
    at=datetime.now(UTC)
    if datetime.fromisoformat(source["capture_utc"])>at:
        raise ValueError("FORECAST_CAPTURE_IN_FUTURE")
    ids=[r.get("game_id") for r in source["games"]]
    if not ids or len(set(ids))!=len(ids):
        raise ValueError("NO_VALID_ORIGINAL_GAMES")
    season=int(source["season"])
    schedule=nfl.load_schedules(season)
    games=[g for g in schedule.to_dicts() if g.get("game_id") in ids
           and g.get("season")==season and g.get("week")==source["week"]]
    if len(games)!=len(ids):
        raise ValueError("SCHEDULE_ID_MISMATCH")
    report=collect_web_context(games,at=at)
    report["paired_original_model_capture_utc"]=source["capture_utc"]
    report["paired_model_artifact_sha256"]=hashlib.sha256(raw).hexdigest()
    a.out.parent.mkdir(parents=True,exist_ok=True)
    blob=json.dumps(report,indent=2,sort_keys=True,default=str)+"\n"
    a.out.write_text(blob,encoding="utf-8")
    print(json.dumps({"status":report["status"],"games":len(report["games"]),
      "source_errors":len(report["source_errors"]),
      "weather_available":sum(r["weather"].get("status")=="ASOF_HOURLY_FORECAST_CAPTURED" for r in report["games"]),
      "model_changed":False,"wager_authorized":False,
      "artifact_sha256":hashlib.sha256(blob.encode()).hexdigest()}))

if __name__=="__main__":
    main()
