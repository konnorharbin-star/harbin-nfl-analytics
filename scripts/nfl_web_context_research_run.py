"""Run a read-only public web-context check alongside a frozen NFL scan."""
import argparse,hashlib,json
from datetime import UTC,datetime
from pathlib import Path
import nflreadpy as nfl
from nfl.web_context_research import collect_web_context
from nfl.player_evidence_bridge import review_player_evidence

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
    # Extra free nflverse source: player GSIS<->ESPN ID map. A missing
    # directory does not invalidate earlier score research, but disables
    # participant-linked injury evidence instead of guessing by name.
    try:
        directory=nfl.load_players().to_dicts()
        report["player_evidence_review"]=review_player_evidence(
            source,report,directory,at=at)
    except Exception as error:  # Optional external crosswalk is fail-closed.
        report["source_errors"].append({
            "source":"NFLVERSE_ESPN_PLAYER_ID_CROSSWALK",
            "error":type(error).__name__+":"+str(error)[:180]})
        report["player_evidence_review"]={
            "status":"PLAYER_ID_SOURCE_NOT_VERIFIED",
            "official_week_specific_inactives_verified":False,
            "model_fair_score_or_prop_adjusted":False,
            "wager_authorized":False}
    report["paired_original_model_capture_utc"]=source["capture_utc"]
    report["paired_model_artifact_sha256"]=hashlib.sha256(raw).hexdigest()
    a.out.parent.mkdir(parents=True,exist_ok=True)
    blob=json.dumps(report,indent=2,sort_keys=True,default=str)+"\n"
    a.out.write_text(blob,encoding="utf-8")
    print(json.dumps({"status":report["status"],"games":len(report["games"]),
      "source_errors":len(report["source_errors"]),
      "weather_available":sum(r["weather"].get("status")=="ASOF_HOURLY_FORECAST_CAPTURED" for r in report["games"]),
      "model_changed":False,"wager_authorized":False,
      "player_evidence":report["player_evidence_review"]["status"],
      "artifact_sha256":hashlib.sha256(blob.encode()).hexdigest()}))

if __name__=="__main__":
    main()
