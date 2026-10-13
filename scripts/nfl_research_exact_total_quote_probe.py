"""Research-priced exact NFL full-game total from frozen before-kickoff inputs."""
import argparse,hashlib,json
from datetime import UTC,datetime
from pathlib import Path
from nfl.exact_total_public_price_probe import compare

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--forecasts",type=Path,required=True)
    p.add_argument("--replay",type=Path,required=True)
    p.add_argument("--quotes",type=Path,default=Path("research/public_nfl_total_quote_20261013_oct15.json"))
    p.add_argument("--out",type=Path,default=Path("reports/nfl_research_exact_total_quote_probe.json"))
    a=p.parse_args()
    original_bytes=a.forecasts.read_bytes()
    replay_bytes=a.replay.read_bytes()
    source_bytes=a.quotes.read_bytes()
    quote=json.loads(source_bytes)
    out=compare(json.loads(original_bytes),json.loads(replay_bytes),
                quote["observations"],now=datetime.now(UTC))
    out["frozen_model_artifact_sha256"]=hashlib.sha256(original_bytes).hexdigest()
    out["frozen_asof_replay_sha256"]=hashlib.sha256(replay_bytes).hexdigest()
    out["dated_public_quote_observation_sha256"]=hashlib.sha256(source_bytes).hexdigest()
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({"status":out["status"],"games_examined":out["games_examined"],
      "raw_models":[{"game":row["game_id"],"line":row["line"],
                     "over_raw_win_p":row["sides"]["over"]["model_raw_win_probability"],
                     "under_raw_win_p":row["sides"]["under"]["model_raw_win_probability"],
                     "raw_over_ev_proxy":row["sides"]["over"]["hypothetical_raw_model_ev_per_unit"],
                     "raw_under_ev_proxy":row["sides"]["under"]["hypothetical_raw_model_ev_per_unit"]}
                    for row in out["quote_pair_observations"]],
      "validated_profitable_edge":False,"betting_authorized":False,
      "output":str(a.out)},sort_keys=True))
if __name__=="__main__":main()
