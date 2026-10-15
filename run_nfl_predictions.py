"""One command for independent NFL score/spread/total/player predictions."""
import argparse
from pathlib import Path
import subprocess
import sys


def generate(season, week=None, out_dir='outputs', quiet=False):
    root = Path(__file__).resolve().parent
    out = Path(out_dir)
    if not out.is_absolute():
        out = root/out
    cmd = [sys.executable,'-m','scripts.nfl_coupled_drive_player_research_run','--season',str(season),
           '--out',str(out/'independent_source_scan.json'),'--replay-inputs',str(out/'independent_source_replay.json')]
    if week is not None:
        cmd += ['--week',str(week)]
    stdout = subprocess.DEVNULL if quiet else None
    subprocess.run(cmd,cwd=root,check=True,stdout=stdout)
    subprocess.run([sys.executable,'-m','scripts.nfl_player_workload_upgrade',
                   '--scan',str(out/'independent_source_scan.json'),
                   '--replay',str(out/'independent_source_replay.json'),
                   '--out',str(out/'independent_workload_v2_scan.json'),
                   '--out-replay',str(out/'independent_workload_v2_replay.json')],
                   cwd=root,check=True,stdout=stdout)
    subprocess.run([sys.executable,'-m','scripts.nfl_independent_predictions',
                   '--scan',str(out/'independent_workload_v2_scan.json'),
                   '--replay',str(out/'independent_workload_v2_replay.json'),
                   '--out',str(out/'independent_forecasts.json'),'--resolve-names'],cwd=root,check=True,stdout=stdout)
    web_out = root/'reports' if out == root/'outputs' else out/'web_research'
    subprocess.run([sys.executable,'-m','scripts.nfl_web_model_intake',
                    '--predictions',str(out/'independent_forecasts.json'),
                    '--out',str(web_out/'nfl_web_model_intake.json'),
                    '--prop-contracts',str(web_out/'nfl_web_prop_contracts.json')],
                   cwd=root,check=True,stdout=stdout)
    return str(out/'independent_forecasts.json')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('season',type=int);p.add_argument('--week',type=int)
    p.add_argument('--out-dir',default='outputs');a=p.parse_args()
    generate(a.season,a.week,a.out_dir)

if __name__=='__main__':main()
