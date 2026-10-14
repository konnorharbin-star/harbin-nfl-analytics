"""Freeze or grade exact full-slate research selections without wagering."""
import argparse
import hashlib
from datetime import UTC, datetime
import json
from pathlib import Path
from nfl.full_slate_validation import freeze, grade
from nfl.coupled_research_journal import read_journal
from nfl.coupled_forward_grading import clock


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('freeze', 'grade', 'check-pregame'))
    p.add_argument('--root', type=Path, default=Path('history/full_slate_selected_v1_2026'))
    p.add_argument('--card', type=Path, default=Path('reports/nfl_full_slate_card.json'))
    p.add_argument('--run-id')
    p.add_argument('--code-sha')
    p.add_argument('--out', type=Path, default=Path('reports/full_slate_validation.json'))
    a = p.parse_args()
    at = datetime.now(UTC)
    if a.mode == 'check-pregame':
        for row in read_journal(a.root/'primary.jsonl'):
            if row['source_run_id'] == a.run_id and at >= clock(row['kickoff_utc']):
                raise ValueError('NEW_SELECTED_CAPTURE_COMMIT_WOULD_BE_POSTKICKOFF')
        return
    if a.mode == 'freeze':
        files = ('nfl/coupled_drive_player_shadow.py', 'nfl/coupled_nflverse_asof.py', 'nfl/ratings.py', 'nfl/data.py', 'nfl/contracts.py', 'scripts/nfl_coupled_drive_player_research_run.py')
        model_sha = hashlib.sha256(b''.join(name.encode()+b'\0'+Path(name).read_bytes() for name in files)).hexdigest()
        result = freeze(a.root, json.loads(a.card.read_text()), at=at, run_id=a.run_id, code_sha=a.code_sha, model_sha=model_sha)
    else:
        import nflreadpy as nfl
        result = grade(a.root, nfl.load_schedules(2026).to_dicts(), at=at)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(json.dumps(result, sort_keys=True))

if __name__ == '__main__':
    main()
