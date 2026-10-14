"""Publish our own score, fair spread/total and player-stat forecast bundle."""
import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path

from nfl.independent_publication import build_bundle, write_bundle


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scan', type=Path, default=Path('reports/coupled_drive_player_shadow.json'))
    p.add_argument('--replay', type=Path, default=Path('reports/coupled_replay_inputs.json'))
    p.add_argument('--out', type=Path, default=Path('outputs/independent_forecasts.json'))
    p.add_argument('--resolve-names', action='store_true')
    a = p.parse_args()
    names, name_errors = {}, []
    if a.resolve_names:
        try:
            import nflreadpy as nfl
            rows = nfl.load_players().to_dicts()
            for r in rows:
                pid = r.get('gsis_id')
                name = r.get('display_name') or r.get('football_name') or r.get('full_name')
                if pid and name:
                    names[str(pid)] = str(name)
            if not names:
                raise ValueError('NO_SOURCE_PLAYER_NAMES_AVAILABLE')
        except Exception as exc:
            # Name lookup cannot change numbers or impersonate verified actives.
            name_errors.append('PLAYER_NAME_SOURCE_FAILED:'+type(exc).__name__)
    bundle = build_bundle(json.loads(a.scan.read_text()),json.loads(a.replay.read_text()),
                          at=datetime.now(UTC), names=names)
    bundle['source_errors'] += name_errors
    bundle['source_scan_sha256'] = hashlib.sha256(a.scan.read_bytes()).hexdigest()
    bundle['source_replay_sha256'] = hashlib.sha256(a.replay.read_bytes()).hexdigest()
    outputs = write_bundle(bundle, a.out)
    print(json.dumps(dict(games_requested=bundle['games_requested'],games_forecast=bundle['games_forecast'],
        named_players=bundle['named_player_rows'],prop_metric_forecasts=bundle['prop_metric_forecasts'],
        accuracy_vs_market=bundle['accuracy_vs_market'],outputs=outputs,source_errors=bundle['source_errors']),indent=2))

if __name__=='__main__':main()
