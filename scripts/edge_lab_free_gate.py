"""Fail-closed, zero-secret presentation of an independent pregame NFL score projection.

This is a research report; it cannot approve bets, account odds, or fitted positive EV.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path

MODEL_TAG = 'harbin-nfl-independent-score-exploratory'


def finite_number(value):
    if isinstance(value, bool):
        return None
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (ValueError, TypeError):
        return None


def build_report(projection, game_id, *, captured_utc=None, source_sha=None):
    timestamp = captured_utc or datetime.now(timezone.utc).isoformat()
    report = {
        'report_version': 'edge-lab-free-pregame-v1',
        'recorded_utc': timestamp,
        'source_repository_sha': source_sha,
        'game_id': game_id,
        'engine_report_status': projection.get('status'),
        'decision': 'PASS',
        'validation': 'UNVALIDATED_FOR_BETMGM_EV',
        'execution_quote_verified': False,
        'bet_authorized': False,
        'estimated_ev': None,
        'model_tag': MODEL_TAG,
        'source_urls': [
            'https://github.com/nflverse/nflverse-data/releases/tag/pbp',
            'https://github.com/konnorharbin-star/harbin-nfl-analytics',
        ],
        'blockers': ['No timestamped account-executable U.S. BetMGM price',
                     'No independent calibrated and prospectively validated EV model',
                     'Final QB/inactive/weather context must be confirmed as-of decision'],
        'projection': None,
    }
    if projection.get('status') != 'READY':
        report['status'] = 'BLOCKED'
        report['blockers'].append('Independent projection engine returned BLOCKED: ' +
                                  str(projection.get('reason', 'unknown')))
        return report
    matching = [x for x in projection.get('games', []) if x.get('game_id') == game_id]
    if len(matching) != 1:
        report['status'] = 'BLOCKED'
        report['blockers'].append(f'Expected exactly one game match; found {len(matching)}')
        return report
    record = matching[0]
    # Modern current-projection engine emits baseline_*; earlier research CSVs emit model_*.
    # Never combine incompatible columns or invent a missing score.
    field_pair = ('baseline_home_margin', 'baseline_total') if (
        'baseline_home_margin' in record or 'baseline_total' in record
    ) else ('model_margin_home', 'model_total')
    margin = finite_number(record.get(field_pair[0]))
    total = finite_number(record.get(field_pair[1]))
    if margin is None or total is None:
        report['status'] = 'BLOCKED'
        report['blockers'].append('Score projection missing or nonfinite')
        return report
    report['status'] = 'RESEARCH_ONLY'
    report['projection'] = {
        'source_score_fields': list(field_pair),
        'home': record.get('home_team'),
        'away': record.get('away_team'),
        'home_margin_points': margin,
        'game_total_points': total,
        'model_implied_home_points': (total + margin) / 2,
        'model_implied_away_points': (total - margin) / 2,
        'home_qb_proxy': record.get('home_qb_proxy_name'),
        'away_qb_proxy': record.get('away_qb_proxy_name'),
        'note': 'Independent exploratory score mean, not a calibrated BetMGM spread/total probability.',
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--projection', type=Path, required=True)
    parser.add_argument('--game-id', default='2026_05_BUF_LA')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    projection = json.loads(args.projection.read_text(encoding='utf8'))
    output = build_report(projection, args.game_id, source_sha=os.getenv('GITHUB_SHA'))
    args.output.parent.mkdir(exist_ok=True, parents=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + '\n', encoding='utf8')
    print(json.dumps(output, indent=2, sort_keys=True))
    print('FROZEN DECISION: PASS. Research is not a bet recommendation.')


if __name__ == '__main__':
    main()
