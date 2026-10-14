"""Rank exact offered contracts against original coupled research distributions.

Raw conditional arithmetic is research only. This adapter cannot promote the
currently unvalidated simulator, execute wagers, or invent missing prop prices.
"""
from __future__ import annotations
import argparse
from datetime import UTC, datetime, timedelta
import json
import math
from pathlib import Path
from nfl.coupled_drive_player_shadow import simulate_joint

MARKETS = ('moneyline', 'spread', 'total')
PROPS = ('passing_yards', 'passing_attempts', 'passing_completions', 'passing_touchdowns',
         'passing_interceptions', 'receiving_yards', 'receptions',
         'receiving_touchdowns', 'rushing_yards', 'carries', 'rushing_touchdowns')


def stamp(value):
    result = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('timestamp requires timezone')
    return result.astimezone(UTC)


def number(value):
    if isinstance(value, bool):
        raise ValueError('boolean is not a number')
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('nonfinite number')
    return result


def contracts(entry):
    """Expand actual same-book pairs; reject mismatched lines and missing prices."""
    for q in entry.get('quotes', []):
        market = q.get('market')
        if market not in MARKETS:
            continue
        try:
            a, b = q['first_side'], q['second_side']
            if {a, b} != ({'over', 'under'} if market == 'total' else {'home', 'away'}):
                raise ValueError('noncomplementary sides')
            lines = [q['first_line'], q['second_line']]
            if market == 'moneyline':
                if lines != [None, None]:
                    raise ValueError('moneyline has point line')
            else:
                lines = [number(x) for x in lines]
                if (lines[0] != lines[1] if market == 'total' else lines[0] != -lines[1]):
                    raise ValueError('mismatched lines')
            prices = [number(q['first_american_odds']), number(q['second_american_odds'])]
            if any(abs(x) < 100 for x in prices):
                raise ValueError('invalid American odds')
            book = q['book_as_attributed_by_aggregator']
            if not book:
                raise ValueError('missing book')
        except (KeyError, TypeError, ValueError):
            yield {'error': 'INVALID_PAIRED_CONTRACT'}
            continue
        for index, (side, line, price) in enumerate(zip((a, b), lines, prices)):
            yield dict(game_id=entry['event_id'], market=market, side=side,
                       line=line, american_odds=price, opposite_american_odds=prices[1-index], book=book,
                       observed_utc=entry['collector_observed_utc'],
                       source_url=entry['source_url'],
                       offer_origin_utc=q.get('source_original_book_offer_update_utc'),
                       offer_origin_verified=q.get('offer_origin_proven') is True,
                       executable_price_verified=False, settlement_verified=False,
                       snapshot_sha256=entry.get('snapshot_sha256'))


def query(contract):
    market, side, line = contract['market'], contract['side'], contract['line']
    if market == 'moneyline':
        return ('HOME_MARGIN', 0.0, 'over' if side == 'home' else 'under')
    if market == 'spread':
        return ('HOME_MARGIN', -line if side == 'home' else line,
                'over' if side == 'home' else 'under')
    if market == 'total':
        return ('TOTAL', line, side)
    if market not in PROPS or side not in ('over', 'under'):
        raise ValueError('unsupported prop contract')
    return (f"{contract['team']}:{contract['player_id']}:{market}", line, side)


def build_card(scan, replay, games, observations, *, asof, prop_contracts=(), simulator=simulate_joint):
    at = stamp(asof)
    originals = {g['game_id']: g for g in scan['games']}
    inputs = {g['game_id']: g for g in replay['games']}
    if stamp(scan['capture_utc']) != stamp(replay['capture_utc']):
        raise ValueError('forecast/replay capture mismatch')
    if len({g['game_id'] for g in games}) != len(games):
        raise ValueError('duplicate target game')
    latest = {}
    errors = []
    displayed_references = []
    for entry in observations:
        try:
            observed = stamp(entry['collector_observed_utc'])
            if observed <= at and (entry['event_id'] not in latest or
                observed > stamp(latest[entry['event_id']]['collector_observed_utc'])):
                latest[entry['event_id']] = entry
        except (KeyError, ValueError):
            errors.append({'reason': 'INVALID_OBSERVATION_TIME'})
    board, candidates = [], []
    for game in games:
        gid = game['game_id']
        if at >= stamp(game['kickoff_utc']):
            errors.append({'game_id': gid, 'reason': 'GAME_ALREADY_STARTED'})
            continue
        row = {**game, 'decision': 'PASS', 'best_candidate': None,
               'priced_contracts': 0, 'evaluated_contracts': 0,
               'missing_game_markets': list(MARKETS), 'prop_price_coverage': 'NONE',
               'status': 'NO_USABLE_PRICES', 'errors': []}
        board.append(row)
        entry = latest.get(gid)
        offered = list(contracts(entry)) if entry else []
        offered += [dict(p) for p in prop_contracts if p.get('game_id') == gid]
        valid = []
        for c in offered:
            try:
                if c.get('error'):
                    raise ValueError(c['error'])
                observed = stamp(c['observed_utc'])
                if not at-timedelta(minutes=15) <= observed <= at < stamp(game['kickoff_utc']):
                    raise ValueError('STALE_OR_FUTURE_PRICE_OBSERVATION')
                if observed >= stamp(game['kickoff_utc']):
                    raise ValueError('POSTKICKOFF_PRICE')
                if entry and c['market'] in MARKETS and stamp(entry['kickoff_utc']) != stamp(game['kickoff_utc']):
                    raise ValueError('QUOTE_SCHEDULE_IDENTITY_MISMATCH')
                percent_reference = c.get('quote_format') == 'DISPLAYED_PERCENT_REFERENCE'
                price = number(c['displayed_probability'] if percent_reference else c['american_odds'])
                price_valid = 0 < price < 1 if percent_reference else abs(price) >= 100
                if not price_valid or not c.get('book') or not c.get('source_url', '').startswith('https://'):
                    raise ValueError('INCOMPLETE_EXACT_PRICE')
                if c['market'] != 'moneyline':
                    c['line'] = number(c['line'])
                metric, threshold, side = query(c)
                valid.append((c, metric, threshold, side))
            except (KeyError, ValueError, TypeError) as exc:
                row['errors'].append(str(exc))
        row['priced_contracts'] = sum(c.get('quote_format') != 'DISPLAYED_PERCENT_REFERENCE' for c, *_ in valid)
        row['displayed_reference_contracts'] = len(valid) - row['priced_contracts']
        row['missing_game_markets'] = [m for m in MARKETS if not any(c['market'] == m for c, *_ in valid)]
        actual_props = any(c['market'] in PROPS and c.get('quote_format') != 'DISPLAYED_PERCENT_REFERENCE' for c, *_ in valid)
        reference_props = any(c['market'] in PROPS and c.get('quote_format') == 'DISPLAYED_PERCENT_REFERENCE' for c, *_ in valid)
        row['prop_price_coverage'] = 'PARTIAL' if actual_props else ('DISPLAYED_REFERENCE_ONLY' if reference_props else 'NONE')
        if not valid:
            continue
        original, source = originals.get(gid), inputs.get(gid)
        if not original or not source or original.get('status') != 'UNVALIDATED_JOINT_DRIVE_PLAYER_RESEARCH':
            row['status'] = 'MODEL_OR_SOURCE_MISSING'
            continue
        try:
            if source['forecast']['event_id'] != gid or any(source['forecast'][k] != game[k] for k in ('home_team', 'away_team')):
                raise ValueError('FORECAST_SCHEDULE_IDENTITY_MISMATCH')
            if stamp(source['forecast']['kickoff_utc']) != stamp(game['kickoff_utc']):
                raise ValueError('FORECAST_KICKOFF_MISMATCH')
            if not stamp(scan['capture_utc']) <= stamp(original['asof_utc']) < stamp(game['kickoff_utc']):
                raise ValueError('INVALID_FORECAST_CHRONOLOGY')
            if stamp(original['asof_utc']) > at:
                raise ValueError('FUTURE_FORECAST')
            supported = [(c, m, l, s) for c, m, l, s in valid if m in original['distribution']]
            row['errors'] += ['NO_MODEL_DISTRIBUTION:'+m for _, m, _, _ in valid if m not in original['distribution']]
            keys = sorted({(m, l) for _, m, l, _ in supported})
            result = simulator(source['forecast'], source['drives'], source['roles'],
                               asof=stamp(original['asof_utc']), draws=original['draws'], seed=original['seed'],
                               queries=[{'metric': m, 'line': l} for m, l in keys])
            if result['forecast_sha256'] != original['forecast_sha256'] or result['distribution'] != original['distribution']:
                raise ValueError('REPLAY_DOES_NOT_MATCH_ORIGINAL_FORECAST')
            probabilities = {(p['metric'], p['line']): p for p in result['query_thresholds']}
            game_candidates = []
            game_references = []
            for c, metric, threshold, side in supported:
                p = probabilities[(metric, threshold)]
                win, push = number(p[side]), number(p['push'])
                loss = number(p['under' if side == 'over' else 'over'])
                if min(win, push, loss) < 0 or abs(win+push+loss-1) > 1e-8:
                    raise ValueError('INVALID_PROBABILITY_MASS')
                if c.get('quote_format') == 'DISPLAYED_PERCENT_REFERENCE':
                    game_references.append({**c, 'model_capture_utc': original['asof_utc'],
                        'model_forecast_sha256': original['forecast_sha256'],
                        'raw_conditional_win_probability': win, 'raw_push_probability': push,
                        'raw_loss_probability': loss,
                        'raw_probability_minus_displayed_reference': win-number(c['displayed_probability']),
                        'raw_conditional_ev_proxy': None, 'estimated_validated_ev': None,
                        'decision': 'PASS', 'qualification': 'DISPLAYED_REFERENCE_ONLY',
                        'blockers': ['MODEL_UNCALIBRATED', 'DISPLAYED_QUOTE_NOT_EXECUTABLE_PRICE',
                                     'FEES_BID_ASK_AND_SETTLEMENT_UNKNOWN', 'PLAYER_PARTICIPATION_UNVERIFIED']})
                    row['evaluated_contracts'] += 1
                    continue
                odds = number(c['american_odds'])
                payout = odds/100 if odds > 0 else 100/-odds
                candidate = {**c, 'model_capture_utc': original['asof_utc'],
                             'model_forecast_sha256': original['forecast_sha256'],
                             'raw_conditional_win_probability': win, 'raw_push_probability': push,
                             'raw_loss_probability': loss, 'price_break_even_excluding_push': 1/(1+payout),
                             'raw_conditional_ev_proxy': win*payout-loss,
                             'calibrated_probability': None, 'estimated_validated_ev': None,
                             'confidence': 'UNVALIDATED', 'qualification': 'RESEARCH_CANDIDATE',
                             'decision': 'PASS', 'blockers': ['MODEL_MARKET_EDGE_UNVALIDATED',
                             'CALIBRATION_UNCERTAINTY_UNKNOWN', 'OT_AND_SETTLEMENT_UNVERIFIED']}
                if not c.get('offer_origin_verified'):
                    candidate['blockers'].append('BOOK_ORIGIN_UNVERIFIED')
                if not c.get('executable_price_verified'):
                    candidate['blockers'].append('EXECUTABLE_PRICE_UNVERIFIED')
                if c['market'] in PROPS:
                    candidate['blockers'].append('PLAYER_PARTICIPATION_UNVERIFIED')
                game_candidates.append(candidate)
                row['evaluated_contracts'] += 1
            candidates.extend(game_candidates)
            displayed_references.extend(game_references)
            row['status'] = 'RESEARCH_EVALUATED'
        except (ValueError, KeyError, TypeError, RuntimeError) as exc:
            row['evaluated_contracts'] = 0
            row['status'] = 'MODEL_REPLAY_BLOCKED'
            row['errors'].append(str(exc))
    candidates.sort(key=lambda c: (-c['raw_conditional_ev_proxy'], c['game_id'], c['market'], str(c.get('player_id')), c['side'], c['book']))
    for rank, c in enumerate(candidates, 1):
        c['research_rank'] = rank
    for row in board:
        row['best_candidate'] = next((c for c in candidates if c['game_id'] == row['game_id']), None)
        row['best_game_market_candidate'] = next((c for c in candidates if c['game_id'] == row['game_id'] and c['market'] in MARKETS), None)
    return {'spec': 'NFL_FULL_SLATE_CARD_V1', 'generated_utc': at.isoformat(),
            'ranking_basis': 'RAW_CONDITIONAL_EV_PROXY_NOT_VALIDATED_EDGE',
            'target_games': len(games), 'games_accounted_for': len(board),
            'qualified_bets': [], 'qualified_bet_count': 0,
            'positive_research_candidates': sum(c['raw_conditional_ev_proxy'] > 0 for c in candidates),
            'automatic_betting_enabled': False, 'games': board, 'ranked_candidates': candidates,
            'displayed_market_probability_comparisons': displayed_references,
            'source_errors': errors}


def markdown(card):
    lines = ['# NFL full-slate card', '', 'Research ranking uses unvalidated conditional EV proxies. Qualified bets: 0.', '',
             '| Game | Coverage | Best research candidate | Raw EV proxy | Decision |',
             '|---|---|---|---:|---|']
    for g in card['games']:
        c = g['best_candidate']
        label = f"{c['market']} {c.get('player_id') or ''} {c['side']} {c['line']} at {c['american_odds']:+g} ({c['book']})" if c else 'None'
        ev = f"{c['raw_conditional_ev_proxy']:.2%}" if c else '—'
        lines.append(f"| {g['away_team']} @ {g['home_team']} | {g['status']} | {label} | {ev} | PASS |")
    lines += ['', 'Prop prices are incomplete; unavailable contracts are never assigned synthetic prices.',
              'Exact prices, raw probabilities, capture times, provenance and blockers appear in the JSON card.']
    refs = card.get('displayed_market_probability_comparisons', [])
    if refs:
        lines += ['', '## Displayed market references — uncalibrated comparisons, EV unknown', '',
                  '| Player | Market/side | Line | Raw model | Displayed reference | Decision |',
                  '|---|---|---:|---:|---:|---|']
        for c in sorted(refs, key=lambda r: -abs(r['raw_probability_minus_displayed_reference']))[:20]:
            lines.append(f"| {c.get('player_name', c['player_id'])} | {c['market']} {c['side']} | {c['line']} | {c['raw_conditional_win_probability']:.1%} | {c['displayed_probability']:.1%} ({c['book']}) | PASS |")
    return '\n'.join(lines)+'\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--forecast', type=Path, default=Path('reports/coupled_drive_player_shadow.json'))
    p.add_argument('--replay', type=Path, default=Path('reports/coupled_replay_inputs.json'))
    p.add_argument('--capture-report', type=Path, default=Path('reports/weekly_public_quote_capture.json'))
    p.add_argument('--history', type=Path, default=Path('history/public_aggregator_quotes_2026.jsonl'))
    p.add_argument('--prop-contracts', type=Path, help='Optional JSON array of exact source-attributed player-ID contracts')
    p.add_argument('--out', type=Path, default=Path('reports/nfl_full_slate_card.json'))
    p.add_argument('--workload-sidecar', type=Path, default=Path('reports/game_state_workload_forward.json'))
    a = p.parse_args()
    load = lambda path: json.loads(path.read_text())
    capture = load(a.capture_report)
    card = build_card(load(a.forecast), load(a.replay), capture['target_game_rows'],
                      [json.loads(s) for s in a.history.read_text().splitlines() if s.strip()] if a.history.exists() else [],
                      asof=datetime.now(UTC).isoformat(), prop_contracts=load(a.prop_contracts) if a.prop_contracts else [])
    card['source_errors'] += capture.get('rejections', [])
    card['game_state_workload_research'] = {'status': 'SIDECAR_UNAVAILABLE', 'used_for_bet_ranking': False, 'promotion_earned': False}
    if a.workload_sidecar.exists():
        workload = load(a.workload_sidecar)
        if workload.get('spec') != 'NFL_QUARTER_STATE_WORKLOAD_RESEARCH_V1' or stamp(workload['captured_utc']) > stamp(card['generated_utc']):
            raise ValueError('INVALID_WORKLOAD_SIDECAR_PROVENANCE')
        card['game_state_workload_research'].update(status=workload['status'], captured_utc=workload['captured_utc'], model_file_sha256=workload.get('model_file_sha256'), errors=workload.get('errors', []))
        for game in card['games']:
            game['workload_research'] = [r for r in workload.get('forecasts', []) if r['game_id'] == game['game_id'] and r['team'] in (game['home_team'], game['away_team']) and stamp(r['kickoff_utc']) == stamp(game['kickoff_utc'])]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(card, indent=2, sort_keys=True, allow_nan=False)+'\n')
    a.out.with_suffix('.md').write_text(markdown(card))
    print(json.dumps({k: card[k] for k in ('target_games', 'games_accounted_for', 'positive_research_candidates', 'qualified_bet_count')}))

if __name__ == '__main__':
    main()
