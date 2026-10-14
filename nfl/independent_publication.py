"""Publish coherent independent score/stat forecasts, never wagering authority.

This is a read-only view of the existing coupled simulation. Its scores and
player means come from the same draws; it neither trains on prices nor claims
calibrated final NFL/OT probabilities or market superiority.
"""
from __future__ import annotations

import csv
import html
import json
from datetime import UTC, datetime
from pathlib import Path

from nfl.coupled_forward_grading import clock, num

PROP_METRICS = (
    'passing_attempts', 'passing_completions', 'passing_yards',
    'passing_touchdowns', 'passing_interceptions', 'carries', 'rushing_yards',
    'rushing_touchdowns', 'receptions', 'receiving_yards', 'receiving_touchdowns',
)
LABELS = ('Pass att', 'Pass cmp', 'Pass yd', 'Pass TD', 'INT', 'Carries',
          'Rush yd', 'Rush TD', 'Rec', 'Rec yd', 'Rec TD')


def distribution(info):
    out = {k: num(info[k]) for k in ('mean', 'p10', 'median', 'p90')}
    if not out['p10'] <= out['median'] <= out['p90']:
        raise ValueError('INVALID_SIMULATION_QUANTILES')
    return out


def build_bundle(scan, replay, *, at, names=None):
    if at.tzinfo is None:
        raise ValueError('UNZONED_PUBLICATION_CLOCK')
    if scan.get('spec') != 'NFL_COUPLED_DRIVE_PLAYER_SHADOW_SCAN_V1' or replay.get('spec') != 'NFL_COUPLED_FROZEN_REPLAY_INPUTS_V1':
        raise ValueError('WRONG_INDEPENDENT_FORECAST_FAMILY')
    if scan['capture_utc'] != replay['capture_utc'] or clock(scan['capture_utc']) > at:
        raise ValueError('SOURCE_CAPTURE_MISMATCH_OR_FUTURE')
    names = names or {}
    inputs = {}
    for r in replay['games']:
        if r['game_id'] in inputs:
            raise ValueError('DUPLICATE_REPLAY_GAME')
        inputs[r['game_id']] = r
    games, players, rejected, seen = [], [], [], set()
    for raw in scan['games']:
        gid = raw['game_id']
        if gid in seen:
            raise ValueError('DUPLICATE_FORECAST_GAME')
        seen.add(gid)
        if raw.get('status') != 'UNVALIDATED_JOINT_DRIVE_PLAYER_RESEARCH':
            games.append(dict(game_id=gid, status='SOURCE_OR_MODEL_BLOCKED',
                              reason=raw.get('reason'), predicted_score=None,
                              fair_home_spread=None, predicted_total=None))
            continue
        try:
            source = inputs[gid]
            f = source['forecast']
            if f != raw['frozen_football_score_input'] or f['event_id'] != gid or f['content_sha256'] != raw['forecast_sha256']:
                raise ValueError('ORIGINAL_FORECAST_IDENTITY_MISMATCH')
            if not clock(f['captured_utc']) <= clock(scan['capture_utc']) <= clock(raw['asof_utc']) <= at < clock(f['kickoff_utc']):
                raise ValueError('NOT_CURRENT_PREKICKOFF_FORECAST')
            home, away = f['home_team'], f['away_team']
            if home == away:
                raise ValueError('INVALID_MATCHUP')
            d = raw['distribution']
            hs, aws = distribution(d[home+':points']), distribution(d[away+':points'])
            total, margin = distribution(d['TOTAL']), distribution(d['HOME_MARGIN'])
            if abs(hs['mean']+aws['mean']-total['mean']) > 1e-7 or abs(hs['mean']-aws['mean']-margin['mean']) > 1e-7:
                raise ValueError('SIMULATED_SCORE_TOTAL_MARGIN_DO_NOT_RECONCILE')
            probabilities = {k: num(raw[k]) for k in ('home_win', 'away_win', 'tie')}
            if min(probabilities.values()) < 0 or abs(sum(probabilities.values())-1) > 1e-8:
                raise ValueError('INVALID_RAW_WIN_PROBABILITIES')
            roles = {}
            for role in source['roles']:
                key = (role['team'], role['player_id'])
                if key in roles:
                    raise ValueError('DUPLICATE_PLAYER_ROLE_IDENTITY')
                roles[key] = role
            production = {}
            for metric, info in sorted(d.items()):
                parts = metric.split(':')
                if len(parts) != 3 or parts[2] not in PROP_METRICS:
                    continue
                team, pid, stat = parts
                if team not in (home, away) or (pid != 'OTHER' and (team, pid) not in roles):
                    raise ValueError('PLAYER_DISTRIBUTION_ROLE_MISMATCH')
                production.setdefault((team, pid), {})[stat] = distribution(info)
            for team in (home, away):
                # OTHER remains visible so named-player totals are not overstated.
                for stat, aggregate_stat in (
                    ('receptions', 'completions'), ('receiving_yards', 'passing_yards'),
                    ('carries', 'rush_attempts'), ('rushing_yards', 'rushing_yards'),
                ):
                    expected = num(d[f'{team}:{aggregate_stat}']['mean'])
                    actual = sum(p[stat]['mean'] for (t, _), p in production.items() if t == team and stat in p)
                    if abs(expected-actual) > 1e-7:
                        raise ValueError('PLAYER_TEAM_PRODUCTION_DOES_NOT_RECONCILE:'+stat)
            game_players = []
            for (team, pid), stats in sorted(production.items()):
                role = roles.get((team, pid), {})
                game_players.append(dict(game_id=gid, team=team, player_id=pid,
                    player_name='Unallocated production' if pid == 'OTHER' else names.get(pid, pid),
                    position=role.get('role'), is_unallocated_bucket=pid == 'OTHER',
                    participation='UNVERIFIED_HISTORICAL_ROLE_PROXY',
                    projections=stats, unavailable_metrics=[s for s in PROP_METRICS if s not in stats],
                    calibrated_prop_probability=None, sportsbook_line=None, estimated_ev=None))
            games.append(dict(game_id=gid, season=f['season'], week=f['week'],
                home_team=home, away_team=away, kickoff_utc=f['kickoff_utc'],
                status='INDEPENDENT_CONDITIONAL_FORECAST',
                predicted_score=dict(home=hs['mean'], away=aws['mean']),
                predicted_home_margin=margin['mean'], fair_home_spread=-margin['mean'],
                fair_away_spread=margin['mean'], predicted_total=total['mean'],
                distributions=dict(home_score=hs, away_score=aws, total=total, home_margin=margin),
                raw_simulation_probabilities=probabilities,
                independent_power_rating_baseline=dict(home_points=f['independent_home_points'],
                    away_points=f['independent_away_points']),
                player_rows=len(game_players), forecast_sha256=raw['forecast_sha256'],
                model_capture_utc=raw['asof_utc'], draws=raw['draws'], seed=raw['seed']))
            players.extend(game_players)
        except (ValueError, KeyError, TypeError) as exc:
            rejected.append(dict(game_id=gid, reason=str(exc)))
            games.append(dict(game_id=gid, status='PUBLICATION_INTEGRITY_BLOCKED',
                              reason=str(exc), predicted_score=None,
                              fair_home_spread=None, predicted_total=None))
    return dict(spec='NFL_INDEPENDENT_FORECAST_PUBLICATION_V1',
        generated_utc=at.astimezone(UTC).isoformat(), model_capture_utc=scan['capture_utc'],
        season=scan['season'], week=scan['week'], games=games, player_props=players,
        games_requested=len(games), games_forecast=sum(g['predicted_score'] is not None for g in games),
        named_player_rows=sum(not p['is_unallocated_bucket'] for p in players),
        prop_metric_forecasts=sum(len(p['projections']) for p in players if not p['is_unallocated_bucket']),
        source_errors=rejected, sportsbook_prices_used=False,
        accuracy_vs_market='UNPROVEN', calibrated_intervals=False, overtime_validated=False,
        spread_definition='NEGATIVE_EXPECTED_HOME_MARGIN_NOT_A_CALIBRATED_50_PERCENT_COVER_LINE',
        supported_player_metrics=list(PROP_METRICS),
        unsupported_markets=['field_goals', 'kicking_points', 'longest_play', 'defensive_props', 'first_scorer'],
        automatic_betting_enabled=False, qualified_bets=0,
        limits='Own conditional research forecasts. Historical roles are not confirmed active lineups. Raw p10/p90 are simulation percentiles, not calibrated intervals. Full-game overtime/unusual scoring and market superiority remain unvalidated.')


def markdown(bundle):
    lines = ['# Our NFL predictions', '',
        'Independent score and player forecasts; sportsbook prices are not inputs. Accuracy versus the market: UNPROVEN.',
        '', bundle['limits'], '',
        '| Game | Our away score | Our home score | Our home spread | Our total |',
        '|---|---:|---:|---:|---:|']
    for g in bundle['games']:
        if g['predicted_score'] is None:
            lines.append(f"| {g['game_id']} — {g['status']} | — | — | — | — |")
        else:
            lines.append(f"| {g['away_team']} @ {g['home_team']} | {g['predicted_score']['away']:.1f} | {g['predicted_score']['home']:.1f} | {g['fair_home_spread']:+.1f} | {g['predicted_total']:.1f} |")
    for g in bundle['games']:
        ps = [p for p in bundle['player_props'] if p['game_id'] == g['game_id'] and not p['is_unallocated_bucket']]
        if not ps:
            continue
        lines += ['', '## '+g['away_team']+' @ '+g['home_team']+' — player projections', '',
                  '| Player | Team | '+' | '.join(LABELS)+' |',
                  '|---|---|'+'---:|'*len(LABELS)]
        for p in ps:
            vals = [f"{p['projections'][s]['mean']:.1f}" if s in p['projections'] else '—' for s in PROP_METRICS]
            lines.append('| '+p['player_name'].replace('|', '/')+' | '+p['team']+' | '+' | '.join(vals)+' |')
    return '\n'.join(lines)+'\n'


def write_bundle(bundle, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bundle, indent=2, sort_keys=True, allow_nan=False)+'\n')
    out.with_suffix('.md').write_text(markdown(bundle))
    fields = ('game_id','away_team','home_team','away_score','home_score','fair_home_spread','predicted_total','status')
    with out.with_name('independent_game_predictions.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for g in bundle['games']:
            row = {k: g.get(k) for k in fields}
            row['away_score'] = (g['predicted_score'] or {}).get('away')
            row['home_score'] = (g['predicted_score'] or {}).get('home')
            w.writerow(row)
    with out.with_name('independent_player_predictions.csv').open('w', newline='') as f:
        fields = ('game_id','team','player_id','player_name','position','metric','mean','p10','median','p90','participation','is_unallocated_bucket')
        w = csv.DictWriter(f, fieldnames=fields);w.writeheader()
        for p in bundle['player_props']:
            for metric, values in p['projections'].items():
                w.writerow({**{k:p.get(k) for k in fields if k in p},'metric':metric,**values})
    # Exact numeric HTML, not an image-derived approximation or odds board.
    parts = ['<!doctype html><meta charset="utf-8"><title>Our NFL predictions</title>',
             '<style>body{background:#101216;color:#eef1f6;font:15px Arial;margin:24px}table{border-collapse:collapse;width:100%;margin-bottom:24px}th,td{padding:8px;border-bottom:1px solid #343943;text-align:right}th:first-child,td:first-child{text-align:left}.note{color:#d6b46a}</style>',
             '<h1>Our independent NFL predictions</h1><p class="note">'+html.escape(bundle['limits'])+'</p>',
             '<table><tr><th>Game</th><th>Away score</th><th>Home score</th><th>Home spread</th><th>Total</th></tr>']
    for g in bundle['games']:
        if g['predicted_score'] is not None:
            parts.append('<tr><td>'+html.escape(g['away_team']+' @ '+g['home_team'])+'</td>'+''.join(f'<td>{v:.1f}</td>' for v in (g['predicted_score']['away'],g['predicted_score']['home'],g['fair_home_spread'],g['predicted_total']))+'</tr>')
        else:
            parts.append('<tr><td>'+html.escape(g['game_id']+' '+g['status'])+'</td><td colspan="4">Unavailable</td></tr>')
    parts.append('</table>')
    for g in bundle['games']:
        ps = [p for p in bundle['player_props'] if p['game_id']==g['game_id'] and not p['is_unallocated_bucket']]
        if not ps:
            continue
        parts.append('<h2>'+html.escape(g['away_team']+' @ '+g['home_team'])+'</h2><table><tr><th>Player</th>'+''.join('<th>'+html.escape(v)+'</th>' for v in LABELS)+'</tr>')
        for p in ps:
            parts.append('<tr><td>'+html.escape(p['player_name']+' ('+p['team']+')')+'</td>'+''.join('<td>'+ (f"{p['projections'][s]['mean']:.1f}" if s in p['projections'] else '—')+'</td>' for s in PROP_METRICS)+'</tr>')
        parts.append('</table>')
    out.with_suffix('.html').write_text(''.join(parts))
    write_score_image(bundle, out.with_suffix('.png'))
    return dict(json=str(out),markdown=str(out.with_suffix('.md')),html=str(out.with_suffix('.html')),png=str(out.with_suffix('.png')))


def write_score_image(bundle, path):
    from PIL import Image, ImageDraw, ImageFont
    def font(size):
        try:
            return ImageFont.truetype('DejaVuSans.ttf',size)
        except OSError:
            return ImageFont.load_default()
    width, header, row_height = 1240, 145, 36
    img = Image.new('RGB',(width,header+row_height*max(1,len(bundle['games']))+40),'#101216')
    draw = ImageDraw.Draw(img)
    draw.text((20,16),f"OUR NFL PREDICTIONS — WEEK {bundle['week']}",font=font(24),fill='#eef1f6')
    draw.text((20,52),'Independent model forecasts. No sportsbook prices used. Market superiority UNPROVEN.',font=font(14),fill='#d6b46a')
    draw.text((20,78),'Conditional simulations; active lineups and overtime unvalidated. Player forecasts are in the HTML/CSV files.',font=font(12),fill='#9ca7b9')
    xs = (20,380,560,740,940)
    for x,label in zip(xs,('GAME','OUR AWAY SCORE','OUR HOME SCORE','OUR HOME SPREAD','OUR TOTAL')):
        draw.text((x,115),label,font=font(13),fill='#9ca7b9')
    for i,g in enumerate(bundle['games']):
        y=header+i*row_height
        if i%2==0:
            draw.rectangle((12,y-4,width-12,y+row_height-5),fill='#191d24')
        if g['predicted_score'] is None:
            labels=(g['game_id'],'UNAVAILABLE','UNAVAILABLE','—','—')
        else:
            labels=(g['away_team']+' @ '+g['home_team'],f"{g['predicted_score']['away']:.1f}",
                    f"{g['predicted_score']['home']:.1f}",f"{g['fair_home_spread']:+.1f}",f"{g['predicted_total']:.1f}")
        for x,label in zip(xs,labels):
            draw.text((x,y+3),label,font=font(16),fill='#eef1f6')
    img.save(path)
