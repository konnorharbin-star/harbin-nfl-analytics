"""First-seen exact-price candidate validation, research only and never wagers.

One selected game-market contract per game. Source-final threshold grading is
not authenticated bookmaker settlement. No ROI, CLV or promotion inferred.
"""
from collections import Counter
from datetime import timedelta
from pathlib import Path
import re

from nfl.coupled_forward_grading import clock, num
from nfl.coupled_nflverse_asof import kickoff_utc
from nfl.coupled_research_journal import append, digest, encoded, read_journal

SPEC = 'NFL_FULL_SLATE_SELECTED_FORWARD_V1'


def decimal(price):
    p = num(price)
    if abs(p) < 100:
        raise ValueError('INVALID_AMERICAN_PRICE')
    return 1 + (p/100 if p > 0 else 100/-p)


def freeze(root, card, *, at, run_id, code_sha, model_sha):
    if at.tzinfo is None:
        raise ValueError('UNZONED_CLOCK')
    if card.get('spec') != 'NFL_FULL_SLATE_CARD_V1':
        raise ValueError('WRONG_CARD_FAMILY')
    if not re.fullmatch(r'[0-9]+', str(run_id)) or not re.fullmatch(r'[0-9a-f]{40}', code_sha):
        raise ValueError('INVALID_RUN_PROVENANCE')
    if not re.fullmatch(r'[0-9a-f]{64}', model_sha):
        raise ValueError('INVALID_MODEL_FINGERPRINT')
    generated = clock(card['generated_utc'])
    if generated > at:
        raise ValueError('FUTURE_CARD')
    root = Path(root)
    original = read_journal(root/'primary.jsonl')
    if original and {r['model_implementation_sha256'] for r in original} != {model_sha}:
        raise ValueError('MODEL_CHANGED_NEW_COHORT_REQUIRED')
    seen = {r['game_id'] for r in original}
    if len(seen) != len(original):
        raise ValueError('DUPLICATE_PRIMARY_GAME')
    candidates, rejected, repeat = [], [], 0
    ids = set()
    for game in card['games']:
        gid = game['game_id']
        if gid in ids:
            raise ValueError('DUPLICATE_CARD_GAME')
        ids.add(gid)
        if gid in seen:
            repeat += 1
            continue
        # Preserve the original game-market selection population when public
        # player props are added to the informational card.
        c = game.get('best_game_market_candidate', game.get('best_candidate'))
        if c is None or num(c['raw_conditional_ev_proxy']) <= 0:
            rejected.append({'game_id': gid, 'reason': 'NO_POSITIVE_PRICED_CANDIDATE'})
            continue
        try:
            kickoff = clock(game['kickoff_utc'])
            if not clock(c['model_capture_utc']) <= clock(c['observed_utc']) <= generated <= at < kickoff:
                raise ValueError('INVALID_PREGAME_CAPTURE_CHRONOLOGY')
            if c['game_id'] != gid or c not in card['ranked_candidates']:
                raise ValueError('SELECTED_CONTRACT_NOT_IN_CARD')
            if c['market'] not in ('moneyline', 'spread', 'total'):
                raise ValueError('PLAYER_MARKET_GRADING_NOT_SUPPORTED')
            if c['side'] not in (('home', 'away') if c['market'] != 'total' else ('over', 'under')):
                raise ValueError('INVALID_SIDE')
            if c['market'] != 'moneyline':
                num(c['line'])
            elif c.get('line') is not None:
                raise ValueError('INVALID_MONEYLINE_LINE')
            decimal(c['american_odds'])
            if not c.get('book') or not c.get('source_url', '').startswith('https://') or not c.get('model_forecast_sha256'):
                raise ValueError('INCOMPLETE_CONTRACT_PROVENANCE')
            probs = [num(c[k]) for k in ('raw_conditional_win_probability', 'raw_push_probability', 'raw_loss_probability')]
            if min(probs) < 0 or abs(sum(probs)-1) > 1e-8 or probs[0]+probs[2] <= 0:
                raise ValueError('INVALID_PROBABILITY_MASS')
            model_conditional = probs[0]/(probs[0]+probs[2])
            market_conditional = None
            if c.get('opposite_american_odds') is not None:
                a = 1/decimal(c['american_odds'])
                b = 1/decimal(c['opposite_american_odds'])
                market_conditional = a/(a+b)
            candidates.append(dict(spec=SPEC, game_id=gid, home_team=game['home_team'],
                away_team=game['away_team'], kickoff_utc=game['kickoff_utc'],
                archived_utc=at.isoformat(), card_generated_utc=card['generated_utc'],
                source_run_id=str(run_id), source_code_sha=code_sha, model_implementation_sha256=model_sha,
                card_content_sha256=digest(encoded(card)), candidate=c,
                model_nonpush_probability=model_conditional,
                same_book_proportional_no_vig_nonpush_proxy=market_conditional,
                research_selection_rule='FIRST_POSITIVE_BEST_RAW_EV_CANDIDATE_PER_GAME_V1',
                independently_validated_edge=False, paper_bet=False, real_bet=False))
        except (ValueError, KeyError, TypeError) as exc:
            rejected.append({'game_id': gid, 'reason': str(exc)})
    append(root/'primary.jsonl', candidates)
    result = dict(spec=SPEC, source_run_id=str(run_id), recorded_utc=at.isoformat(),
                  new_primary_games=len(candidates), repeated_games=repeat,
                  rejected=rejected, independent_primary_games=len(original)+len(candidates),
                  qualified_bets=0, economic_roi=None, profitable_edge_validated=False)
    append(root/'captures.jsonl', [result])
    return result


def grade(root, finals, *, at):
    if at.tzinfo is None:
        raise ValueError('UNZONED_CLOCK')
    root = Path(root)
    primaries = read_journal(root/'primary.jsonl')
    if len({p['game_id'] for p in primaries}) != len(primaries):
        raise ValueError('DUPLICATE_PRIMARY_GAME')
    latest = {r['game_id']: r for r in read_journal(root/'grades.jsonl')}
    source = {}
    for f in finals:
        gid = f.get('game_id')
        if gid in source:
            raise ValueError('DUPLICATE_FINAL_GAME')
        if gid:
            source[gid] = f
    counts, new, current = Counter(), [], []
    for primary in primaries:
        gid = primary['game_id']
        try:
            kickoff = clock(primary['kickoff_utc'])
            if not clock(primary['archived_utc']) < kickoff:
                raise ValueError('INVALID_ORIGINAL_ARCHIVE')
            if at < kickoff + timedelta(hours=72):
                counts['not_mature'] += 1
                continue
            f = source.get(gid)
            if not f or f.get('game_type') != 'REG' or f.get('home_score') is None or f.get('away_score') is None:
                counts['missing_final'] += 1
                continue
            if kickoff_utc(f) != kickoff or any(f.get(k) != primary[k] for k in ('home_team', 'away_team')):
                raise ValueError('FINAL_IDENTITY_MISMATCH')
            home, away = num(f['home_score']), num(f['away_score'])
            if min(home, away) < 0 or not home.is_integer() or not away.is_integer():
                raise ValueError('INVALID_FINAL_SCORE')
            c = primary['candidate']
            if c['market'] == 'total':
                value = home+away-num(c['line'])
                if c['side'] == 'under':
                    value = -value
            else:
                value = home-away if c['side'] == 'home' else away-home
                if c['market'] == 'spread':
                    value += num(c['line'])
            outcome = 'WIN' if value > 0 else 'LOSS' if value < 0 else 'PUSH_OR_TIE'
            y = int(value > 0)
            model, reference = primary['model_nonpush_probability'], primary['same_book_proportional_no_vig_nonpush_proxy']
            decided = value != 0
            source_id = digest(encoded({k: f.get(k) for k in ('game_id', 'game_type', 'home_team', 'away_team', 'gameday', 'gametime', 'home_score', 'away_score')}))
            row = dict(spec=SPEC, game_id=gid, primary_receipt_sha256=primary['receipt_sha256'],
                       source_final_sha256=source_id, graded_utc=at.isoformat(),
                       threshold_outcome=outcome, home_score=home, away_score=away,
                       model_binary_nonpush_brier=(model-y)**2 if decided else None,
                       market_binary_nonpush_brier=(reference-y)**2 if decided and reference is not None else None,
                       verified_bookmaker_settlement=False, economic_roi=None, clv=None,
                       source='FREE_NFLVERSE_SCHEDULE_FINAL_NOT_AUTHENTICATED_BOOK_SETTLEMENT')
            row['market_minus_model_brier'] = (row['market_binary_nonpush_brier']-row['model_binary_nonpush_brier']) if row['market_binary_nonpush_brier'] is not None else None
            previous = latest.get(gid)
            if previous is None or previous['source_final_sha256'] != source_id:
                new.append(row)
                latest[gid] = row
            current.append(row)
            counts['graded'] += 1
        except (ValueError, KeyError, TypeError) as exc:
            counts['invalid_original_or_source'] += 1
            # Do not silently reuse a formerly valid grade in today's summary.
            latest.pop(gid, None)
    append(root/'grades.jsonl', new)
    paired = [r for r in current if r['market_minus_model_brier'] is not None]
    return dict(spec=SPEC, asof_utc=at.isoformat(), primary_games=len(primaries),
                currently_graded_games=counts['graded'], new_or_corrected_grade_events=len(new),
                counts=dict(counts), decided_same_contract_market_comparisons=len(paired),
                mean_market_minus_model_brier=sum(r['market_minus_model_brier'] for r in paired)/len(paired) if paired else None,
                confidence_interval=None, economic_roi=None, clv=None,
                qualified_bets=0, profitable_edge_validated=False,
                limits='One selected contract per game; selection is research. Same-book proportional no-vig proxy, not calibrated market truth. No release gates bypassed.')
