from copy import deepcopy
import pytest
from scripts.nfl_full_slate_card import build_card, contracts, query

NOW = '2026-10-14T02:00:00Z'
GAME = dict(game_id='g', home_team='DEN', away_team='SEA', kickoff_utc='2026-10-16T00:15:00Z')

def fixture():
    original = dict(game_id='g', status='UNVALIDATED_JOINT_DRIVE_PLAYER_RESEARCH',
                    asof_utc='2026-10-14T01:50:00Z', draws=1000, seed=42,
                    distribution={'HOME_MARGIN': {}, 'TOTAL': {}, 'SEA:p:receiving_yards': {}}, forecast_sha256='hash')
    scan = dict(capture_utc=original['asof_utc'], games=[original])
    replay = dict(capture_utc=scan['capture_utc'], games=[dict(game_id='g',
                  forecast=dict(event_id='g', home_team='DEN', away_team='SEA', kickoff_utc=GAME['kickoff_utc']), drives=[], roles=[])])
    entry = dict(event_id='g', collector_observed_utc='2026-10-14T01:59:00Z',
                 kickoff_utc=GAME['kickoff_utc'], source_url='https://source.example',
                 quotes=[dict(market='spread', first_side='home', second_side='away', first_line=-3.5,
                              second_line=3.5, first_american_odds=-110, second_american_odds=-110,
                              book_as_attributed_by_aggregator='Book')])
    def simulate(*args, queries, **kwargs):
        return dict(forecast_sha256='hash', distribution=original['distribution'],
                    query_thresholds=[dict(**q, over=.65, push=.1, under=.25) for q in queries])
    return scan, replay, entry, simulate


def test_full_coverage_ranking_and_no_research_promotion():
    scan, replay, entry, sim = fixture()
    missing = dict(GAME, game_id='missing')
    card = build_card(scan, replay, [GAME, missing], [entry], asof=NOW, simulator=sim)
    assert card['games_accounted_for'] == 2
    assert card['games'][1]['status'] == 'NO_USABLE_PRICES'
    best = card['games'][0]['best_candidate']
    assert best['side'] == 'home'
    assert best['raw_conditional_ev_proxy'] == pytest.approx(.65*100/110-.25)
    assert best['estimated_validated_ev'] is None
    assert card['qualified_bets'] == []
    assert best['decision'] == 'PASS'
    assert card['games'][0]['missing_game_markets'] == ['moneyline', 'total']


def test_exact_spread_orientation_and_invalid_pairs():
    assert query(dict(market='spread', side='away', line=3.5)) == ('HOME_MARGIN', 3.5, 'under')
    _, _, entry, _ = fixture()
    entry['quotes'][0]['second_line'] = 4.5
    assert list(contracts(entry)) == [{'error': 'INVALID_PAIRED_CONTRACT'}]


@pytest.mark.parametrize('observed', ['2026-10-14T00:59:00Z', '2026-10-14T02:01:00Z'])
def test_stale_or_future_quotes_never_rank(observed):
    scan, replay, entry, sim = fixture()
    entry['collector_observed_utc'] = observed
    card = build_card(scan, replay, [GAME], [entry], asof=NOW, simulator=sim)
    assert not card['ranked_candidates']


def test_replay_mismatch_blocks_game():
    scan, replay, entry, sim = fixture()
    scan['games'][0]['forecast_sha256'] = 'wrong'
    card = build_card(scan, replay, [GAME], [entry], asof=NOW, simulator=sim)
    assert card['games'][0]['status'] == 'MODEL_REPLAY_BLOCKED'
    assert not card['ranked_candidates']


def test_exact_prop_contract_supported_without_fake_prices():
    scan, replay, entry, sim = fixture()
    prop = dict(game_id='g', market='receiving_yards', team='SEA', player_id='p', side='over',
                line=89.5, american_odds=-115, book='Book', observed_utc='2026-10-14T01:59:00Z',
                source_url='https://source.example')
    card = build_card(scan, replay, [GAME], [entry], asof=NOW, prop_contracts=[prop], simulator=sim)
    assert len(card['ranked_candidates']) == 3
    assert card['games'][0]['prop_price_coverage'] == 'PARTIAL'
    assert 'PLAYER_PARTICIPATION_UNVERIFIED' in next(c for c in card['ranked_candidates'] if c['market']=='receiving_yards')['blockers']


def test_percentage_reference_is_compared_without_ev_or_betting_rank():
    scan, replay, entry, sim = fixture()
    prop = dict(game_id='g', market='receiving_yards', team='SEA', player_id='p', side='over',
                line=89.5, american_odds=None, quote_format='DISPLAYED_PERCENT_REFERENCE',
                displayed_probability=.55, opposite_displayed_probability=.52,
                book='Kalshi Sports', observed_utc='2026-10-14T01:59:00Z', source_url='https://source.example')
    result = build_card(scan, replay, [GAME], [entry], asof=NOW, prop_contracts=[prop], simulator=sim)
    assert len(result['ranked_candidates']) == 2
    assert len(result['displayed_market_probability_comparisons']) == 1
    reference = result['displayed_market_probability_comparisons'][0]
    assert reference['raw_conditional_win_probability'] == .65
    assert reference['raw_probability_minus_displayed_reference'] == pytest.approx(.1)
    assert reference['raw_conditional_ev_proxy'] is None
    assert reference['american_odds'] is None
    assert result['games'][0]['prop_price_coverage'] == 'DISPLAYED_REFERENCE_ONLY'
    assert result['qualified_bet_count'] == 0

def test_missing_model_and_duplicate_game_are_explicit():
    scan, replay, entry, sim = fixture()
    scan['games'] = []
    card = build_card(scan, replay, [GAME], [entry], asof=NOW, simulator=sim)
    assert card['games'][0]['status'] == 'MODEL_OR_SOURCE_MISSING'
    with pytest.raises(ValueError, match='duplicate'):
        build_card(scan, replay, [GAME, GAME], [entry], asof=NOW, simulator=sim)



def test_qb_passing_price_is_not_ranked_when_starter_unverified():
    """Historical QB pass shares do not constitute pregame starter confirmation."""
    scan, replay, entry, sim = fixture()
    scan["games"][0]["distribution"]["SEA:p:passing_yards"] = {}
    quoted = dict(
        game_id="g", market="passing_yards", team="SEA", player_id="p",
        side="over", line=160.5, american_odds=-110, book="PublicBook",
        observed_utc="2026-10-14T01:59:00Z", source_url="https://source.example",
    )
    card = build_card(
        scan, replay, [GAME], [entry], asof=NOW,
        prop_contracts=[quoted], simulator=sim,
    )
    assert len(card["ranked_candidates"]) == 2
    assert all(x["market"] != "passing_yards" for x in card["ranked_candidates"])
    game = card["games"][0]
    assert game["qb_passing_contracts_excluded_unverified_starter"] == 1
    assert "QB_STARTER_UNVERIFIED_PASSING_CONTRACT_EXCLUDED" in game["errors"]
    assert game["decision"] == "PASS"
    assert not card["qualified_bets"]
