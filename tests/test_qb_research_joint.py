from test_coupled_drive_player_shadow import run


def test_qb_research_conserves_passing_volume():
    x = run()
    d = x['distribution']
    for team in ('LA', 'BUF'):
        for player_stat, team_stat in (('passing_attempts', 'pass_attempts'), ('passing_completions', 'completions'), ('passing_yards', 'passing_yards')):
            assert abs(d[f'{team}:{team}_qb:{player_stat}']['mean'] + d[f'{team}:OTHER:{player_stat}']['mean'] - d[f'{team}:{team_stat}']['mean']) < 1e-8
    assert x['qb_passing_player_ids_not_confirmed_starters'] is True
