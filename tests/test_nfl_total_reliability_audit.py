from scripts.nfl_total_reliability_audit import audit
import pytest


def test_status_remains_research():
    row = {'market_type': 'total', 'season': '2025', 'game_id': 'g1', 'side': 'over', 'model_probability': '0.57', 'expected_value_per_unit': '0.08', 'net_units': '-1', 'result': 'loss', 'price_stage': 'espn_archived_open', 'entry_timestamp_verified': 'false'}
    output = audit([row])
    assert output['wager_authorized'] is False
    assert output['season_results']['2025']['older_model_over_probability_55_60pct']['games'] == 1


def test_duplicate_rejected():
    row = {'market_type': 'total', 'season': '2025', 'game_id': 'g1', 'side': 'over', 'model_probability': '0.57', 'expected_value_per_unit': '0.08', 'net_units': '-1', 'result': 'loss', 'price_stage': 'espn_archived_open', 'entry_timestamp_verified': 'false'}
    with pytest.raises(ValueError):
        audit([row, row])
