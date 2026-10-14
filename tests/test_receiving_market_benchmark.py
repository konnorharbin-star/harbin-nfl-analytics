import io
import json
import urllib.error

import polars as pl
import pytest

from nfl.receiving_market_benchmark import decimal_odds, import_histories, score_quotes
from scripts.nfl_receiving_market_benchmark import fetch_history, select_players


def history(**updates):
    quote = {
        "game_id": "2025_05_A_B",
        "line": 49.5,
        "actual": 70,
        "over_odds": -110,
        "under_odds": -110,
    }
    quote.update(updates)
    return {
        "data": {
            "player_id": "00-001",
            "market": "player_reception_yds",
            "bookmaker": "draftkings",
            "games": [quote],
            "truncated": False,
        }
    }


def test_same_book_actual_line_import_and_no_timestamp_invention():
    quotes, audit = import_histories(history())
    assert len(quotes) == 1
    assert quotes[0]["line"] == 49.5
    assert quotes[0]["market_over_proxy"] == 0.5
    assert quotes[0]["quote_origin_utc"] is None
    assert not audit.get("invalid_or_duplicate_contract", 0)


@pytest.mark.parametrize(
    "updates",
    [
        {"over_odds": None},
        {"over_odds": 0},
        {"over_odds": True},
        {"line": 50},
        {"actual": None},
        {"actual": -1},
        {"line": float("nan")},
    ],
)
def test_missing_or_unsupported_input_cannot_make_a_price(updates):
    quotes, audit = import_histories(history(**updates))
    assert quotes == []
    assert audit["invalid_or_duplicate_contract"] == 1


def test_consensus_and_duplicate_quotes_are_not_single_book_contracts():
    data = history()
    data["data"]["bookmaker"] = None
    assert import_histories(data)[0] == []
    assert import_histories({"responses": [history(), history()]})[0] == []


def test_outcome_conflict_and_missing_forecast_remain_unknown():
    quotes, _ = import_histories(history())
    assert score_quotes({}, quotes)["matched_contracts"] == 0
    row = {"actual": 69}
    result = score_quotes({("2025_05_A_B", "00-001"): row}, quotes)
    assert result["audit"]["provider_nflverse_outcome_conflict"] == 1
    assert result["actual_roi"] is None
    assert result["validated_edges"] == 0


def test_selection_does_not_depend_on_outcome_and_cannot_promote():
    def run(actual):
        quotes, _ = import_histories(history(actual=actual))
        row = {
            "actual": actual,
            "challenger_mean": 50,
            "usage_mean": 80,
            "base_errors": [-1, 0, 1] * 60,
            "usage_errors": [-1, 0, 1] * 60,
        }
        return score_quotes({("2025_05_A_B", "00-001"): row}, quotes)

    first, second = run(70), run(10)
    a, b = first["research_selections"][0], second["research_selections"][0]
    assert (a["player_id"], a["side"], a["raw_ev_proxy"]) == (
        b["player_id"],
        b["side"],
        b["raw_ev_proxy"],
    )
    assert a["nominal_return_proxy"] != b["nominal_return_proxy"]
    assert first["promotion_earned"] is False
    assert first["game_cluster_brier_difference_interval_98_75"] is None
    assert first["verified_executable_decisions"] == 0


def test_player_selection_uses_2024_targets_only():
    rows = [
        {
            "player_id": "00-a",
            "season": 2024,
            "season_type": "REG",
            "position": "WR",
            "targets": 50,
        },
        {
            "player_id": "00-b",
            "season": 2024,
            "season_type": "REG",
            "position": "TE",
            "targets": 40,
        },
        {
            "player_id": "00-b",
            "season": 2025,
            "season_type": "REG",
            "position": "TE",
            "targets": 500,
        },
    ]
    assert select_players(pl.DataFrame(rows), 1) == ["00-a"]
    with pytest.raises(ValueError):
        select_players(pl.DataFrame(rows), 21)


def test_free_fetch_has_no_retry_and_checkpoints_completed_responses():
    calls, checkpoints = [], []

    def opener(request, timeout):
        calls.append(request)
        if len(calls) == 2:
            raise urllib.error.HTTPError(request.full_url, 402, "no credits", {}, None)
        return io.BytesIO(json.dumps(history()).encode())

    with pytest.raises(RuntimeError, match="HTTP_402"):
        fetch_history(["00-a", "00-b"], "test-secret", opener=opener, checkpoint=checkpoints.append)
    assert len(calls) == 2
    assert len(checkpoints[0]["responses"]) == 1
    assert checkpoints[0]["complete"] is False
    assert "test-secret" not in json.dumps(checkpoints)


def test_missing_key_and_invalid_odds_fail_before_any_network():
    with pytest.raises(ValueError, match="KEY_REQUIRED"):
        fetch_history(["00-a"], None, opener=lambda *args, **kwargs: pytest.fail("network called"))
    assert decimal_odds(+150) == 2.5
    assert decimal_odds(-200) == 1.5
