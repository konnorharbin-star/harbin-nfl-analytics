import pytest

from nfl.receiving_market_benchmark import import_histories, score_quotes


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
