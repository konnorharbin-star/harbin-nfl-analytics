import polars as pl

from nfl.actionable_board import build_actionable_board


def _row(**overrides):
    row = {
        "season": 2026, "week": 6, "game_id": "g1",
        "away_team": "BUF", "home_team": "KC",
        "model_margin_home": 4.0, "model_total": 48.0,
        "quant_market": "spread", "quant_side": "home",
        "quant_odds": -110, "quant_quote_at": "2026-10-08T18:00:00Z",
        "execution_ready": True, "context_veto": False,
        "probability_reliability_veto": False, "portfolio_stake_units": 0.5,
    }
    row.update(overrides)
    return row


def test_readable_model_score_and_executable_action():
    row = build_actionable_board(pl.DataFrame([_row()])).to_dicts()[0]
    assert row["predicted_home_score"] == 26.0
    assert row["predicted_away_score"] == 22.0
    assert row["projected_winner"] == "KC"
    assert row["betting_action"] == "BET"


def test_no_quote_or_veto_never_promotes_bet():
    no_quote = _row(quant_quote_at=None)
    no_quote["game_id"] = "g2"
    veto = _row(context_veto=True)
    veto["game_id"] = "g3"
    zero = _row(portfolio_stake_units=0)
    zero["game_id"] = "g4"
    board = build_actionable_board(pl.DataFrame([no_quote, veto, zero]))
    assert set(board["betting_action"].to_list()) == {"PASS"}


def test_without_market_keeps_game_predictions():
    row = build_actionable_board(pl.DataFrame([
        {k: v for k, v in _row().items() if not k.startswith("quant_")}
    ])).to_dicts()[0]
    assert row["betting_action"] == "NO_VERIFIED_MARKET"
    assert row["predicted_home_score"] == 26.0
