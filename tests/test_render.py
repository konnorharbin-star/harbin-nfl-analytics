from __future__ import annotations

import polars as pl

from nfl.render import build_weekly_board, write_weekly_publication


def _current() -> pl.DataFrame:
    base = {
        "season": 2026,
        "week": 4,
        "game_id": "g1",
        "date": "2026-10-04",
        "kickoff": "2026-10-04T17:00:00+00:00",
        "away_team": "NYJ",
        "home_team": "PIT",
        "model_margin_home": 4.0,
        "model_total": 44.0,
        "calibrated_home_probability": 0.64,
        "portfolio_candidate_units": 0.2,
        "portfolio_stake_units": 0.0,
        "portfolio_action": "PAPER",
        "execution_ready": True,
    }
    return pl.DataFrame(
        [
            {
                **base,
                "quant_market": "moneyline",
                "quant_side": "home",
                "quant_price": None,
                "quant_odds": -150,
                "quant_signal": "PASS",
                "research_signal": "STRONG",
            },
            {
                **base,
                "quant_market": "spread",
                "quant_side": "away",
                "quant_price": 3.5,
                "quant_odds": -110,
                "quant_signal": "PASS",
                "research_signal": "BET",
            },
            {
                **base,
                "quant_market": "total",
                "quant_side": "over",
                "quant_price": 42.5,
                "quant_odds": -110,
                "quant_signal": "PASS",
                "research_signal": "LEAN",
            },
        ]
    )


def test_weekly_board_preserves_score_probability_and_markets() -> None:
    board = build_weekly_board(_current())
    assert board.height == 1
    row = board.to_dicts()[0]
    assert row["away_score"] == 20
    assert row["home_score"] == 24
    assert row["winner"] == "PIT"
    assert row["win_pct"] == 64
    assert row["proj_total"] == 44.0
    assert row["moneyline"]["quant_signal"] == "PASS"
    assert row["moneyline"]["research_signal"] == "STRONG"
    assert row["spread"]["research_signal"] == "BET"
    assert row["total"]["research_signal"] == "LEAN"


def test_weekly_board_breaks_display_tie_toward_projected_winner() -> None:
    current = _current().with_columns(
        pl.lit(0.2).alias("model_margin_home"),
        pl.lit(51.3).alias("model_total"),
        pl.lit(0.51).alias("calibrated_home_probability"),
    )
    row = build_weekly_board(current).to_dicts()[0]
    assert row["winner"] == "PIT"
    assert row["home_score"] == row["away_score"] + 1


def test_weekly_publication_hides_unallocated_or_closed_signals(tmp_path) -> None:
    current = _current().with_columns(
        pl.lit(0.0).alias("portfolio_candidate_units"),
        pl.lit("PASS").alias("portfolio_action"),
        pl.lit(False).alias("execution_ready"),
    )
    write_weekly_publication(
        current,
        week=4,
        updated_at="2026-10-02T16:41:28+00:00",
        release_state="PAPER",
        output_dir=tmp_path,
    )
    document = (tmp_path / "nfl_week_4.html").read_text()
    assert '<span class="badge strong">STRONG</span>' not in document
    assert '<span class="badge bet">BET</span>' not in document
    assert '<span class="badge lean">LEAN</span>' not in document


def test_weekly_publication_matches_cfb_picks_layout(tmp_path) -> None:
    result = write_weekly_publication(
        _current(),
        week=4,
        updated_at="2026-10-02T16:41:28+00:00",
        release_state="PAPER",
        output_dir=tmp_path,
    )

    document = (tmp_path / "nfl_week_4.html").read_text()
    assert "NFL MODEL · WEEK 4 PICKS" in document
    assert "Projected scores &amp; best bets · Updated Oct 2, 2026 · 11:41 AM CT" in document
    assert "MATCHUP (WINNER BOLD)" in document
    assert "<th>CTX</th>" not in document
    assert "PIT -150" in document
    assert "NYJ +3.5" in document
    assert "O 42.5" in document
    assert "proj 44" in document
    assert '<span class="badge strong">STRONG</span>' in document
    assert '<span class="badge bet">BET</span>' in document
    assert '<span class="badge lean">LEAN</span>' in document
    assert "STRONG/PAPER" not in document
    assert "PAPER evidence mode" in document
    assert "not production staking" in document
    assert result["presentation"] == "cfb_style_weekly_picks_v2"
    assert (tmp_path / "nfl_week_4_page1.png").exists()
    assert (tmp_path / "latest.png").exists()
    assert (tmp_path / "latest.png").read_bytes() == (
        tmp_path / "nfl_week_4_page1.png"
    ).read_bytes()
