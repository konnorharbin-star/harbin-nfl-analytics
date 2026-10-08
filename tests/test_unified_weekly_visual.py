"""Every exported NFL picks page shares the canonical screenshot format."""
from __future__ import annotations

from pathlib import Path

import polars as pl
from PIL import Image

from nfl.render import write_weekly_publication


def test_screenshot_style_consistent_across_two_png_pages_and_html(tmp_path: Path):
    rows = []
    for i in range(15):
        rows.append({
            "season": 2026, "week": 5, "game_id": f"week5_{i}",
            "away_team": "BUF", "home_team": "KC",
            "model_margin_home": 3.0, "model_total": 45.0,
            "calibrated_home_probability": 0.60,
            "quant_market": "spread", "quant_side": "home",
            "quant_price": -2.5, "quant_odds": -110,
            "quant_signal": "PASS", "research_signal": "BET",
            "execution_ready": False,
        })
    result = write_weekly_publication(
        pl.DataFrame(rows), week=5,
        updated_at="2026-10-08T20:44:00+00:00",
        release_state="RESEARCH", output_dir=tmp_path,
    )
    assert result["board_games"] == 15
    assert result["rows_per_page"] == 14
    assert len(result["png_pages"]) == 2
    assert len(result["cache_safe_png_pages"]) == 2
    assert result["presentation"] == "unified_nfl_weekly_picks_screenshot_v1"
    for path in result["png_pages"] + result["cache_safe_png_pages"]:
        with Image.open(path) as image:
            assert image.size == (1320, 690)
    html = (tmp_path / "nfl_week_5.html").read_text()
    for heading in ("MATCHUP", "PROJ", "MONEYLINE", "SPREAD", "TOTAL"):
        assert heading in html.upper()
    assert "WATCH" in html
    assert Path(result["latest_png"]).read_bytes() == Path(
        result["png_pages"][0]
    ).read_bytes()
