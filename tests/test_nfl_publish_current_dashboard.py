"""Checks for the research-only Pages publisher and its chronology gates."""
from datetime import UTC, datetime
import json

import pytest

from scripts.nfl_publish_current_dashboard import dashboard, publish, validate

NOW = datetime(2026, 10, 19, 12, tzinfo=UTC)


def sample():
    return {
        "spec": "NFL_INDEPENDENT_FORECAST_PUBLICATION_V1",
        "season": 2026, "week": 6, "games_requested": 1, "games_forecast": 1,
        "generated_utc": "2026-10-19T11:55:00+00:00",
        "model_capture_utc": "2026-10-19T11:50:00+00:00",
        "sportsbook_prices_used": False,
        "automatic_betting_enabled": False, "qualified_bets": 0,
        "accuracy_vs_market": "UNPROVEN",
        "predictive_family": "NFL_PLAYER_WORKLOAD_SHRINKAGE_V2",
        "games": [
            {
                "game_id": "2026_06_WAS_SF", "home_team": "SF",
                "away_team": "WAS", "kickoff_utc": "2026-10-20T00:15:00Z",
                "predicted_score": {"home": 28.5, "away": 22.0},
                "fair_home_spread": -6.5, "predicted_total": 50.5,
                "status": "INDEPENDENT_CONDITIONAL_FORECAST",
            }
        ],
    }


def seed_assets(tmp_path, bundle=None):
    s = tmp_path / "outputs"
    s.mkdir()
    (s / "independent_forecasts.json").write_text(json.dumps(bundle or sample()))
    for ext in (".html", ".png"):
        (s / ("independent_forecasts" + ext)).write_bytes(b"TEST FILE")
    (s / "independent_game_predictions.csv").write_text("game_id\nsample\n")
    (s / "independent_player_predictions.csv").write_text("player_id\nsample\n")
    return s


def test_current_research_pages_preserve_legacy_and_all_numeric_sources(tmp_path):
    source = seed_assets(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "index.html").write_text("<title>NFL Model Week 5</title>Archive content")
    result = publish(source, docs, at=NOW)
    page = (docs / "index.html").read_text()
    assert result["upcoming_games"] == 1
    assert result["qualified_bets"] == 0 and result["market_superiority"] == "UNPROVEN"
    assert "Week 6" in page and "WAS @ SF" in page and "-6.5" in page
    assert "50.5" in page and "28.5" in page and "22.0" in page
    assert "NO APPROVED BETS" in page and "lineups" in page.lower()
    assert "independent_latest.png" in page and "independent_latest.html" in page
    assert (docs / "legacy_market_week5.html").read_text().endswith("Archive content")
    assert (docs / "independent_latest.json").read_bytes() == (source / "independent_forecasts.json").read_bytes()
    assert (docs / "independent_latest.png").read_bytes() == (source / "independent_forecasts.png").read_bytes()


@pytest.mark.parametrize("change", [
    lambda b: b.update(sportsbook_prices_used=True),
    lambda b: b.update(automatic_betting_enabled=True),
    lambda b: b.update(qualified_bets=1),
    lambda b: b.update(accuracy_vs_market="VALIDATED"),
    lambda b: b.update(spec="OTHER"),
    lambda b: b.update(generated_utc="2026-10-17T11:00:00Z"),
    lambda b: b.update(model_capture_utc="2026-10-20T11:50:00Z"),
    lambda b: b["games"][0].update(kickoff_utc="2026-10-19T11:00:00Z"),
    lambda b: b["games"][0].update(predicted_total=42.0),
    lambda b: b["games"][0].update(fair_home_spread=+6.5),
])
def test_misleading_or_stale_publication_fails_closed(change):
    b = sample()
    change(b)
    with pytest.raises(ValueError):
        validate(b, at=NOW)


def test_previously_published_newer_bundle_cannot_be_replaced():
    b = sample()
    old = {"generated_utc": "2026-10-19T11:56:00Z"}
    with pytest.raises(ValueError, match="WOULD_REPLACE_NEWER"):
        validate(b, at=NOW, previous=old)


def test_started_games_are_archived_and_not_live(tmp_path):
    b = sample()
    old = dict(b["games"][0])
    old["game_id"], old["kickoff_utc"] = "OLD", "2026-10-18T17:00:00Z"
    b["games"].insert(0, old)
    b["games_requested"] = 2
    page = dashboard(b, at=NOW)
    assert "Started – frozen forecast" in page
    assert "Upcoming matchups at publication: 1" in page


def test_missing_source_assets_cannot_erase_current_pages(tmp_path):
    source = seed_assets(tmp_path)
    (source / "independent_forecasts.png").unlink()
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "index.html").write_text("old board")
    with pytest.raises(ValueError, match="MISSING_BUNDLE_ASSET"):
        publish(source, docs, at=NOW)
    assert (docs / "index.html").read_text() == "old board"


def test_no_completed_or_blocked_only_dashboard():
    b = sample()
    b["games"][0]["predicted_score"] = None
    b["games"][0]["status"] = "PUBLICATION_INTEGRITY_BLOCKED"
    with pytest.raises(ValueError, match="NO_UPCOMING_GAMES"):
        validate(b, at=NOW)


def test_untrusted_team_text_is_escaped():
    b = sample()
    b["games"][0]["away_team"] = "<script>alert(1)</script>"
    output = dashboard(b, at=NOW)
    assert "<script>" not in output
    assert "&lt;script&gt;" in output
