from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from scripts.price_comparison_scan import book_key, decimal, normalize, scan

NOW = datetime(2026, 10, 10, 19, tzinfo=UTC)


def quotes():
    game = {
        "game_id": "g",
        "home_team": "Home",
        "away_team": "Away",
        "date": (NOW + timedelta(hours=2)).isoformat(),
    }
    result = []
    for book in ["DraftKings", "FanDuel", "BetMGM", "Other"]:
        result.extend(
            normalize(
                game,
                {
                    "provider": book,
                    "source": "test",
                    "home_ml": 120 if book == "DraftKings" else -110,
                    "away_ml": -140 if book == "DraftKings" else -110,
                },
                NOW.isoformat(),
            )
        )
    return result


def test_excludes_best_book_from_reference_and_never_authorizes_bet():
    report = scan(quotes(), NOW)
    candidate = report["candidates"][0]
    assert candidate["reference_probability"] == 0.5
    assert candidate["reference_books"] == 3
    assert candidate["american_odds"] == 120
    assert candidate["reference_price_gap_pp"] == pytest.approx(100 * (0.5 - 1 / 2.2))
    assert not report["betting_authorized"]
    assert report["qualified_bets"] == candidate["stake_units"] == 0


def test_duplicate_book_alias_does_not_inflate_reference_count():
    rows = quotes()[:3]
    duplicate = deepcopy(rows[0])
    duplicate["book_label"] = "ActionNetwork book 15"
    duplicate["book_key"] = book_key(duplicate["book_label"])
    assert scan(rows + [duplicate], NOW)["candidates"] == []
    assert book_key("Draft Kings") == book_key("DraftKings")


def test_different_handicaps_cannot_supply_references():
    rows = quotes()
    for i, r in enumerate(rows):
        r.update(market="spread", line=-3 if i == 0 else -3.5)
    assert not scan(rows, NOW)["candidates"]


@pytest.mark.parametrize("time", [NOW - timedelta(minutes=16), NOW + timedelta(seconds=1)])
def test_stale_and_future_observations_rejected(time):
    rows = quotes()
    for r in rows:
        r["observed_at"] = time.isoformat()
    assert scan(rows, NOW)["counts"]["stale_future_or_started"] == 4


def test_large_capture_skew_cannot_create_price_candidate():
    rows = quotes()
    rows[0]["observed_at"] = (NOW - timedelta(minutes=3)).isoformat()
    assert not scan(rows, NOW)["candidates"]


def test_started_games_and_naive_timestamps_rejected():
    rows = quotes()
    for r in rows:
        r["kickoff"] = NOW.isoformat()
    assert not scan(rows, NOW)["candidates"]
    rows[0]["observed_at"] = "2026-10-10T18:59:00"
    assert scan(rows, NOW)["counts"]["missing_aware_timestamps"] == 1


def test_no_default_odds_and_invalid_prices_rejected():
    assert decimal(0) is None
    assert decimal(99) is None
    assert decimal(float("nan")) is None
    game = {
        "game_id": "g",
        "home_team": "h",
        "away_team": "a",
        "date": (NOW + timedelta(hours=1)).isoformat(),
    }
    assert (
        normalize(
            game,
            {"provider": "Book", "home_spread": -3, "home_spread_price": -110},
            NOW.isoformat(),
        )
        == []
    )


def test_away_spread_is_opposite_of_home_reference_line():
    rows = quotes()
    for i, r in enumerate(rows):
        r.update(market="spread", line=-3, odds=[-140, 120] if i == 0 else [-110, -110])
    assert scan(rows, NOW)["candidates"][0]["line"] == 3


def test_stress_haircut_is_at_least_two_percentage_points():
    candidate = scan(quotes(), NOW)["candidates"][0]
    assert candidate["reference_price_gap_pp"] - candidate["sensitivity_gap_pp"] == pytest.approx(2)


def test_conflicting_game_identity_and_anonymous_book_fail_closed():
    rows = quotes()
    rows[0]["home_team"] = "Different team"
    assert not scan(rows, NOW)["candidates"]
    assert book_key(None) == book_key("") == ""


def test_collector_timestamp_does_not_certify_book_origin():
    row = quotes()[0]
    assert row["observed_at"] == NOW.isoformat()
    assert not row["source_quote_time_verified"]
    assert not row["executable_price_verified"]


@pytest.mark.parametrize("market", ["spread", "total"])
def test_both_side_lines_required_for_non_moneyline_pairs(market):
    game = {
        "game_id": "g",
        "home_team": "h",
        "away_team": "a",
        "date": (NOW + timedelta(hours=1)).isoformat(),
    }
    q = (
        {
            "provider": "Book",
            "home_spread": -3,
            "home_spread_price": -110,
            "away_spread_price": -110,
        }
        if market == "spread"
        else {"provider": "Book", "market_total": 40, "over_price": -110, "under_price": -110}
    )
    assert normalize(game, q, NOW.isoformat()) == []

    q.update(away_spread=3) if market == "spread" else q.update(over_total=40, under_total=40)
    assert len(normalize(game, q, NOW.isoformat())) == 1
    q.update(away_spread=3.5) if market == "spread" else q.update(under_total=41)
    assert normalize(game, q, NOW.isoformat()) == []


@pytest.mark.skipif(not Path("harbin").is_dir(), reason="CFB collector runtime")
def test_cfb_bulk_refresh_imports_runtime_and_never_calls_paid_client(monkeypatch):
    from harbin.market_intel import MarketIntelligence

    from scripts.price_comparison_scan import refresh

    monkeypatch.setenv("THE_ODDS_API_KEY", "unused-test-value")

    def forbidden(self):
        raise AssertionError("Paid client called")

    def public(self, games):
        self.action_games = [
            {
                "home_names": {"home"},
                "away_names": {"away"},
                "start_time": (NOW + timedelta(hours=2)).isoformat(),
                "quotes": [
                    {
                        "provider": "Book",
                        "source": "action_network",
                        "home_ml": -110,
                        "away_ml": -110,
                    }
                ],
            }
        ]

    monkeypatch.setattr(MarketIntelligence, "_load_odds_api", forbidden)
    monkeypatch.setattr(MarketIntelligence, "_load_action_network", public)
    game = {
        "game_id": "g",
        "home_team": "Home",
        "away_team": "Away",
        "date": (datetime.now(UTC) + timedelta(hours=2)).isoformat(),
        "season": 2026,
        "week": 6,
    }

    # Use the same actual future kickoff in the mocked bulk payload.
    def current_public(self, games):
        public(self, games)
        self.action_games[0]["start_time"] = game["date"]

    monkeypatch.setattr(MarketIntelligence, "_load_action_network", current_public)
    rows, errors = refresh("cfb", [game])
    assert len(rows) == 1 and errors == []


def test_custom_collector_report_does_not_overwrite_another_writer(monkeypatch, tmp_path):
    import sys

    from scripts import price_comparison_scan as cli

    monkeypatch.chdir(tmp_path)
    Path("docs").mkdir()
    default = Path("docs/price_comparison_scan.json")
    default.write_text('{"other_collector":"preserve"}')
    monkeypatch.setattr(cli, "load", lambda sport: ([], []))
    monkeypatch.setattr(
        sys, "argv", ["scan", "--sport", "nfl", "--out", "docs/price_comparison_scan_line.json"]
    )
    cli.main()
    assert default.read_text() == '{"other_collector":"preserve"}'
    assert Path("docs/price_comparison_scan_line.json").exists()
    assert len(list(Path("history/price_scan_v1/captures").glob("*.json"))) == 1
