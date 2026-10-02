from __future__ import annotations

import polars as pl

from run_espn_verified_market_backtest import schedule_seasons_for_evidence

from nfl.espn_historical import (
    build_espn_archive_bets,
    fetch_espn_archive_quotes,
    parse_espn_archive_item,
)


def _price(value: str) -> dict[str, object]:
    return {
        "american": value,
        "alternateDisplayValue": value,
    }


def _provider_item(
    *,
    provider_id: str = "58",
    provider_name: str = "ESPN BET",
) -> dict[str, object]:
    return {
        "provider": {
            "id": provider_id,
            "name": provider_name,
        },
        "homeTeamOdds": {
            "open": {
                "moneyLine": _price("-280"),
                "pointSpread": _price("-4.5"),
                "spread": _price("-110"),
            },
            "close": {
                "moneyLine": _price("-400"),
                "pointSpread": _price("-7.5"),
                "spread": _price("-110"),
            },
        },
        "awayTeamOdds": {
            "open": {
                "moneyLine": _price("+230"),
                "pointSpread": _price("+4.5"),
                "spread": _price("-110"),
            },
            "close": {
                "moneyLine": _price("+300"),
                "pointSpread": _price("+7.5"),
                "spread": _price("-110"),
            },
        },
        "open": {
            "total": _price("47.5"),
            "over": _price("-110"),
            "under": _price("-110"),
        },
        "close": {
            "total": _price("45.5"),
            "over": _price("-110"),
            "under": _price("-110"),
        },
    }


def _target_projection() -> dict[str, object]:
    return {
        "season": 2025,
        "week": 5,
        "game_id": "2025_05_DAL_PHI",
        "home_team": "PHI",
        "away_team": "DAL",
        "projected_home_margin": 6.0,
        "projected_total": 46.0,
        "actual_home_margin": 7.0,
        "actual_total": 48.0,
    }


def _projections() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(64):
        projected_margin = float((index % 9) - 4)
        projected_total = 42.0 + float(index % 7)
        rows.append(
            {
                "season": 2024,
                "week": 1 + (index % 16),
                "game_id": f"2024_{index:03d}",
                "home_team": "AAA",
                "away_team": "BBB",
                "projected_home_margin": projected_margin,
                "projected_total": projected_total,
                "actual_home_margin": projected_margin + float((index % 5) - 2),
                "actual_total": projected_total + float((index % 7) - 3),
            }
        )
    rows.append(_target_projection())
    return pl.DataFrame(rows)


def test_parse_espn_archive_item_builds_complete_open_and_close_pairs() -> None:
    item = _provider_item()

    opened = parse_espn_archive_item(
        item,
        game_id="2025_05_DAL_PHI",
        event_id="401772510",
        stage="open",
    )
    closed = parse_espn_archive_item(
        item,
        game_id="2025_05_DAL_PHI",
        event_id="401772510",
        stage="close",
    )

    assert len(opened) == 6
    assert len(closed) == 6
    assert {row["market_type"] for row in opened} == {
        "moneyline",
        "spread",
        "total",
    }

    home_spread = next(
        row
        for row in opened
        if row["market_type"] == "spread" and row["side"] == "home"
    )
    away_ml = next(
        row
        for row in opened
        if row["market_type"] == "moneyline" and row["side"] == "away"
    )
    over = next(
        row
        for row in opened
        if row["market_type"] == "total" and row["side"] == "over"
    )
    assert home_spread["line"] == -4.5
    assert home_spread["american_odds"] == -110
    assert away_ml["american_odds"] == 230
    assert over["line"] == 47.5
    assert over["american_odds"] == -110


def test_live_odds_provider_alias_uses_same_canonical_book() -> None:
    rows = parse_espn_archive_item(
        _provider_item(
            provider_id="59",
            provider_name="ESPN Bet - Live Odds",
        ),
        game_id="2025_05_DAL_PHI",
        event_id="401772510",
        stage="close",
    )

    assert len(rows) == 6
    assert {row["book"] for row in rows} == {"ESPN BET"}
    assert {row["canonical_book"] for row in rows} == {"espnbet"}


def test_fetch_archive_quotes_prefers_non_live_same_book_provider() -> None:
    projection = pl.DataFrame([_target_projection()])

    class FixtureClient:
        max_workers = 1

        def scoreboard(self, *, season: int, week: int) -> dict[str, object]:
            assert season == 2025
            assert week == 5
            return {
                "events": [
                    {
                        "id": "401772510",
                        "competitions": [
                            {
                                "competitors": [
                                    {
                                        "homeAway": "home",
                                        "team": {"abbreviation": "PHI"},
                                    },
                                    {
                                        "homeAway": "away",
                                        "team": {"abbreviation": "DAL"},
                                    },
                                ]
                            }
                        ],
                    }
                ]
            }

        def core_odds(self, event_id: str) -> list[dict[str, object]]:
            assert event_id == "401772510"
            return [
                _provider_item(
                    provider_id="59",
                    provider_name="ESPN Bet - Live Odds",
                ),
                _provider_item(),
            ]

    opened, closed, coverage = fetch_espn_archive_quotes(
        projection,
        client=FixtureClient(),
    )

    assert opened.height == 6
    assert closed.height == 6
    assert set(opened.get_column("book").to_list()) == {"ESPN BET"}
    assert coverage.requested_games == 1
    assert coverage.matched_events == 1
    assert coverage.games_with_open == 1
    assert coverage.games_with_close == 1
    assert coverage.entry_pairs == 3
    assert coverage.closing_pairs == 3


def test_espn_archive_bets_are_verified_without_fabricated_timestamps() -> None:
    item = _provider_item()
    entries = pl.DataFrame(
        parse_espn_archive_item(
            item,
            game_id="2025_05_DAL_PHI",
            event_id="401772510",
            stage="open",
        )
    )
    closes = pl.DataFrame(
        parse_espn_archive_item(
            item,
            game_id="2025_05_DAL_PHI",
            event_id="401772510",
            stage="close",
        )
    )

    bets = build_espn_archive_bets(_projections(), entries, closes)

    assert bets.height == 3
    assert set(bets.get_column("market_type").to_list()) == {
        "moneyline",
        "spread",
        "total",
    }
    assert bets.get_column("entry_price_verified").all()
    assert bets.get_column("entry_quote_verified").all()
    assert not bets.get_column("entry_timestamp_verified").any()
    assert set(bets.get_column("entry_price_stage").to_list()) == {
        "espn_archived_open"
    }
    assert bets.get_column("closing_quote_verified").all()
    assert bets.get_column("clv_proxy").null_count() == 0
    assert bets.get_column("captured_at").null_count() == 3
    assert bets.get_column("decision_time").null_count() == 3


def test_espn_evidence_schedule_window_includes_2022_for_2024_start() -> None:
    assert schedule_seasons_for_evidence(2024, 2025) == [
        2022,
        2023,
        2024,
        2025,
    ]
