from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime

import polars as pl

from nfl.backtest_runtime import build_backtest_runtime_report, normalize_clv_proxy
from nfl.espn_market import ESPNTwoWayMarket
from nfl.market_intel import build_market_intelligence
from nfl.policy import DEFAULT_POLICY
from nfl.portfolio import apply_portfolio_controls
from nfl.pro_market import CurrentOddsAPIClient, collect_current_markets
from nfl.render import write_weekly_publication


def _historical_games(count: int = 80) -> pl.DataFrame:
    rows = []
    for index in range(count):
        margin = float((index % 9) - 4)
        total = 44.0 + float(index % 7)
        rows.append(
            {
                "projected_home_margin": margin,
                "projected_total": total,
                "actual_home_margin": margin + float((index % 5) - 2) * 2.0,
                "actual_total": total + float((index % 6) - 3) * 2.0,
            }
        )
    return pl.DataFrame(rows)


def _market(
    *,
    book: str,
    provider: str,
    home_odds: int,
    away_odds: int,
) -> ESPNTwoWayMarket:
    return ESPNTwoWayMarket(
        game_id="g1",
        market_type="moneyline",
        provider=provider,
        book=book,
        source_event_id=f"{provider}-1",
        captured_at=datetime(2026, 10, 1, 15, tzinfo=UTC),
        first_side="home",
        first_line=None,
        first_american_odds=home_odds,
        second_side="away",
        second_line=None,
        second_american_odds=away_odds,
    )


def test_market_intelligence_line_shops_and_deduplicates_book_identity() -> None:
    projection = pl.DataFrame(
        [
            {
                "season": 2026,
                "week": 4,
                "game_id": "g1",
                "gameday": "2026-10-04",
                "away_team": "NYJ",
                "home_team": "PIT",
                "baseline_home_margin": 7.0,
                "baseline_total": 44.0,
            }
        ]
    )
    targets = pl.DataFrame(
        [
            {
                "game_id": "g1",
                "gameday": "2026-10-04",
                "gametime": "13:00",
                "away_team": "NYJ",
                "home_team": "PIT",
            }
        ]
    )
    markets = [
        _market(book="Caesars", provider="espn", home_odds=-160, away_odds=140),
        _market(book="Caesars", provider="the_odds_api", home_odds=-155, away_odds=135),
        _market(book="Book B", provider="the_odds_api", home_odds=-120, away_odds=105),
    ]
    frame, meta = build_market_intelligence(
        projection,
        targets,
        markets,
        _historical_games(),
    )
    assert frame.height == 1
    row = frame.row(0, named=True)
    assert row["quant_book"] == "Book B"
    assert row["market_book_count"] == 2
    assert meta["multi_book_coverage"] == 1.0


def test_paper_mode_keeps_research_picks_when_production_markets_are_disabled() -> None:
    projection = pl.DataFrame(
        [
            {
                "season": 2026,
                "week": 4,
                "game_id": "g1",
                "gameday": "2026-10-04",
                "away_team": "NYJ",
                "home_team": "PIT",
                "baseline_home_margin": 7.0,
                "baseline_total": 44.0,
            }
        ]
    )
    targets = pl.DataFrame(
        [
            {
                "game_id": "g1",
                "gameday": "2026-10-04",
                "gametime": "13:00",
                "away_team": "NYJ",
                "home_team": "PIT",
            }
        ]
    )
    policy = deepcopy(DEFAULT_POLICY)
    policy["deployment_mode"] = "paper"
    for config in policy["markets"].values():
        config["enabled"] = False
        config["disabled_reason"] = "production evidence gate not passed"

    frame, _ = build_market_intelligence(
        projection,
        targets,
        [_market(book="Book B", provider="espn", home_odds=-120, away_odds=105)],
        _historical_games(),
        policy=policy,
    )
    row = frame.row(0, named=True)
    assert row["quant_signal"] == "PASS"
    assert row["research_signal"] in {"LEAN", "BET", "STRONG"}
    assert row["research_stake_units"] > 0

    allocated, summary = apply_portfolio_controls(
        frame,
        policy=policy,
        release_gate={"release_state": "PAPER", "production_eligible": False},
        now=datetime(2026, 10, 1, 15, 5, tzinfo=UTC),
    )
    decision = allocated.row(0, named=True)
    assert decision["portfolio_signal"] == row["research_signal"]
    assert decision["portfolio_action"] == "PAPER"
    assert decision["portfolio_candidate_units"] > 0
    assert summary["bets"] == 1


def test_regime_reliability_blocks_production_signal_when_missing() -> None:
    projection = pl.DataFrame(
        [
            {
                "season": 2026,
                "week": 4,
                "game_id": "g1",
                "gameday": "2026-10-04",
                "away_team": "NYJ",
                "home_team": "PIT",
                "baseline_home_margin": 7.0,
                "baseline_total": 44.0,
            }
        ]
    )
    targets = pl.DataFrame(
        [
            {
                "game_id": "g1",
                "gameday": "2026-10-04",
                "gametime": "13:00",
                "away_team": "NYJ",
                "home_team": "PIT",
            }
        ]
    )
    policy = deepcopy(DEFAULT_POLICY)
    policy["regime_reliability"] = {
        "status": "READY",
        "operational_ready": False,
        "fail_closed": True,
        "required_dimensions_by_market": {
            "moneyline": [
                "market",
                "side",
                "role",
                "confidence",
                "edge",
                "margin_environment",
                "season_phase",
            ]
        },
        "segments": {},
    }

    frame, _ = build_market_intelligence(
        projection,
        targets,
        [_market(book="Book B", provider="espn", home_odds=-120, away_odds=105)],
        _historical_games(),
        policy=policy,
    )
    row = frame.row(0, named=True)

    assert row["research_signal"] in {"LEAN", "BET", "STRONG"}
    assert row["production_signal"] == "PASS"
    assert row["quant_signal"] == "PASS"
    assert row["regime_reliability_ready"] is False
    assert row["regime_reliability_missing_segments"]


def test_current_odds_api_uses_bookmaker_last_update() -> None:
    payload = [
        {
            "id": "event-1",
            "home_team": "Pittsburgh Steelers",
            "away_team": "New York Jets",
            "bookmakers": [
                {
                    "key": "book-a",
                    "title": "Book A",
                    "last_update": "2026-10-01T14:22:00Z",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Pittsburgh Steelers", "price": -130},
                                {"name": "New York Jets", "price": 115},
                            ],
                        }
                    ],
                }
            ],
        }
    ]
    client = CurrentOddsAPIClient(api_key="test", fetch_json=lambda _: payload)
    targets = pl.DataFrame(
        [{"game_id": "g1", "home_team": "PIT", "away_team": "NYJ"}]
    )
    markets = client.current_markets(targets)
    assert len(markets) == 1
    assert markets[0].captured_at == datetime(2026, 10, 1, 14, 22, tzinfo=UTC)


def test_collect_current_markets_does_not_double_count_same_book() -> None:
    class FakeESPN:
        def current_markets(self, targets: pl.DataFrame, *, week: int):
            del targets, week
            return [_market(book="Caesars", provider="espn", home_odds=-130, away_odds=115)]

    class FakePro:
        configured = True

        def current_markets(self, targets: pl.DataFrame):
            del targets
            return [
                _market(
                    book="Caesars",
                    provider="the_odds_api",
                    home_odds=-125,
                    away_odds=110,
                )
            ]

    targets = pl.DataFrame(
        [{"game_id": "g1", "home_team": "PIT", "away_team": "NYJ"}]
    )
    markets, meta = collect_current_markets(
        targets,
        week=4,
        espn_client=FakeESPN(),
        pro_client=FakePro(),
    )
    assert len(markets) == 2
    assert meta["books"] == 1
    assert meta["multi_book_coverage"] == 0.0


def test_backtest_runtime_uses_week_blocks_and_normalizes_clv() -> None:
    rows = []
    projections = []
    for index in range(40):
        week = index // 4 + 1
        market = ("moneyline", "spread", "total")[index % 3]
        side = "home" if market != "total" else "over"
        line = None if market == "moneyline" else (-3.0 if market == "spread" else 44.5)
        raw_clv = 0.02 if market == "moneyline" else 1.5
        rows.append(
            {
                "season": 2025,
                "week": week,
                "game_id": f"g{index}",
                "market_type": market,
                "side": side,
                "line": line,
                "american_odds": -110,
                "probability_edge": 0.04,
                "expected_value_per_unit": 0.03,
                "result": "win" if index % 2 == 0 else "loss",
                "net_units": 0.91 if index % 2 == 0 else -1.0,
                "clv_proxy": raw_clv,
            }
        )
        projections.append(
            {
                "projected_home_margin": float((index % 5) - 2),
                "actual_home_margin": float((index % 5) - 2) + float((index % 4) - 1),
                "projected_total": 44.0,
                "actual_total": 44.0 + float((index % 6) - 3),
            }
        )
    enriched, report = build_backtest_runtime_report(
        pl.DataFrame(rows),
        pl.DataFrame(projections),
    )
    assert report["overall"]["roi_ci_95"] != [None, None]
    assert "favorite" in report["by_role"]
    spread = enriched.filter(pl.col("market_type") == "spread").row(0, named=True)
    assert 0 < spread["clv_probability_equivalent"] < spread["clv_raw"]
    assert normalize_clv_proxy(0.02, "moneyline", margin_sigma=12, total_sigma=12) == 0.02


def test_weekly_publication_labels_shadow_as_non_production(tmp_path) -> None:
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
        "calibrated_home_probability": 0.63,
        "quant_signal": "BET",
        "quant_book": "Book A",
        "quant_odds": -110,
        "portfolio_candidate_units": 0.2,
        "portfolio_stake_units": 0.0,
        "portfolio_action": "SHADOW",
        "execution_ready": True,
        "context_quality": 1.0,
    }
    current = pl.DataFrame(
        [
            {**base, "quant_market": "moneyline", "quant_side": "home", "quant_price": None},
            {**base, "quant_market": "spread", "quant_side": "home", "quant_price": -2.5},
            {**base, "quant_market": "total", "quant_side": "over", "quant_price": 42.5},
        ]
    )
    result = write_weekly_publication(
        current,
        week=4,
        updated_at="2026-10-01T15:00:00+00:00",
        release_state="SHADOW",
        output_dir=tmp_path,
    )
    html = (tmp_path / "nfl_week_4.html").read_text()
    assert "SHADOW evidence mode" in html
    assert "not production staking" in html
    # SHADOW/RESEARCH pages never promote unvalidated raw bets.
    assert '<span class="badge bet">BET</span>' not in html
    assert "BET/SHADOW" not in html
    assert result["board_games"] == 1
    assert (tmp_path / "nfl_week_4_page1.png").exists()
