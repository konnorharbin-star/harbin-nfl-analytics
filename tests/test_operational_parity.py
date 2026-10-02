from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl

from nfl.decision_ledger import append_portfolio_decisions
from nfl.espn_market import ESPNTwoWayMarket
from nfl.execution_market import validate_execution_row
from nfl.grading import grade_portfolio_decisions, summarize_live_grading
from nfl.line_history import append_market_snapshots, load_market_snapshots
from nfl.policy import DEFAULT_POLICY, fractional_kelly_units, signal_from_policy
from nfl.portfolio import apply_portfolio_controls
from nfl.release_gate import build_release_gate


def _candidate(**overrides: object) -> dict[str, object]:
    now = datetime.now(UTC)
    row: dict[str, object] = {
        "season": 2026,
        "week": 4,
        "game_id": "2026_04_AAA_BBB",
        "date": "2026-10-04",
        "kickoff": "2026-10-04T18:00:00+00:00",
        "away_team": "AAA",
        "home_team": "BBB",
        "model_margin_home": 3.0,
        "model_total": 45.0,
        "calibrated_home_probability": 0.60,
        "quant_signal": "BET",
        "quant_market": "spread",
        "quant_side": "home",
        "quant_book": "ESPN BET",
        "quant_price": -2.5,
        "quant_odds": -110,
        "quant_quote_at": now.isoformat(),
        "quant_probability": 0.58,
        "quant_ev": 0.08,
        "quant_edge": 0.05,
        "market_book_count": 1,
        "stake_units": 0.8,
        "execution_ready": True,
    }
    row.update(overrides)
    return row


def test_policy_and_kelly_are_conservative() -> None:
    assert signal_from_policy(0.0, 0.10, 0.70, "moneyline") == "PASS"
    assert signal_from_policy(0.08, 0.05, 0.58, "spread") == "STRONG"
    portfolio = DEFAULT_POLICY["portfolio"]
    assert isinstance(portfolio, dict)
    assert portfolio["min_market_book_count_for_execution"] == 1
    assert portfolio["require_open_exposure_ledger_for_production"] is True
    units = fractional_kelly_units(0.58, -110, kelly_fraction=0.20, max_units=1.0)
    assert 0.0 < units <= 1.0


def test_execution_market_rejects_stale_quote() -> None:
    now = datetime.now(UTC)
    row = _candidate(quant_quote_at=(now - timedelta(hours=2)).isoformat())
    limits = {"require_quote_timestamp_for_execution": True, "max_quote_age_minutes": 60}
    ready, reason = validate_execution_row(row, limits=limits, now=now)
    assert not ready
    assert "stale" in reason


def test_production_invalid_quote_does_not_consume_portfolio_capacity(tmp_path) -> None:
    now = datetime.now(UTC)
    row = _candidate(quant_quote_at=(now - timedelta(hours=2)).isoformat())
    policy = dict(DEFAULT_POLICY)
    portfolio = DEFAULT_POLICY["portfolio"]
    assert isinstance(portfolio, dict)
    policy["portfolio"] = {
        **portfolio,
        "require_live_history_for_production": False,
        "require_open_exposure_ledger_for_production": False,
    }
    policy["deployment_mode"] = "production"

    frame, summary = apply_portfolio_controls(
        pl.DataFrame([row]),
        policy=policy,
        release_gate={"release_state": "PRODUCTION", "production_eligible": True},
        live_bets_path=tmp_path / "none.csv",
        now=now,
    )

    assert summary["mode"] == "production"
    assert summary["production_gate_open"] is True
    assert summary["production_eligible"] is True
    assert summary["execution_blocked_bets"] == 1
    assert frame.get_column("portfolio_candidate_units").sum() == 0.0
    assert frame.get_column("portfolio_stake_units").sum() == 0.0
    assert frame.get_column("portfolio_action")[0] == "PASS"
    assert "stale" in frame.get_column("portfolio_limit_reason")[0]


def test_production_bankroll_hard_stop_is_explicitly_halted(tmp_path) -> None:
    now = datetime.now(UTC)
    history = tmp_path / "live.csv"
    history.write_text(
        "net_units,execution_clv\n"
        "20.0,0.02\n"
        "-20.0,-0.02\n",
        encoding="utf-8",
    )
    policy = dict(DEFAULT_POLICY)
    portfolio = DEFAULT_POLICY["portfolio"]
    assert isinstance(portfolio, dict)
    policy["portfolio"] = {
        **portfolio,
        "require_open_exposure_ledger_for_production": False,
    }
    policy["deployment_mode"] = "production"

    frame, summary = apply_portfolio_controls(
        pl.DataFrame([_candidate()]),
        policy=policy,
        release_gate={"release_state": "PRODUCTION", "production_eligible": True},
        live_bets_path=history,
        now=now,
    )

    assert summary["production_gate_open"] is True
    assert summary["production_eligible"] is False
    assert summary["mode"] == "halted"
    assert summary["bankroll_risk"]["hard_stop"] is True
    assert "hard stop" in summary["production_block_reason"]
    assert frame.get_column("portfolio_candidate_units").sum() == 0.0
    assert frame.get_column("portfolio_stake_units").sum() == 0.0
    assert frame.get_column("portfolio_action")[0] == "PASS"


def test_production_requires_committed_exposure_ledger(tmp_path) -> None:
    now = datetime.now(UTC)
    policy = dict(DEFAULT_POLICY)
    portfolio = DEFAULT_POLICY["portfolio"]
    assert isinstance(portfolio, dict)
    policy["portfolio"] = {
        **portfolio,
        "require_live_history_for_production": False,
    }
    policy["deployment_mode"] = "production"

    frame, summary = apply_portfolio_controls(
        pl.DataFrame([_candidate()]),
        policy=policy,
        release_gate={"release_state": "PRODUCTION", "production_eligible": True},
        live_bets_path=tmp_path / "none.csv",
        decision_ledger_path=tmp_path / "missing_decisions.csv",
        now=now,
    )

    assert summary["production_gate_open"] is True
    assert summary["production_eligible"] is False
    assert summary["mode"] == "halted"
    assert "committed-exposure ledger" in summary["production_block_reason"]
    assert frame.get_column("portfolio_candidate_units").sum() == 0.0
    assert frame.get_column("portfolio_stake_units").sum() == 0.0


def test_committed_production_exposure_reserves_future_caps(tmp_path) -> None:
    now = datetime.now(UTC)
    decisions = tmp_path / "decisions.csv"
    decisions.write_text(
        "decision_at,game_id,season,week,kickoff,away_team,home_team,"
        "quant_market,quant_side,quant_book,portfolio_stake_units,portfolio_action\n"
        "2026-10-01T12:00:00+00:00,2026_04_AAA_BBB,2026,4,"
        "2026-10-04T18:00:00+00:00,AAA,BBB,spread,home,ESPN BET,0.6,BET\n",
        encoding="utf-8",
    )

    policy = dict(DEFAULT_POLICY)
    portfolio = DEFAULT_POLICY["portfolio"]
    assert isinstance(portfolio, dict)
    policy["portfolio"] = {
        **portfolio,
        "require_live_history_for_production": False,
        "max_slate_units": 1.0,
        "max_game_units": 1.0,
        "max_team_units": 1.5,
        "max_market_units": 2.5,
        "max_book_units": 2.0,
        "max_kickoff_window_units": 2.0,
    }
    policy["deployment_mode"] = "production"

    rows = [
        _candidate(
            game_id="2026_04_AAA_BBB",
            quant_market="spread",
            quant_ev=0.09,
            quant_quote_at=now.isoformat(),
        ),
        _candidate(
            game_id="2026_04_CCC_DDD",
            away_team="CCC",
            home_team="DDD",
            quant_market="total",
            quant_side="over",
            quant_price=44.5,
            quant_ev=0.08,
            stake_units=0.8,
            quant_quote_at=now.isoformat(),
        ),
    ]
    frame, summary = apply_portfolio_controls(
        pl.DataFrame(rows),
        policy=policy,
        release_gate={"release_state": "PRODUCTION", "production_eligible": True},
        live_bets_path=tmp_path / "none.csv",
        decision_ledger_path=decisions,
        now=now,
    )

    same_market = frame.filter(pl.col("game_id") == "2026_04_AAA_BBB")
    new_market = frame.filter(pl.col("game_id") == "2026_04_CCC_DDD")
    assert same_market.get_column("portfolio_action")[0] == "PASS"
    assert same_market.get_column("portfolio_stake_units")[0] == 0.0
    assert "already committed" in same_market.get_column("portfolio_limit_reason")[0]

    assert new_market.get_column("portfolio_action")[0] == "BET"
    assert abs(new_market.get_column("portfolio_stake_units")[0] - 0.4) < 1e-9
    committed = summary["committed_exposure"]
    assert committed["open_bets"] == 1
    assert abs(committed["reserved_open_units"] - 0.6) < 1e-9
    assert committed["total_open_bets_after_allocation"] == 2
    assert committed["total_open_units_after_allocation"] <= 1.0 + 1e-9
    assert abs(summary["approved_units"] - 0.4) < 1e-9


def test_paper_portfolio_applies_game_cap_without_real_stake(tmp_path) -> None:
    now = datetime.now(UTC)
    rows = [
        _candidate(quant_market="spread", stake_units=0.8),
        _candidate(
            quant_market="total",
            quant_side="over",
            quant_price=44.5,
            stake_units=0.8,
            quant_ev=0.07,
        ),
    ]
    policy = dict(DEFAULT_POLICY)
    policy["deployment_mode"] = "paper"
    frame, summary = apply_portfolio_controls(
        pl.DataFrame(rows),
        policy=policy,
        release_gate={"release_state": "PAPER", "production_eligible": False},
        live_bets_path=tmp_path / "none.csv",
        now=now,
    )
    assert frame.get_column("portfolio_candidate_units").sum() <= 1.0 + 1e-9
    assert frame.get_column("portfolio_stake_units").sum() == 0.0
    assert set(frame.get_column("portfolio_action").to_list()) <= {"PAPER", "PASS"}
    assert summary["approved_units"] == 0.0


def test_decision_ledger_is_change_only(tmp_path) -> None:
    path = tmp_path / "decisions.csv"
    frame = pl.DataFrame(
        [
            {
                **_candidate(),
                "portfolio_candidate_units": 0.5,
                "portfolio_stake_units": 0.0,
                "paper_stake_units": 0.5,
                "bankroll_adjusted_units": 0.5,
                "execution_ready": True,
                "portfolio_action": "PAPER",
                "portfolio_limit_reason": "",
            }
        ]
    )
    first = append_portfolio_decisions(frame, path=path, decision_at="2026-10-01T00:00:00+00:00")
    second = append_portfolio_decisions(frame, path=path, decision_at="2026-10-01T01:00:00+00:00")
    assert first["appended_rows"] == 1
    assert second["appended_rows"] == 0


def test_line_history_deduplicates_exact_snapshots(tmp_path) -> None:
    captured = datetime(2026, 10, 1, 12, tzinfo=UTC)
    market = ESPNTwoWayMarket(
        game_id="2026_04_AAA_BBB",
        market_type="spread",
        provider="ESPN",
        book="ESPN BET",
        source_event_id="123",
        captured_at=captured,
        first_side="home",
        first_line=-2.5,
        first_american_odds=-110,
        second_side="away",
        second_line=2.5,
        second_american_odds=-110,
    )
    targets = pl.DataFrame(
        [
            {
                "season": 2026,
                "week": 4,
                "game_id": "2026_04_AAA_BBB",
                "home_team": "BBB",
                "away_team": "AAA",
                "gameday": "2026-10-04",
                "gametime": "13:00",
            }
        ]
    )
    path = tmp_path / "markets.csv"
    first = append_market_snapshots([market], targets, path=path)
    second = append_market_snapshots([market], targets, path=path)
    assert first["appended_rows"] == 1
    assert second["appended_rows"] == 0
    assert load_market_snapshots(path).height == 1


def test_grading_uses_persisted_entry_not_final_market() -> None:
    decision = {
        **_candidate(
            quant_price=3.5,
            quant_side="away",
            quant_quote_at="2026-10-01T11:55:00+00:00",
        ),
        "decision_at": "2026-10-01T12:00:00+00:00",
        "portfolio_candidate_units": 0.5,
        "portfolio_action": "PAPER",
    }
    schedules = pl.DataFrame(
        [
            {
                "game_id": "2026_04_AAA_BBB",
                "gameday": "2026-10-04",
                "home_team": "BBB",
                "away_team": "AAA",
                "home_score": 24.0,
                "away_score": 22.0,
            }
        ]
    )
    graded = grade_portfolio_decisions(pl.DataFrame([decision]), schedules)
    assert graded.height == 1
    assert graded.get_column("result")[0] == "win"
    assert graded.get_column("net_units")[0] > 0
    report = summarize_live_grading(graded)
    assert report["portfolio_verified"] is True
    assert report["graded_bets"] == 1


def test_grading_excludes_invalid_execution_provenance() -> None:
    decision = {
        **_candidate(
            quant_quote_at="2026-10-01T12:05:00+00:00",
            execution_ready=True,
        ),
        "decision_at": "2026-10-01T12:00:00+00:00",
        "portfolio_candidate_units": 0.5,
        "portfolio_action": "PAPER",
    }
    schedules = pl.DataFrame(
        [
            {
                "game_id": "2026_04_AAA_BBB",
                "gameday": "2026-10-04",
                "home_team": "BBB",
                "away_team": "AAA",
                "home_score": 24.0,
                "away_score": 22.0,
            }
        ]
    )

    graded = grade_portfolio_decisions(pl.DataFrame([decision]), schedules)

    assert graded.is_empty()


def test_live_summary_reports_entry_and_clv_coverage() -> None:
    decision = {
        **_candidate(quant_quote_at="2026-10-01T11:55:00+00:00"),
        "decision_at": "2026-10-01T12:00:00+00:00",
        "portfolio_candidate_units": 0.5,
        "portfolio_action": "PAPER",
    }
    schedules = pl.DataFrame(
        [
            {
                "game_id": "2026_04_AAA_BBB",
                "gameday": "2026-10-04",
                "home_team": "BBB",
                "away_team": "AAA",
                "home_score": 24.0,
                "away_score": 22.0,
            }
        ]
    )

    graded = grade_portfolio_decisions(pl.DataFrame([decision]), schedules)
    report = summarize_live_grading(graded)

    assert report["graded_bets"] == 1
    assert report["entry_quote_coverage"] == 1.0
    assert report["execution_ready_coverage"] == 1.0
    assert report["clv_samples"] == 0
    assert report["clv_coverage"] == 0.0


def test_release_gate_cannot_promote_without_independent_evidence(tmp_path) -> None:
    meta = {
        "market_coverage": {"games": 16, "moneyline": 16, "spread": 16, "total": 16},
        "market_intelligence": {"multi_book_coverage": 1.0},
        "current_context": {"coverage": 1.0, "qb_coverage": 1.0},
        "probability": {
            "home_win_brier": 0.23,
            "margin_80_coverage": 0.80,
            "total_80_coverage": 0.80,
        },
    }
    gate = build_release_gate(
        meta,
        {"live_readiness_score": 95},
        {"status": "OK"},
        evidence_path=tmp_path / "missing_evidence.json",
        live_path=tmp_path / "missing_live.json",
    )
    assert gate["release_state"] == "PAPER"
    assert gate["production_eligible"] is False
    assert gate["historical_edge_ready"] is False
