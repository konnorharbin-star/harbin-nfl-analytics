from datetime import UTC, datetime

import polars as pl

from nfl.qb_current import (
    apply_qb_certainty_veto,
    build_expected_qb_state,
)


AS_OF = datetime(2026, 10, 8, 16, tzinfo=UTC)


def _targets() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "game_id": "2026_05_AAA_BBB",
                "home_team": "BBB",
                "away_team": "AAA",
            }
        ]
    )


def _projection() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "game_id": "2026_05_AAA_BBB",
                "home_team": "BBB",
                "away_team": "AAA",
                "home_qb_proxy_id": "qb-bbb-1",
                "home_qb_proxy_name": "BBB One",
                "away_qb_proxy_id": "qb-aaa-1",
                "away_qb_proxy_name": "AAA One",
            }
        ]
    )


def _depth() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "team": "AAA",
                "player_key": "id:qb-aaa-1",
                "gsis_id": "qb-aaa-1",
                "player_name": "AAA One",
                "position": "QB",
                "depth_rank": 1,
                "depth_week": None,
                "depth_captured_at": "2026-10-07T14:00:00+00:00",
            },
            {
                "team": "BBB",
                "player_key": "id:qb-bbb-1",
                "gsis_id": "qb-bbb-1",
                "player_name": "BBB One",
                "position": "QB",
                "depth_rank": 1,
                "depth_week": None,
                "depth_captured_at": "2026-10-07T14:00:00+00:00",
            },
        ]
    )


def _rosters() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "team": "AAA",
                "player_key": "id:qb-aaa-1",
                "gsis_id": "qb-aaa-1",
                "player_name": "AAA One",
                "position": "QB",
                "roster_status": "Active",
                "active": True,
                "status_known": True,
            },
            {
                "team": "BBB",
                "player_key": "id:qb-bbb-1",
                "gsis_id": "qb-bbb-1",
                "player_name": "BBB One",
                "position": "QB",
                "roster_status": "Active",
                "active": True,
                "status_known": True,
            },
        ]
    )


def _injuries() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "team": pl.String,
            "player_key": pl.String,
            "gsis_id": pl.String,
            "player_name": pl.String,
            "position": pl.String,
            "report_status": pl.String,
            "practice_status": pl.String,
            "severity": pl.Float64,
        }
    )


def test_expected_qb_uses_current_qb1_not_last_game_guess() -> None:
    state, meta = build_expected_qb_state(
        _targets(),
        _projection(),
        depth=_depth(),
        rosters=_rosters(),
        injuries=_injuries(),
        as_of=AS_OF,
    )

    aaa = state.filter(pl.col("team") == "AAA").row(0, named=True)
    assert aaa["expected_qb_id"] == "qb-aaa-1"
    assert aaa["expected_qb_source"] == "depth_chart"
    assert aaa["expected_qb_confidence_label"] == "HIGH"
    assert aaa["expected_qb_matches_last_observed"] is True
    assert aaa["expected_qb_decision_ready"] is True
    assert meta["identity_coverage"] == 1.0
    assert meta["decision_ready_team_coverage"] == 1.0


def test_out_qb1_selects_depth_replacement_but_blocks_betting() -> None:
    depth = pl.concat(
        [
            _depth(),
            pl.DataFrame(
                [
                    {
                        "team": "AAA",
                        "player_key": "id:qb-aaa-2",
                        "gsis_id": "qb-aaa-2",
                        "player_name": "AAA Two",
                        "position": "QB",
                        "depth_rank": 2,
                        "depth_week": None,
                        "depth_captured_at": "2026-10-07T14:00:00+00:00",
                    }
                ]
            ),
        ],
        how="vertical_relaxed",
    )
    rosters = pl.concat(
        [
            _rosters(),
            pl.DataFrame(
                [
                    {
                        "team": "AAA",
                        "player_key": "id:qb-aaa-2",
                        "gsis_id": "qb-aaa-2",
                        "player_name": "AAA Two",
                        "position": "QB",
                        "roster_status": "Active",
                        "active": True,
                        "status_known": True,
                    }
                ]
            ),
        ],
        how="vertical_relaxed",
    )
    injuries = pl.DataFrame(
        [
            {
                "team": "AAA",
                "player_key": "id:qb-aaa-1",
                "gsis_id": "qb-aaa-1",
                "player_name": "AAA One",
                "position": "QB",
                "report_status": "Out",
                "practice_status": "Did Not Participate",
                "severity": 1.0,
            }
        ]
    )

    state, _ = build_expected_qb_state(
        _targets(),
        _projection(),
        depth=depth,
        rosters=rosters,
        injuries=injuries,
        as_of=AS_OF,
    )
    aaa = state.filter(pl.col("team") == "AAA").row(0, named=True)

    assert aaa["expected_qb_id"] == "qb-aaa-2"
    assert aaa["expected_qb_source"] == "depth_chart_replacement"
    assert aaa["expected_qb_changed_from_last_observed"] is True
    assert aaa["expected_qb_decision_ready"] is False
    assert "differs from last-observed QB" in aaa["expected_qb_reason"]


def test_uncertain_qb_state_fails_closed_in_betting_layer() -> None:
    candidates = pl.DataFrame(
        [
            {
                "game_id": "g1",
                "quant_signal": "BET",
                "production_signal": "BET",
                "research_signal": "STRONG",
                "stake_units": 0.5,
                "research_stake_units": 0.5,
                "home_expected_qb_decision_ready": True,
                "away_expected_qb_decision_ready": False,
                "qb_context_reason": "away: unresolved expected starter",
            }
        ]
    )

    row = apply_qb_certainty_veto(candidates).row(0, named=True)

    assert row["qb_certainty_veto"] is True
    assert row["quant_signal"] == "PASS"
    assert row["research_signal"] == "PASS"
    assert row["stake_units"] == 0.0
    assert row["research_stake_units"] == 0.0


def test_missing_expected_qb_columns_also_fail_closed() -> None:
    candidates = pl.DataFrame(
        [
            {
                "game_id": "g1",
                "quant_signal": "BET",
                "research_signal": "BET",
                "stake_units": 0.25,
                "research_stake_units": 0.25,
            }
        ]
    )

    row = apply_qb_certainty_veto(candidates).row(0, named=True)

    assert row["qb_certainty_veto"] is True
    assert row["quant_signal"] == "PASS"
    assert row["research_signal"] == "PASS"
    assert row["stake_units"] == 0.0
