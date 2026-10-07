from __future__ import annotations

import polars as pl

from nfl.candidate_ledger import append_candidate_observations


def test_candidate_ledger_persists_blocked_model_candidate(tmp_path) -> None:
    path = tmp_path / "candidate.csv"
    frame = pl.DataFrame(
        [
            {
                "game_id": "g1",
                "quant_market": "spread",
                "model_candidate_signal": "BET",
                "candidate_watch": True,
                "candidate_state": "WATCH_BLOCKED",
                "candidate_block_reason": "injury report pending",
                "selected_quote_outlier": False,
            }
        ]
    )

    first = append_candidate_observations(
        frame,
        path=path,
        observed_at="2026-10-07T00:00:00+00:00",
    )
    second = append_candidate_observations(
        frame,
        path=path,
        observed_at="2026-10-07T00:01:00+00:00",
    )

    assert first["eligible_rows"] == 1
    assert first["appended_rows"] == 1
    assert second["appended_rows"] == 0
    stored = pl.read_csv(path)
    assert stored.height == 1
    assert stored.row(0, named=True)["candidate_state"] == "WATCH_BLOCKED"


def test_candidate_ledger_persists_quote_outlier_even_without_signal(tmp_path) -> None:
    path = tmp_path / "candidate.csv"
    frame = pl.DataFrame(
        [
            {
                "game_id": "g1",
                "quant_market": "moneyline",
                "model_candidate_signal": "PASS",
                "selected_quote_outlier": True,
                "selected_quote_consensus_distance": 0.15,
            }
        ]
    )

    result = append_candidate_observations(
        frame,
        path=path,
        observed_at="2026-10-07T00:00:00+00:00",
    )

    assert result["eligible_rows"] == 1
    assert result["appended_rows"] == 1
