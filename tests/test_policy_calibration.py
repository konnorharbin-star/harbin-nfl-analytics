from __future__ import annotations

import polars as pl

from nfl.policy_calibration import derive_policy_from_frame, derive_production_policy


def _rows(*, evaluation_profit: float = 0.9, verified: bool = True) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for season in (2023, 2024, 2025):
        profit = evaluation_profit if season == 2025 else 0.9
        for index in range(50):
            rows.append(
                {
                    "season": season,
                    "week": 1 + (index % 10),
                    "game_id": f"{season}-{index}",
                    "market_type": "moneyline",
                    "model_probability": 0.65,
                    "probability_edge": 0.08,
                    "expected_value_per_unit": 0.10,
                    "net_units": profit,
                    "clv_proxy": 0.02,
                    "entry_price_verified": verified,
                    "entry_quote_verified": verified,
                    "entry_timestamp_verified": verified,
                }
            )
    return pl.DataFrame(rows)


def test_policy_calibration_fails_closed_without_verified_entries() -> None:
    policy = derive_policy_from_frame(_rows(verified=False))

    assert policy["deployment_mode"] == "paper"
    assert policy["source"] == "paper only; no verified archived opening-entry sample"
    assert policy["diagnostics"]["promotion_rows"] == 0


def test_policy_uses_nested_chronology_and_keeps_release_separate() -> None:
    policy = derive_policy_from_frame(
        _rows(),
        evidence={
            "status": "ROBUST",
            "promotion_sample": {"entry_quote_verified": True},
        },
    )

    moneyline = policy["markets"]["moneyline"]
    assert moneyline["enabled"] is True
    assert policy["diagnostics"]["selection_uses_evaluation"] is False
    assert policy["diagnostics"]["moneyline"]["development_bets"] == 50
    assert policy["diagnostics"]["moneyline"]["tune_bets"] == 50
    assert policy["diagnostics"]["moneyline"]["evaluation_bets"] == 50

    # One passing market cannot independently open production.
    assert policy["deployment_mode"] == "paper"


def test_untouched_evaluation_can_disable_frozen_threshold() -> None:
    policy = derive_policy_from_frame(_rows(evaluation_profit=-1.0))

    moneyline = policy["markets"]["moneyline"]
    assert moneyline["enabled"] is False
    assert moneyline["disabled_reason"] == (
        "frozen threshold failed untouched chronological evaluation"
    )
    assert policy["diagnostics"]["moneyline"]["evaluation_passed"] is False


def test_unverified_profitable_rows_never_select_policy() -> None:
    verified = _rows()
    unverified = _rows(verified=False).with_columns(
        pl.lit(10.0).alias("net_units"),
        pl.lit(0.25).alias("probability_edge"),
        pl.lit(0.40).alias("expected_value_per_unit"),
    )
    policy = derive_policy_from_frame(pl.concat([verified, unverified]))

    assert policy["diagnostics"]["raw_archive_rows"] == 300
    assert policy["diagnostics"]["promotion_rows"] == 150


def test_file_calibration_reads_optional_verified_provider_bets(tmp_path) -> None:
    free_path = tmp_path / "free.csv"
    _rows(verified=False).write_csv(free_path)
    verified_path = tmp_path / "verified.csv"
    _rows().write_csv(verified_path)
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(
        '{"status":"ROBUST","promotion_sample":{"entry_quote_verified":true}}',
        encoding="utf-8",
    )

    policy = derive_production_policy(
        bets_path=free_path,
        verified_bets_path=verified_path,
        evidence_path=evidence_path,
        output_path=tmp_path / "policy.json",
    )

    assert policy["diagnostics"]["promotion_rows"] == 150
    assert policy["diagnostics"]["selection_uses_evaluation"] is False
    assert policy["markets"]["moneyline"]["enabled"] is True
