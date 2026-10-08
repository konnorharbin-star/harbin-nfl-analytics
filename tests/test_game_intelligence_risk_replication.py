from __future__ import annotations

import polars as pl
import pytest

from nfl.contracts import DataContractError
from nfl.game_intelligence_risk_replication import audit_risk_combinations


def _fixture() -> pl.DataFrame:
    rows = []
    for year in (2024, 2025):
        for i in range(140):
            risk = i < 50
            rows.append({
                "season": year, "week": 5 + i // 10,
                "game_id": f"{year}_g{i}",
                "last_observed_qb_switch": risk,
                "historical_qb_sack_exposure": risk,
                "pregame_ol_injury_stress": risk,
                "margin_residual": 18.0 if risk else 2.0,
                "total_residual": 19.0 if risk else 2.0,
            })
    return pl.DataFrame(rows)


def test_joint_exposure_requires_both_seasons():
    report = audit_risk_combinations(_fixture())
    for target in ("margin", "total"):
        joint = report["targets"][target]["qb_sack_and_ol"]
        assert joint["status"] == "REPLICATED_DESCRIPTIVE_ASSOCIATION"
        assert joint["by_season"]["2024"]["joint_risk_present"]["games"] == 50
        assert joint["staking_authorized"] is False


def test_future_season_outcomes_cannot_rewrite_earlier_season():
    original = _fixture()
    changed = original.with_columns(
        pl.when(pl.col("season") == 2025).then(0)
        .otherwise(pl.col("margin_residual")).alias("margin_residual")
    )
    a = audit_risk_combinations(original)
    b = audit_risk_combinations(changed)
    assert a["targets"]["margin"]["qb_and_sack"]["by_season"]["2024"] == (
        b["targets"]["margin"]["qb_and_sack"]["by_season"]["2024"]
    )


def test_partial_or_missing_ol_never_becomes_healthy():
    original = _fixture()
    missing = original.with_columns(
        pl.when(pl.col("season") == 2025).then(None)
        .otherwise(pl.col("pregame_ol_injury_stress"))
        .alias("pregame_ol_injury_stress")
    )
    report = audit_risk_combinations(missing)
    joint = report["targets"]["margin"]["qb_sack_and_ol"]
    assert joint["status"] == "NOT_REPLICATED_OR_UNDERPOWERED"
    assert joint["by_season"]["2025"]["missing_games"] == 140


def test_duplicate_keys_fail_closed():
    original = _fixture()
    with pytest.raises(DataContractError):
        audit_risk_combinations(pl.concat([original, original.head(1)]))
