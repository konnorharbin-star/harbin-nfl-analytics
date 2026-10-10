from datetime import UTC, datetime

import pytest

from scripts.recommendation_ledger import archive_decision

NOW = datetime(2026, 10, 10, 12, tzinfo=UTC)


def candidate(**changes):
    row = {
        "sport": "nfl",
        "game_id": "n1",
        "decision": "BET_READY",
        "bet_approved": True,
        "kickoff": "2026-10-11T18:00:00Z",
        "quoted_at": "2026-10-10T11:30:00Z",
        "quote_observed_at": "2026-10-10T11:31:00Z",
        "market": "spread",
        "side": "home",
        "line": -3.5,
        "american_odds": -110,
        "book": "DraftKings",
        "model_version": "test-only",
        "source_url": "https://example.com/fixture",
        "confidence": "fixture",
        "risks": ["fixture"],
        "model_probability": 0.60,
        "conservative_probability": 0.56,
        "reasons": "",
    }
    row.update(changes)
    return row


def test_receipt_idempotency_preserves_exact_bytes(tmp_path):
    first = archive_decision(
        candidate(), gate={"production_eligible": True}, root=tmp_path, observed_at=NOW
    )
    path = tmp_path / "bets" / (first["bet_id"] + ".json")
    original = path.read_bytes()
    second = archive_decision(
        candidate(), gate={"production_eligible": True}, root=tmp_path, observed_at=NOW
    )
    assert first == second
    assert path.read_bytes() == original
    assert first["stake_units"] == 1
    assert first["publication_status"] == "PENDING_GITHUB_COMMIT"
    assert first["calculated_ev"] == pytest.approx(0.6 * 100 / 110 - 0.4)


@pytest.mark.parametrize(
    "changes",
    [
        {"source_url": ""},
        {"model_version": None},
        {"quote_observed_at": None},
        {"kickoff": "2026-10-10T11:00:00Z"},
        {"quoted_at": "2026-10-10T12:01:00Z"},
        {"conservative_probability": 0.50},
        {"american_odds": 0},
        {"model_probability": float("nan")},
    ],
)
def test_incomplete_or_invalid_receipt_never_bet(tmp_path, changes):
    row = candidate(**changes)
    # Nonfinite input must fail closed before it can be serialized into any receipt.
    if changes.get("model_probability") != changes.get("model_probability"):
        with pytest.raises(ValueError):
            archive_decision(
                row, gate={"production_eligible": True}, root=tmp_path, observed_at=NOW
            )
    else:
        receipt = archive_decision(
            row, gate={"production_eligible": True}, root=tmp_path, observed_at=NOW
        )
        assert receipt["decision"] == "NO_BET"
        assert receipt["stake_units"] == 0
        assert not list((tmp_path / "bets").glob("*.json"))


def test_release_blocked_decision_recorded_separately(tmp_path):
    r = archive_decision(
        candidate(), gate={"production_eligible": False}, root=tmp_path, observed_at=NOW
    )
    assert r["decision"] == "NO_BET"
    assert "GLOBAL_RELEASE_BLOCKED" in r["blockers"]
