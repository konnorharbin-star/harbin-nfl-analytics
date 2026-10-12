"""Fixtures are synthetic; no real bettor records or sportsbook fills are invented."""
from datetime import datetime, timedelta, timezone
import pytest
from scripts.professional_signal_shadow import audit


def rec(i=0, outcome="WIN", league="NFL", analyst="FIXTURE"):
    t = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=i * 2)
    stamp = lambda d: d.isoformat()
    line = "70.5"
    return {"record_id": f"synthetic-{i}", "analyst": analyst, "league": league,
            "event_id": f"event-{i}", "kickoff_utc": stamp(t + timedelta(hours=6)),
            "published_utc": stamp(t), "captured_utc": stamp(t + timedelta(minutes=1)),
            "source_url": "https://example.org/individual-pick", "market": "PLAYER_RECEIVING_YARDS",
            "period": "FULL_GAME", "participant": "synthetic-player", "selection": "OVER", "line": line,
            "market_pair": {"bookmaker": "SYNTHETIC_BOOK", "region": "US",
                            "snapshot_id": f"sample-{i}",
                            "quote_utc": stamp(t + timedelta(minutes=5)),
                            "captured_utc": stamp(t + timedelta(minutes=6)),
                            "source_url": "https://example.org/market-snapshot",
                            "sides": [{"selection": "OVER", "line": line, "american_odds": -110},
                                      {"selection": "UNDER", "line": line, "american_odds": -110}]},
            "outcome": outcome,
            "settled_utc": None if outcome == "PENDING" else stamp(t + timedelta(hours=10)),
            "independent_model_p": 0.53,
            "independent_model_frozen_utc": stamp(t - timedelta(hours=1))}


def test_empty_cannot_claim_profit_or_signal():
    result = audit([], "NFL")
    assert result["decision"] == "PASS" and not result["bet_authorized"]
    assert result["status"] == "NO_REAL_SOURCE_RECORDS"


def test_exact_market_pair_and_no_vig():
    result = audit([rec()], "NFL")
    row = result["records"][0]
    assert row["market_p"] == pytest.approx(0.5)
    assert row["expert_offset"] == 0
    assert not row["executable_account_price_verified"]
    assert result["model_weights_changed"] is False


def test_future_outcomes_cannot_train_past_predictions():
    sample = [rec(i) for i in range(27)]
    rows = audit(sample, "NFL")["records"]
    assert rows[0]["frozen_prior_picks"] == 0
    assert rows[24]["frozen_prior_picks"] == 24
    assert rows[25]["frozen_prior_picks"] == 25
    assert rows[25]["expert_offset"] > 0
    assert audit(list(reversed(sample)), "NFL")["records"] == rows


def test_unsettled_games_cannot_train_source_weights():
    rows = audit([rec(i, "PENDING") for i in range(30)], "NFL")["records"]
    assert all(row["expert_offset"] == 0 and row["frozen_prior_picks"] == 0 for row in rows)


def test_analysts_train_separately():
    sample = [rec(i) for i in range(30)] + [rec(40, analyst="OTHER")]
    assert audit(sample, "NFL")["records"][-1]["frozen_prior_picks"] == 0


def test_mismatched_market_line_rejected():
    row = rec()
    row["market_pair"]["sides"][1]["line"] = "71.5"
    with pytest.raises(ValueError, match="threshold"):
        audit([row], "NFL")


def test_after_kickoff_social_post_rejected():
    row = rec()
    row["published_utc"] = row["kickoff_utc"]
    row["captured_utc"] = row["kickoff_utc"]
    with pytest.raises(ValueError, match="before kickoff"):
        audit([row], "NFL")


def test_only_after_post_market_quote_allowed():
    row = rec()
    row["market_pair"]["quote_utc"] = "2025-12-31T23:00:00Z"
    with pytest.raises(ValueError, match="contemporaneous"):
        audit([row], "NFL")


def test_independent_model_must_predate_social_post():
    row = rec()
    row["independent_model_frozen_utc"] = row["kickoff_utc"]
    with pytest.raises(ValueError, match="not frozen"):
        audit([row], "NFL")


def test_binary_shadow_rejects_integer_player_lines():
    row = rec()
    row["line"] = "70"
    for side in row["market_pair"]["sides"]:
        side["line"] = "70"
    with pytest.raises(ValueError, match="half-point"):
        audit([row], "NFL")


def test_duplicate_and_wrong_league_fail_closed():
    with pytest.raises(ValueError, match="duplicate record_id"):
        audit([rec(), rec()], "NFL")
    with pytest.raises(ValueError, match="unsupported league"):
        audit([rec()], "CFB")


def test_cfb_separate_evaluation_supported():
    assert audit([rec(league="CFB")], "CFB")["records"][0]["market_p"] == 0.5


def test_cannot_use_game_result_that_finishes_after_next_post():
    first = rec(0)
    second = rec(0)
    second["record_id"] = "other-id"
    second["event_id"] = "other-event"
    second["published_utc"] = "2026-01-01T02:00:00+00:00"
    second["captured_utc"] = "2026-01-01T02:01:00+00:00"
    second["market_pair"]["quote_utc"] = "2026-01-01T02:05:00+00:00"
    second["market_pair"]["captured_utc"] = "2026-01-01T02:06:00+00:00"
    assert audit([first, second], "NFL")["records"][1]["frozen_prior_picks"] == 0


def test_cli_filters_mixed_leagues_and_fails_on_unknown(tmp_path):
    import json
    import subprocess
    import sys
    source = tmp_path / "mixed.json"
    output = tmp_path / "out.json"
    rows = [rec(0, league="NFL"), rec(1, league="CFB")]
    source.write_text(json.dumps({"records": rows}))
    result = subprocess.run([sys.executable, "-m", "scripts.professional_signal_shadow",
                             "--league", "NFL", "--input", str(source), "--output", str(output)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    saved = json.loads(output.read_text())
    assert saved["league"] == "NFL" and len(saved["records"]) == 1
    source.write_text(json.dumps({"records": [{**rec(0), "league": "NBA"}]}))
    result = subprocess.run([sys.executable, "-m", "scripts.professional_signal_shadow",
                             "--league", "NFL", "--input", str(source), "--output", str(output)],
                            capture_output=True, text=True)
    assert result.returncode != 0
