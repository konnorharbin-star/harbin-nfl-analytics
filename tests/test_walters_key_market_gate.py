"""Research-only key-number spread economics and strict independent quote gate."""
import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from scripts.forward_price_validation import first_commit_time
from scripts.walters_key_market_gate import (
    audit,
    blockers,
    compare,
    outcomes,
    price_ev,
)

NOW = datetime(2026, 10, 10, 19, 0, tzinfo=UTC)
KICKOFF = NOW + timedelta(days=1)


def stamp(dt):
    return dt.isoformat()


def forecast():
    return {
        "spec": "walters_key_number_forward_v1",
        "release_state": "SHADOW",
        "sportsbook_odds_used": False,
        "fixed_alpha": 1.0,
        "season": 2026,
        "sport": "nfl",
        "game_id": "2026_05_BAL_ATL",
        "home_team": "ATL",
        "away_team": "BAL",
        "kickoff": stamp(KICKOFF),
        "captured_at": stamp(NOW),
        "model_mean": .3,
        "model_sigma": 13.0,
        "independent_home_margin": 5.5,
        "key_multipliers": {"3": 2.0, "7": 1.5},
    }


def quote():
    return {
        "spec": "walters_key_spread_quote_v1",
        "quote_id": "atl_balance_001",
        "game_id": "2026_05_BAL_ATL",
        "sport": "nfl",
        "home_team": "ATL",
        "away_team": "BAL",
        "kickoff": stamp(KICKOFF),
        "book": "ExampleVerifiedBook",
        "source_url": "https://book.example.test/events/456",
        "source_evidence_note": "Checked exact sportsbook two-side screen manually at capture.",
        "source_quote_at": stamp(NOW + timedelta(minutes=4)),
        "captured_at": stamp(NOW + timedelta(minutes=5)),
        "home_spread": -3.0,
        "away_spread": 3.0,
        "home_american_odds": -110,
        "away_american_odds": -110,
        "book_identity_verified": True,
        "source_quote_time_verified": True,
        "executable_price_verified": True,
        "book_access_verified": True,
        "settlement_rules_verified": True,
    }


def publisher(path):
    return NOW + (
        timedelta(minutes=6) if path.parent.name == "book_quotes"
        else timedelta(minutes=1)
    )


def write(root, f=None, q=None):
    forecast_dir = root / "forecasts"
    quote_dir = root / "book_quotes"
    forecast_dir.mkdir(parents=True, exist_ok=True)
    quote_dir.mkdir(parents=True, exist_ok=True)
    if f is not None:
        (forecast_dir / (f["game_id"] + ".json")).write_text(json.dumps(f))
    if q is not None:
        (quote_dir / (q["quote_id"] + ".json")).write_text(json.dumps(q))
    return forecast_dir, quote_dir


def test_exact_negative_three_push_returns_stake_not_loss():
    p = [0.0] * 201
    p[103] = .1
    p[106] = .5
    p[99] = .4
    sides = outcomes(p, -3)
    assert sides["home"] == pytest.approx([.5, .1, .4])
    assert sides["away"] == pytest.approx([.4, .1, .5])
    assert price_ev(sides["home"], -110) == pytest.approx(.5 * 100 / 110 - .4)
    wrong_binary = .5 * (1 + 100 / 110) - 1
    assert price_ev(sides["home"], -110) - wrong_binary == pytest.approx(.1)
    assert sum(sides["away"]) == pytest.approx(1.0)


def test_at_half_point_no_push_and_opposite_bet_is_complement():
    p = [1 / 201] * 201
    both = outcomes(p, -3.5)
    assert both["home"][1] == 0
    assert both["away"][1] == 0
    assert both["home"][0] == pytest.approx(both["away"][2])
    assert both["away"][0] == pytest.approx(both["home"][2])


def test_actual_price_comparison_never_issues_bet():
    report = compare(forecast(), quote())
    assert report["wager_authorized"] is False
    assert report["status"] == "ATTESTED_REVIEW_ONLY_NO_ECONOMIC_EDGE_PROVEN"
    for name in ("discrete_gaussian", "key_number"):
        home = report["models"][name]["home"]
        away = report["models"][name]["away"]
        assert home["push"] == away["push"] > 0
        assert home["win"] + home["push"] + home["loss"] == pytest.approx(1.0)
        assert home["market_break_even_probability"] == pytest.approx(110 / 210)
        assert home["market_conditional_no_vig_probability"] == pytest.approx(.5)
    assert report["models"]["key_number"]["home"]["push"] != pytest.approx(
        report["models"]["discrete_gaussian"]["home"]["push"]
    )


def test_unverified_quotes_and_corrupt_lines_fail_closed():
    f, q = forecast(), quote()
    assert not blockers(q, f, publisher(Path("book_quotes/x")),
                        publisher(Path("forecasts/y")), "atl_balance_001")
    q["executable_price_verified"] = False
    q["home_spread"] = -3.25
    q["home_american_odds"] = -99
    reason = blockers(q, f, publisher(Path("book_quotes/x")),
                      publisher(Path("forecasts/y")), "atl_balance_001")
    assert "UNATTESTED_EXECUTABLE_PRICE_VERIFIED" in reason
    assert "INVALID_SAME_BOOK_PAIRED_PRICE" in reason


def test_price_must_be_after_published_independent_model():
    f, q = forecast(), quote()
    result = blockers(q, f, NOW + timedelta(minutes=6),
                      NOW + timedelta(minutes=10), q["quote_id"])
    assert "MODEL_NOT_PUBLISHED_BEFORE_PRICE_OBSERVATION" in result
    q["source_quote_at"] = stamp(NOW - timedelta(hours=2))
    result = blockers(q, f, NOW + timedelta(minutes=6),
                      NOW + timedelta(minutes=1), q["quote_id"])
    assert "PROVIDER_QUOTE_ORIGIN_STALE" in result
    result = blockers(q, f, None, None, q["quote_id"])
    assert "QUOTE_NOT_PUBLISHED_PREGAME" in result
    assert "MODEL_NOT_PUBLISHED_BEFORE_PRICE_OBSERVATION" in result


def test_audit_never_makes_up_a_quote_from_first_published_score(tmp_path):
    fr, qr = write(tmp_path, forecast())
    report = audit(fr, qr, commit_lookup=publisher)
    assert report["published_pregame_forecasts"] == 1
    assert report["quote_files"] == 0
    assert report["comparisons"] == []
    assert report["betting_authorized"] is False
    assert report["historical_profitability_and_clv_verified"] is False


def test_independently_attested_immutable_pair_is_still_not_actionable(tmp_path):
    fr, qr = write(tmp_path, forecast(), quote())
    report = audit(fr, qr, commit_lookup=publisher)
    assert report["published_pregame_forecasts"] == 1
    assert report["attested_two_sided_quote_observations"] == 1
    assert report["comparisons"][0]["wager_authorized"] is False
    assert report["status"] == "RESEARCH_ONLY_NOT_A_VALIDATED_ECONOMIC_EDGE"
    assert report["book_execution_independently_confirmed_by_code"] is False
    assert report["actual_wagers"] == 0


def test_mismatched_forecast_identity_cannot_cross_game(tmp_path):
    bad_quote = quote()
    bad_quote["kickoff"] = stamp(KICKOFF + timedelta(minutes=10))
    fr, qr = write(tmp_path, forecast(), bad_quote)
    report = audit(fr, qr, commit_lookup=publisher)
    assert report["attested_two_sided_quote_observations"] == 0
    assert "WRONG_GAME_OR_KICKOFF" in report["blocked_quote_files"][
        "atl_balance_001.json"
    ]


def test_git_addition_timestamp_remains_traceable_after_second_commit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    def git(*args):
        return subprocess.run(["git", *args], check=True,
                              capture_output=True, text=True).stdout.strip()
    git("init", "-b", "main")
    git("config", "user.name", "Prospective Test")
    git("config", "user.email", "test@example.invalid")
    root = Path("history/walters_key_forward_v1/forecasts")
    root.mkdir(parents=True)
    first = root / "test_game.json"
    first.write_text('{"immutable": true}\n')
    git("add", str(first))
    git("commit", "-m", "first original pregame publication")
    earliest = first_commit_time(first)
    assert earliest is not None
    Path("later.txt").write_text("separate published update")
    git("add", "later.txt")
    git("commit", "-m", "later pipeline run")
    assert first_commit_time(first) == earliest
    first.write_text('{"immutable": false}\n')
    assert first_commit_time(first) is None


def test_repo_model_writer_has_full_git_history_for_forward_grading():
    root = Path(__file__).resolve().parents[1]
    for filename in ("nfl-model.yml", "cfb-model.yml"):
        wf = root / ".github" / "workflows" / filename
        if not wf.exists():
            continue
        text = wf.read_text()
        assert "fetch-depth: 0" in text
    assert "fetch-depth: 0" in (
        root / ".github" / "workflows" / "walters-key-forward.yml"
    ).read_text()
