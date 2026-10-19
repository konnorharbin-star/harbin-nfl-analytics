"""Prospective personnel uncertainty receipts must never rewrite game forecasts."""
from copy import deepcopy
from datetime import UTC, datetime, timedelta
import hashlib
import json

import pytest

from nfl.personnel_asof_watch import watch, append, read_journal

AT = datetime(2026, 10, 19, 19, tzinfo=UTC)
HASH = "a"*64
OTHER = "b"*64


def sources():
    game = dict(
        game_id="2026_06_WAS_SF", season=2026, week=6,
        away_team="WAS", home_team="SF",
        kickoff_utc="2026-10-20T00:15:00+00:00",
        forecast_sha256="f"*64,
        model_capture_utc="2026-10-19T18:45:00+00:00",
        predicted_score=dict(home=26.5, away=21.2),
        fair_home_spread=-5.3, predicted_total=47.7,
    )
    player = dict(
        game_id=game["game_id"], team="SF", player_id="00-123",
        player_name="Example Quarterback", is_unallocated_bucket=False,
        projections={"passing_attempts": {"mean": 29.3}},
    )
    bundle = dict(
        spec="NFL_INDEPENDENT_FORECAST_PUBLICATION_V1",
        model_capture_utc=game["model_capture_utc"],
        generated_utc="2026-10-19T18:46:00+00:00",
        sportsbook_prices_used=False, qualified_bets=0,
        games=[game], player_props=[player],
    )
    player_ctx = dict(
        player_id="00-123", team="SF",
        historical_qb_attempts_proxy_mean=29.3,
        availability_evidence_status="ESPN_REPORTED_INJURY_NOT_FINAL_GAME_STATUS",
        espn_reported_status="Questionable",
        nflverse_report_evidence="SOURCE_DATED_REPORTED_STATUS",
        nflverse_reported_status="questionable",
        nflverse_report_modified_utc="2026-10-19T18:20:00+00:00",
    )
    context = dict(
        version="NFL_PUBLIC_WEB_CONTEXT_RESEARCH_V1",
        captured_utc="2026-10-19T18:50:00+00:00",
        games=[dict(
            game_id=game["game_id"], kickoff_utc=game["kickoff_utc"],
            away_team="WAS", home_team="SF",
        )],
        player_evidence_review={
            "games": [dict(
                game_id=game["game_id"], kickoff_utc=game["kickoff_utc"],
                context_is_supplemental=True,
                teams={"SF": {"players": [player_ctx]}, "WAS": {"players": []}},
            )],
        },
    )
    return bundle, context


def get(b, c, at=AT):
    return watch(b, c, captured=at, bundle_sha=HASH, research_sha=OTHER, run_id="123")


def test_asof_observation_links_exact_player_and_preserves_fair_scores():
    b, c = sources()
    before = deepcopy((b, c))
    report = get(b, c)
    assert (b, c) == before
    assert report["eligible_pregame_games"] == 1
    assert report["starting_qbs_confirmed"] == 0
    r = report["games"][0]
    assert r["our_frozen_at_capture_projection"]["predicted_score"]["home"] == 26.5
    assert r["forecast_sha256"] == "f"*64
    assert r["source_web_context_sha256"] == OTHER
    team = r["teams"]["SF"]
    assert team["historical_passing_volume_proxies"][0]["name"] == "Example Quarterback"
    assert team["starting_qb_id_confirmed_by_first_party"] is None
    assert team["starting_qb_verified"] is False
    assert team["model_player_evidence"][0]["nflverse_reported_status"] == "questionable"
    assert team["model_player_evidence"][0]["official_starting_qb_verified"] is False
    assert team["model_player_evidence"][0]["point_adjustment"] is None
    assert r["model_parameters_changed"] is False and r["valid_bet_identified"] is False
    assert report["betting_authorized"] is False


def test_journal_is_append_only_and_original_jsonl_line_stays_identical(tmp_path):
    b, c = sources()
    path = tmp_path/"history"/"observations.jsonl"
    first = get(b, c)
    a = append(first, path)
    original = path.read_bytes()
    assert a["new_observations"] == 1
    assert append(first, path)["repeat_observations"] == 1
    assert path.read_bytes() == original
    new = get(b, c)
    for g in new["games"]:
        g["source_web_context_sha256"] = "c"*64
    newer = append(new, path)
    assert newer["new_observations"] == 1
    assert path.read_bytes().startswith(original)
    assert len(read_journal(path)) == 2


def test_only_future_games_captured_without_fake_current_lineups():
    b, c = sources()
    r = get(b, c, at=AT+timedelta(days=1))
    assert r["games"] == [] and r["eligible_pregame_games"] == 0
    c["games"] = []
    c["player_evidence_review"]["games"] = []
    r = get(b, c)
    assert not r["games"][0]["web_context_game_observed"]
    assert r["source_context_games_missing"] == 1
    assert r["games"][0]["teams"]["SF"]["starting_qb_verified"] is False


@pytest.mark.parametrize("field", [
    "context_wrong_kickoff", "context_wrong_team", "postkickoff_injury",
    "unlinked_player", "duplicate_player", "future_context", "wrong_spec", "market_bet",
])
def test_invalid_or_leaky_evidence_fails_closed(field):
    b, c = sources()
    if field == "context_wrong_kickoff":
        c["games"][0]["kickoff_utc"] = "2026-10-20T01:15:00+00:00"
    if field == "context_wrong_team":
        c["games"][0]["away_team"] = "NE"
    if field == "postkickoff_injury":
        c["player_evidence_review"]["games"][0]["teams"]["SF"]["players"][0][
            "nflverse_report_modified_utc"
        ] = "2026-10-20T00:16:00+00:00"
    if field == "unlinked_player":
        c["player_evidence_review"]["games"][0]["teams"]["SF"]["players"][0][
            "player_id"
        ] = "00-999"
    if field == "duplicate_player":
        people = c["player_evidence_review"]["games"][0]["teams"]["SF"]["players"]
        people.append(deepcopy(people[0]))
    if field == "future_context":
        c["captured_utc"] = "2026-10-20T00:16:00+00:00"
    if field == "wrong_spec":
        b["spec"] = "UNTRUSTED"
    if field == "market_bet":
        b["qualified_bets"] = 1
    with pytest.raises((ValueError, KeyError)):
        get(b, c)


def test_original_context_observation_timestamp_is_not_a_verified_starter():
    b, c = sources()
    c["player_evidence_review"]["games"][0]["teams"]["SF"]["players"][0][
        "nflverse_reported_status"
    ] = "out"
    r = get(b, c)
    assert r["games"][0]["teams"]["SF"]["starting_qb_verified"] is False
    assert r["games"][0]["confirmed_qb_effect_applied"] is False
    assert r["games"][0]["injury_points_adjustment_applied"] is False
