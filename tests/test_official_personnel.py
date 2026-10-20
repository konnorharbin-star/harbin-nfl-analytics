import copy
import hashlib
import json
from datetime import timedelta

import pytest

from nfl.official_personnel import article_content, personnel_from_articles
from nfl.public_article_dates import publisher_issue_time
from nfl.public_context_inputs import context_contract
from tests.test_public_context_inputs import AT, GAME, fixture

PLAYERS = [
    {
        "game_id": GAME["game_id"],
        "team": team,
        "player_id": pid,
        "player_name": name,
        "source_player_names": [name],
        "position": position,
    }
    for team, pid, name, position in [
        ("SF", "home", "Home Quarterback", "QB"),
        ("WAS", "away", "Away Quarterback", "QB"),
        ("SF", "backup", "Home Backup", "QB"),
        ("WAS", "backup2", "Away Backup", "QB"),
        ("SF", "receiver", "Home Receiver", "WR"),
    ]
]


def article(
    body=None, *, headline="Week 6 inactives: Washington Commanders at San Francisco 49ers"
):
    if body is None:
        body = (
            "<h3>COMMANDERS</h3><ul><li>QB Away Backup</li></ul>"
            "<h3>NINERS</h3><ul><li>QB Home Backup (emergency third QB)</li></ul>"
            "<p>Home Quarterback will start for the 49ers.</p>"
            "<p>Away Quarterback is the starting quarterback for Washington.</p>"
            "<p>Home Receiver is active for tonight.</p>"
        )
    meta = {
        "@type": "NewsArticle",
        "headline": headline,
        "datePublished": (AT - timedelta(minutes=10)).isoformat(),
        "dateModified": (AT - timedelta(minutes=5)).isoformat(),
    }
    html = (
        '<script type="application/ld+json">' + json.dumps(meta) + "</script>"
        '<article><div class="nfl-c-body-part nfl-c-body-part--text">' + body + "</div></article>"
        "<nav><p>Home Backup will start for the 49ers.</p></nav>"
    )
    return {
        "source_url": "https://www.nfl.com/news/week-6-inactives",
        "captured_utc": AT.isoformat(),
        "snapshot_sha256": hashlib.sha256(html.encode()).hexdigest(),
        **publisher_issue_time(html, observed=AT),
        "official_article": article_content(html),
    }


def test_explicit_statements_and_inactive_lists_are_separate_from_inferred_actives():
    result = personnel_from_articles([article()], GAME, PLAYERS, asof=AT)
    assert set(result["starters"]) == {"SF", "WAS"}
    assert set(result["inactive_reports"]) == {"SF", "WAS"}
    assert result["confirmed_player_designations"] == 3
    assert result["not_listed_does_not_prove_active"]
    assert (
        result["inactive_reports"]["SF"]["players"][0]["source_annotation"] == "emergency third QB"
    )
    assert result["model_adjustment"] is None


@pytest.mark.parametrize(
    "body",
    [
        "<p>Home Quarterback is expected to start for the 49ers.</p>",
        "<p>If he is healthy, Home Quarterback will start for the 49ers.</p>",
        "<p>It is not true that Home Quarterback will start for the 49ers.</p>",
        "<p>Home Quarterback will start for the 49ers?</p>",
        "<p>The depth chart lists Home Quarterback first.</p>",
        "<p>Home Quarterback will start in Week 7 for the 49ers.</p>",
        "<h3>NINERS</h3><ul><li>QB Home Backup</li></ul>",
    ],
)
def test_conditional_statements_depth_charts_and_backup_inactives_never_confirm_starter(body):
    result = personnel_from_articles([article(body)], GAME, PLAYERS, asof=AT)
    assert result["starters"] == {}


@pytest.mark.parametrize(
    "change",
    [
        lambda a: a.update(source_url="https://nfl.com.evil.test/news"),
        lambda a: a.update(source_url="https://www.49erswebzone.com/news"),
        lambda a: a.update(source_publication_utc=None),
        lambda a: a.update(source_modified_utc="2026-10-20T01:00:00Z"),
        lambda a: a.update(snapshot_sha256="bad"),
        lambda a: a["official_article"].update(headline="Week 5 inactives: Commanders at 49ers"),
        lambda a: a["official_article"].update(truncated=True),
    ],
)
def test_untrusted_wrong_week_missing_clock_or_incomplete_article_is_not_evidence(change):
    record = article()
    change(record)
    result = personnel_from_articles([record], GAME, PLAYERS, asof=AT)
    assert result["starters"] == {} and result["inactive_reports"] == {}


def test_postkickoff_capture_never_becomes_pregame_evidence_even_with_pregame_publication():
    record = article()
    record["captured_utc"] = "2026-10-20T00:16:00Z"
    result = personnel_from_articles([record], GAME, PLAYERS, asof=record["captured_utc"])
    assert result["starters"] == {} and result["inactive_reports"] == {}
    assert result["rejections"][0]["reason"] == "OFFICIAL_ARTICLE_NOT_CAPTURED_PREGAME"


def test_ambiguous_names_and_conflicting_official_designations_do_not_confirm_player():
    record = article()
    ambiguous = [*PLAYERS, {**PLAYERS[0], "player_id": "different"}]
    result = personnel_from_articles([record], GAME, ambiguous, asof=AT)
    assert "SF" not in result["starters"]
    contradiction = article(
        "<h3>NINERS</h3><ul><li>WR Home Receiver</li></ul><p>Home Receiver is active.</p>"
    )
    result = personnel_from_articles([contradiction], GAME, PLAYERS, asof=AT)
    assert result["confirmed_player_designations"] == 0
    assert result["conflicting_player_designations"] == [("SF", "receiver")]


def test_generic_league_game_day_report_can_bind_each_named_team_without_headline_matchup():
    record = article(headline="Week 6 inactives: Players ruled out for Monday's games")
    result = personnel_from_articles([record], GAME, PLAYERS, asof=AT)
    assert set(result["inactive_reports"]) == {"SF", "WAS"}
    # Generic reports support game-day inactive lists; they cannot establish
    # an unrelated opponent's starting-QB announcement in a different section.
    wrong_day = copy.deepcopy(record)
    wrong_day["source_publication_utc"] = "2026-10-18T23:00:00Z"
    assert personnel_from_articles([wrong_day], GAME, PLAYERS, asof=AT)["inactive_reports"] == {}


def test_club_source_cannot_make_statements_for_the_opposing_team():
    record = article()
    record["source_url"] = "https://www.49ers.com/news/week-6-inactives"
    result = personnel_from_articles([record], GAME, PLAYERS, asof=AT)
    assert set(result["starters"]) == {"SF"}


def test_contract_rechecks_official_sources_and_removes_only_resolved_blockers():
    observation, _, _ = fixture()
    context = {"official_articles": [article()]}
    result = context_contract(observation, context, GAME, asof=AT, players=PLAYERS)
    assert result["confirmed_starting_qb_teams"] == ["SF", "WAS"]
    assert result["official_inactive_teams"] == ["SF", "WAS"]
    assert "confirmed_starting_qbs" not in result["missing_required_inputs"]
    assert "final_game_specific_player_availability" in result["missing_required_inputs"]
    assert not result["ready_for_context_adjusted_prediction"]


def test_jsonld_issue_and_modification_times_are_checked_and_ambiguous_articles_rejected():
    record = article()
    assert record["source_publication_utc"] == (AT - timedelta(minutes=10)).isoformat()
    meta = {
        "@type": "NewsArticle",
        "datePublished": "2026-10-19T22:00:00Z",
        "dateModified": "2026-10-20T01:00:00Z",
    }
    html = '<script type="application/ld+json">' + json.dumps(meta) + "</script>"
    assert publisher_issue_time(html, observed=AT)["source_publication_utc"] is None
    meta["dateModified"] = "2026-10-19T22:10:00Z"
    html = '<script type="application/ld+json">' + json.dumps([meta, meta]) + "</script>"
    assert (
        publisher_issue_time(html, observed=AT)["publication_metadata_status"]
        == "AMBIGUOUS_ARTICLE_METADATA"
    )


def test_default_intake_connects_official_search_article_dates_roof_and_contract():
    from nfl.public_roof_inputs import schedule_snapshot
    from scripts.nfl_web_model_intake import collect
    from tests.test_public_context_inputs import COMPETITION
    from tests.test_public_roof_inputs import ROW

    record = article()
    metadata = {
        "@type": "NewsArticle",
        "headline": record["official_article"]["headline"],
        "datePublished": record["source_publication_utc"],
        "dateModified": record["source_modified_utc"],
    }
    body = "".join(
        f"<{s['tag']}>{s['text']}</{s['tag']}>" for s in record["official_article"]["segments"]
    )
    html = (
        '<script type="application/ld+json">' + json.dumps(metadata) + "</script>"
        '<div class="nfl-c-body-part--text">' + body + "</div>"
    )
    observation, sources, _ = fixture()

    def fetch(url, **kwargs):
        if kwargs.get("search"):
            return (
                "<rss><channel><item><title>Official inactives</title><link>"
                + record["source_url"]
                + "</link></item></channel></rss>"
            ).encode()
        if url == record["source_url"]:
            assert kwargs.get("publisher_bound") is True
            return html.encode()
        if "/scoreboard?" in url:
            event = {
                "id": observation["provider_event_id"],
                "date": GAME["kickoff_utc"],
                "status": COMPETITION["status"],
                "competitions": [COMPETITION],
            }
            return json.dumps({"events": [event]}).encode()
        if url in sources:
            return json.dumps(sources[url]).encode()
        return b"<html>No data</html>"

    game = {**GAME, "status": "INDEPENDENT_CONDITIONAL_FORECAST"}
    result = collect(
        {"games": [game], "player_props": PLAYERS},
        fetch=fetch,
        now=lambda: AT,
        schedules=schedule_snapshot([ROW], captured=AT),
    )
    gate = result["games"][0]["input_data_contract"]
    assert result["starting_qbs_confirmed"] == 2
    assert result["official_inactive_team_reports_verified"] == 2
    assert "game_day_roof_position" not in gate["missing_required_inputs"]
    assert result["outdoor_weather_forecasts_verified"] == 1
    json.dumps(result)  # No native date objects escape into the published report.


def test_official_article_redirect_cannot_change_publisher(monkeypatch):
    from nfl.web_model_intake import fetch_page

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def geturl(self):
            return "https://www.espn.com/other-article"

        def read(self, limit):
            raise AssertionError("A different publisher must not become official team evidence")

    monkeypatch.setattr("nfl.web_model_intake.urlopen", lambda *a, **k: Response())
    with pytest.raises(ValueError, match="CROSS_PUBLISHER"):
        fetch_page("https://www.49ers.com/news/week-6-inactives", publisher_bound=True)
