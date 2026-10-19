"""Publisher issued-at chronology: research lead dates are not starter proof."""
from datetime import UTC, datetime, timedelta

import pytest

from nfl.public_article_dates import publisher_issue_time, bind_publication_to_game
from scripts.nfl_web_model_intake import collect

AT = datetime(2026, 10, 14, 12, tzinfo=UTC)
GAME = {
    "game_id": "2026_06_SEA_DEN", "home_team": "DEN", "away_team": "SEA",
    "kickoff_utc": "2026-10-16T00:15:00+00:00",
    "status": "INDEPENDENT_CONDITIONAL_FORECAST",
}


def html_date(value, *, key="article:published_time"):
    return f'<head><meta property="{key}" content="{value}"></head><article>Pregame context</article>'


def test_dated_meta_accepted_but_not_a_verified_lineup():
    date = "2026-10-13T16:41:00Z"
    issued = publisher_issue_time(html_date(date), observed=AT)
    assert issued["source_publication_utc"] == "2026-10-13T16:41:00+00:00"
    assert issued["publication_metadata_status"] == "DATED_PUBLISHER_META_NOT_IMMUTABLE_ORIGINAL"
    link = bind_publication_to_game(
        {**issued, "captured_utc": AT.isoformat()}, kickoff_utc=GAME["kickoff_utc"]
    )
    assert link["publication_evidence_status"] == "PUBLISHER_META_PREKICKOFF_NOT_FACT_VERIFICATION"


@pytest.mark.parametrize("markup,expected", [
    ("<p>No source date</p>", "NO_DATED_PUBLISHER_META"),
    (html_date("2026-10-14"), "NO_DATED_PUBLISHER_META"),
    (html_date("2026-10-14T12:30:00Z"), "FUTURE_PUBLISHER_META"),
    (html_date("not-a-date"), "NO_DATED_PUBLISHER_META"),
    (html_date("2026-10-13T10:00:00Z") +
     html_date("2026-10-13T14:00:00Z", key="datePublished"),
     "CONFLICTING_PUBLISHER_META"),
])
def test_unknown_conflicting_or_future_metadata_never_gets_an_issue_time(markup, expected):
    result = publisher_issue_time(markup, observed=AT)
    assert result["source_publication_utc"] is None
    assert result["publication_metadata_status"] == expected


def test_issued_after_kickoff_cannot_be_labeled_pregame_even_if_read_later():
    issued = publisher_issue_time(
        html_date("2026-10-16T00:16:00Z"), observed=AT + timedelta(days=3)
    )
    link = bind_publication_to_game(
        {**issued, "captured_utc": (AT + timedelta(days=3)).isoformat()},
        kickoff_utc=GAME["kickoff_utc"],
    )
    assert link["source_publication_utc"] is None
    assert link["publication_evidence_status"] == "PAGE_CAPTURE_AFTER_KICKOFF"


def test_article_metadata_is_bound_to_every_factor_without_asserting_verified_values():
    url = "https://www.espn.com/nfl/story/pregame"
    def fetch(url_requested, **kwargs):
        if kwargs.get("search"):
            return (
                "<rss><channel><item><title>Injuries</title><link>"
                + url + "</link></item></channel></rss>"
            ).encode()
        if url_requested == url:
            return html_date("2026-10-13T16:41:00Z").encode()
        return b"<div>No dated published-at metadata</div>"
    result = collect({"games": [GAME], "player_props": []}, fetch=fetch, now=lambda: AT)
    factors = result["games"][0]["situational_factors"]
    assert len(factors) >= 14
    assert all(
        f["evidence"][0]["source_publication_utc"] == "2026-10-13T16:41:00+00:00"
        for f in factors.values()
    )
    assert all(
        f["evidence"][0]["publication_evidence_status"]
        == "PUBLISHER_META_PREKICKOFF_NOT_FACT_VERIFICATION"
        for f in factors.values()
    )
    assert result["verified_situational_values"] == 0
    assert all(f["verified_value"] is None and f["model_adjustment"] is None
               for f in factors.values())


def test_article_metadata_missing_is_fully_disclosed_in_factor_evidence():
    url = "https://www.espn.com/nfl/story"
    def fetch(link, **kwargs):
        if kwargs.get("search"):
            return f"<rss><channel><item><title>QB</title><link>{url}</link></item></channel></rss>".encode()
        return b"<html><p>Unverified news and no published timestamp</p></html>"
    result = collect({"games": [GAME], "player_props": []}, fetch=fetch, now=lambda: AT)
    ev = result["games"][0]["situational_factors"]["quarterback"]["evidence"][0]
    assert ev["source_publication_utc"] is None
    assert ev["publication_evidence_status"] == "NO_DATED_PUBLISHER_META"
    assert result["games"][0]["situational_facts_verified"] is False
