"""Synthetic external-feed tests for independent NFL web research evidence."""
from datetime import UTC, datetime, timedelta
from xml.etree import ElementTree
import pytest

from nfl.web_context_research import (
    collect_web_context, news_search, parse_espn_injuries, _nws_forecast, _request,
)

NOW=datetime(2026,10,13,11,30,tzinfo=UTC)
GAME={"game_id":"2026_06_BUF_PHI","season":2026,"week":6,
      "home_team":"PHI","away_team":"BUF",
      "gameday":"2026-10-18","gametime":"13:00",
      "roof":"outdoors","stadium":"Lincoln Financial Field"}

def football_feeds(url):
    if "/injuries" in url:
        return {"injuries":[
            {"team":{"abbreviation":"PHI"},"injuries":[
                {"athlete":{"fullName":"Example Philadelphia WR",
                   "position":{"abbreviation":"WR"}},"status":"Questionable"}]},
            {"team":{"abbreviation":"BUF"},"injuries":[
                {"athlete":{"fullName":"Example Buffalo QB",
                   "position":{"abbreviation":"QB"}},"status":"Limited"}]},
        ]}
    if "api.weather.gov/points" in url:
        return {"properties":{"forecastHourly":"https://api.weather.gov/gridpoints/PHI/1,2/forecast/hourly"}}
    if "gridpoints" in url:
        return {"properties":{"updated":"2026-10-13T11:01:00+00:00",
           "periods":[{"startTime":"2026-10-18T16:00:00+00:00",
             "endTime":"2026-10-18T18:00:00+00:00",
             "temperature":58,"windSpeed":"14 mph","windDirection":"W",
             "probabilityOfPrecipitation":{"value":35},"shortForecast":"Rain possible"}]}}
    raise ValueError("UNEXPECTED_TEST_URL")

def feed(url):
    return ElementTree.fromstring("""<rss><channel>
      <item><title>Philadelphia Eagles injury QB news</title>
      <link>https://news.google.com/rss/articles/example</link>
      <pubDate>Tue, 13 Oct 2026 10:00:00 GMT</pubDate>
      <source>Test Sports News</source></item>
      <item><title>Future injury article must be excluded</title>
      <link>https://news.google.com/rss/articles/future</link>
      <pubDate>Mon, 19 Oct 2026 10:00:00 GMT</pubDate></item>
    </channel></rss>""")

def test_context_collects_only_asof_leads_and_nws_without_adjusting_models():
    r=collect_web_context([GAME],at=NOW,fetch_json=football_feeds,fetch_xml=feed)
    assert r["status"]=="RESEARCH_ONLY_WEB_CONTEXT_NOT_SCORED"
    assert not r["source_errors"]
    g=r["games"][0]
    assert g["weather"]["status"]=="ASOF_HOURLY_FORECAST_CAPTURED"
    assert g["weather"]["wind_speed_text"]=="14 mph"
    assert g["weather"]["precipitation_probability_percent"]==35
    assert g["weather"]["effect_coefficient_independently_validated"] is False
    assert g["weather"]["model_points_adjustment"] is None
    assert g["structured_injury_snapshot"]["BUF"][0]["player"]=="Example Buffalo QB"
    assert g["structured_injury_snapshot"]["PHI"][0]["reported_status"]=="Questionable"
    assert g["starting_qb_verified"] is False and g["player_activity_verified"] is False
    assert len(g["news_search_leads"]["PHI"]["leads"])==1
    assert g["news_search_leads"]["PHI"]["leads"][0]["confirmed_injury_or_qb_status"] is False
    assert r["score_projections_changed"] is False
    assert r["wager_authorized"] is False and r["estimated_ev"] is None

def test_indoor_and_neutral_venue_do_not_infer_nws_stadium_weather():
    x=dict(GAME,roof="closed roof")
    wx=_nws_forecast(x,at=NOW,fetch_json=lambda url:1/0)
    assert wx["status"]=="INDOOR_ROOF_REPORTED_NO_OUTDOOR_WEATHER_ASSUMPTION"
    assert wx["model_points_adjustment"] is None
    x=dict(GAME,neutral_site=True)
    assert _nws_forecast(x,at=NOW,fetch_json=lambda url:1/0)["status"]=="NEUTRAL_VENUE_COORDINATES_NOT_VERIFIED"

def test_forecast_from_after_asof_is_rejected():
    def j(url):
        data=football_feeds(url)
        if "gridpoints" in url:
            data["properties"]["updated"]="2026-10-14T12:00:00+00:00"
        return data
    assert _nws_forecast(GAME,at=NOW,fetch_json=j)["status"]=="NWS_FORECAST_ISSUED_AFTER_CAPTURE"

def test_fail_closed_with_missing_sources_and_no_healthy_roster_fiction():
    def bad_json(url):
        raise ValueError("HTTP_SOURCE_FAILED")
    def bad_xml(url):
        raise ValueError("RSS_SOURCE_FAILED")
    r=collect_web_context([GAME],at=NOW,fetch_json=bad_json,fetch_xml=bad_xml)
    assert len(r["source_errors"])==4
    g=r["games"][0]
    assert g["weather"]["status"]=="WEATHER_SOURCE_UNAVAILABLE"
    assert g["structured_injury_snapshot"]["BUF"] is None
    assert g["news_search_leads"]["PHI"]["status"]=="NEWS_SOURCE_UNAVAILABLE"
    assert g["starting_qb_verified"] is False
    assert r["wager_authorized"] is False

def test_original_pregame_boundaries_and_duplicate_games():
    with pytest.raises(ValueError,match="DUPLICATE"):
        collect_web_context([GAME,GAME],at=NOW,fetch_json=football_feeds,fetch_xml=feed)
    assert not collect_web_context([GAME],at=datetime(2026,10,19,tzinfo=UTC),
                                   fetch_json=football_feeds,fetch_xml=feed)["games"]
    with pytest.raises(ValueError,match="ASOF_TIMEZONE"):
        collect_web_context([GAME],at=NOW.replace(tzinfo=None))

def test_injuries_belong_to_correct_teams_and_unknown_is_unknown():
    response=football_feeds("https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries")
    assert len(parse_espn_injuries(response,"PHI"))==1
    assert parse_espn_injuries(response,"CHI")==[]
    with pytest.raises(ValueError,match="MALFORMED"):
        parse_espn_injuries({"injuries":"unrecognized"},"BUF")
    assert news_search("BUF",at=NOW,fetch_xml=feed)["status"].startswith("HEADLINE_LEADS")

def test_get_request_blocks_unknown_source_hosts_before_any_io():
    with pytest.raises(ValueError,match="UNAPPROVED_SOURCE_HOST"):
        _request("https://example.com/anything",accept="application/json",
                 allow_hosts={"api.weather.gov"})
