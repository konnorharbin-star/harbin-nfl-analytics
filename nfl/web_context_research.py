"""Free web-context evidence collector for independent NFL research.

Sources are NWS hourly forecast, ESPN's public injury snapshot and Google News
RSS headline search. These are *observations / leads*, not verified starters,
injury overrides, historical weather or calibrated point adjustments.

No sportsbook inputs and never change the canonical model's predictions.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from hashlib import sha256
import json
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from .weather import TEAM_HOME, is_indoor_roof
from .coupled_nflverse_asof import kickoff_utc

VERSION = "NFL_PUBLIC_WEB_CONTEXT_RESEARCH_V1"
INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"
NEWS_URL = "https://news.google.com/rss/search"
NWS_POINTS = "https://api.weather.gov/points/"
# Source domains identify established publishers; RSS alone is not fact verification.
PREFERRED_NEWS_DOMAINS = (
    "nfl.com", "espn.com", "cbssports.com", "apnews.com",
    "reuters.com", "nbcsports.com", "foxsports.com",
)

def preferred_publisher_domain(source_url):
    """Require an HTTPS publisher hostname, never trust a spoofed source label."""
    if not isinstance(source_url, str):
        return None
    try:
        parsed = urlparse(source_url)
        hostname = (parsed.hostname or "").lower().rstrip(".")
    except ValueError:
        return None
    if parsed.scheme != "https" or parsed.username or parsed.password:
        return None
    for domain in PREFERRED_NEWS_DOMAINS:
        if hostname == domain or hostname.endswith("." + domain):
            return domain
    return None
TEAM_NAMES = {
    "ARI":"Arizona Cardinals","ATL":"Atlanta Falcons","BAL":"Baltimore Ravens",
    "BUF":"Buffalo Bills","CAR":"Carolina Panthers","CHI":"Chicago Bears",
    "CIN":"Cincinnati Bengals","CLE":"Cleveland Browns","DAL":"Dallas Cowboys",
    "DEN":"Denver Broncos","DET":"Detroit Lions","GB":"Green Bay Packers",
    "HOU":"Houston Texans","IND":"Indianapolis Colts","JAX":"Jacksonville Jaguars",
    "KC":"Kansas City Chiefs","LA":"Los Angeles Rams","LAC":"Los Angeles Chargers",
    "LV":"Las Vegas Raiders","MIA":"Miami Dolphins","MIN":"Minnesota Vikings",
    "NE":"New England Patriots","NO":"New Orleans Saints",
    "NYG":"New York Giants","NYJ":"New York Jets","PHI":"Philadelphia Eagles",
    "PIT":"Pittsburgh Steelers","SEA":"Seattle Seahawks",
    "SF":"San Francisco 49ers","TB":"Tampa Bay Buccaneers",
    "TEN":"Tennessee Titans","WAS":"Washington Commanders",
}

def _utc(s):
    if not isinstance(s, str):
        raise ValueError("MISSING_TIMESTAMP")
    try:
        d=datetime.fromisoformat(s.replace("Z","+00:00"))
    except ValueError as e:
        raise ValueError("MALFORMED_TIMESTAMP") from e
    if d.tzinfo is None:
        raise ValueError("NAIVE_TIMESTAMP")
    return d.astimezone(UTC)

def _request(url, *, accept, allow_hosts):
    parsed=urlparse(url)
    if parsed.scheme!="https" or parsed.hostname not in allow_hosts:
        raise ValueError("UNAPPROVED_SOURCE_HOST")
    headers={"User-Agent":"HarbinFootballResearch/1.0 (public pregame context; read-only)",
             "Accept":accept}
    request=Request(url,headers=headers)
    # No user-supplied URL is passed through this client.
    try:
        with urlopen(request, timeout=12) as response:
            if response.geturl().startswith("https://") is False:
                raise ValueError("UNSAFE_REDIRECT")
            size=1_500_000
            raw=response.read(size+1)
            if len(raw)>size:
                raise ValueError("SOURCE_RESPONSE_TOO_LARGE")
            return raw
    except (HTTPError, URLError, TimeoutError, OSError) as e:
        raise ValueError("SOURCE_HTTP_ERROR: "+type(e).__name__) from e

def get_json(url):
    raw=_request(url,accept="application/geo+json, application/json",allow_hosts={
        "site.api.espn.com","api.weather.gov"})
    try:
        return json.loads(raw)
    except (ValueError,UnicodeDecodeError) as e:
        raise ValueError("SOURCE_JSON_INVALID") from e

def get_xml(url):
    raw=_request(url,accept="application/rss+xml, application/xml, text/xml",allow_hosts={"news.google.com"})
    try:
        return ElementTree.fromstring(raw)
    except ElementTree.ParseError as e:
        raise ValueError("SOURCE_RSS_INVALID") from e

def _status(athlete):
    obj=athlete.get("status") or athlete.get("type") or {}
    if isinstance(obj,dict):
        return str(obj.get("name") or obj.get("description") or obj.get("abbreviation") or "")
    return str(obj)

def parse_espn_injuries(payload, team):
    """Only return data explicitly attributable to a requested team ID."""
    records=[]
    nodes=payload.get("injuries") or []
    if not isinstance(nodes,list):
        raise ValueError("INJURY_FEED_MALFORMED")
    for node in nodes:
        if not isinstance(node,dict):
            continue
        t=node.get("team") or {}
        abbr=str(t.get("abbreviation") or "").upper()
        if abbr!=team:
            continue
        items=node.get("injuries") or []
        if not isinstance(items,list):
            continue
        for item in items:
            if not isinstance(item,dict):
                continue
            person=item.get("athlete") or {}
            name=person.get("fullName") or person.get("displayName")
            if not name:
                continue
            position=person.get("position") or {}
            status=_status(item)
            details=item.get("details") or {}
            records.append({
                "player":str(name),
                "position":str(position.get("abbreviation") or "") if isinstance(position,dict) else "",
                "reported_status":status,
                "details":str(details.get("detail") or "")[:160] if isinstance(details,dict) else "",
                "source":"ESPN_PUBLIC_INJURY_FEED",
                "status_is_official_week_specific_final_game_status":False,
            })
    return records[:55]

def news_search(team, *, at, fetch_xml=get_xml):
    name=TEAM_NAMES.get(team)
    if name is None: raise ValueError("UNKNOWN_TEAM")
    query=f'\"{name}\" (injury OR quarterback OR inactive OR starter) when:7d'
    url=NEWS_URL+"?"+urlencode({"q":query,"hl":"en-US","gl":"US","ceid":"US:en"})
    root=fetch_xml(url)
    items=[]
    for node in root.findall(".//channel/item")[:25]:
        title=str(node.findtext("title") or "").strip()[:220]
        link=str(node.findtext("link") or "").strip()
        published=str(node.findtext("pubDate") or "").strip()
        source=node.find("source")
        publisher=source.text.strip() if source is not None and source.text else None
        source_url=source.attrib.get("url") if source is not None else None
        preferred_domain=preferred_publisher_domain(source_url)
        try:
            issued=parsedate_to_datetime(published).astimezone(UTC)
        except (ValueError,TypeError,OverflowError):
            continue
        if issued>at or issued<at-timedelta(days=7):
            continue
        if not title or not link.startswith("https://"):
            continue
        items.append({
            "headline":title,"original_publication_time_utc":issued.isoformat(),
            "rss_redirect_link":link,
            "publisher_named_in_feed":publisher,
            "publisher_source_url_in_feed":source_url,
            "preferred_publisher_domain":preferred_domain,
            "established_publisher_rss_attribution":preferred_domain is not None,
            "publisher_attribution_verified_by_article_fetch":False,
            "source_type":"SEARCH_RESULT_LEAD_UNVERIFIED",
            "confirmed_injury_or_qb_status":False,
            "article_body_independently_read":False,
        })
    # Trusted reporting is prioritized, but an RSS headline remains a lead.
    items.sort(key=lambda item: (
        not item["established_publisher_rss_attribution"],
        -datetime.fromisoformat(item["original_publication_time_utc"]).timestamp(),
    ))
    selected=items[:6]
    return {"query":query,"search_url":url,"leads":selected,
            "preferred_reported_leads":sum(
                item["established_publisher_rss_attribution"] for item in selected
            ),
            "other_unverified_leads":sum(
                not item["established_publisher_rss_attribution"] for item in selected
            ),
            "source_priority":"ESPN_CBS_NFL_AP_AND_OTHER_ESTABLISHED_OUTLETS",
            "status":"HEADLINE_LEADS_ONLY_NOT_STARTER_VERIFICATION"}

def _nws_forecast(game, *, at, fetch_json=get_json):
    roof=game.get("roof")
    if is_indoor_roof(roof):
        return {"status":"INDOOR_ROOF_REPORTED_NO_OUTDOOR_WEATHER_ASSUMPTION",
                "outdoor_forecast_applicable":False,"model_points_adjustment":None}
    if game.get("neutral_site") is True:
        return {"status":"NEUTRAL_VENUE_COORDINATES_NOT_VERIFIED",
                "outdoor_forecast_applicable":None,"model_points_adjustment":None}
    home=game.get("home_team")
    location=TEAM_HOME.get(home)
    if not location:
        return {"status":"UNKNOWN_VENUE_COORDINATES","model_points_adjustment":None}
    latitude,longitude,_=location
    if not (24<=latitude<=50 and -125<=longitude<=-66):
        return {"status":"OUTSIDE_NWS_COVERAGE","model_points_adjustment":None}
    point_url=NWS_POINTS+f"{latitude:.4f},{longitude:.4f}"
    pt=fetch_json(point_url)
    hourly=(pt.get("properties") or {}).get("forecastHourly")
    if not isinstance(hourly,str) or not hourly.startswith("https://api.weather.gov/"):
        return {"status":"NWS_POINT_NO_HOURLY_GRID","source":point_url,
                "model_points_adjustment":None}
    forecast=fetch_json(hourly)
    props=forecast.get("properties") or {}
    update=props.get("updated") or props.get("generatedAt")
    if not update:
        return {"status":"NWS_NO_SOURCE_ISSUE_TIMESTAMP",
                "source":hourly,"model_points_adjustment":None}
    issued=_utc(update)
    if issued>at:
        return {"status":"NWS_FORECAST_ISSUED_AFTER_CAPTURE",
                "source":hourly,"model_points_adjustment":None}
    kickoff=kickoff_utc(game)
    periods=props.get("periods") or []
    for row in periods:
        try:
            start,end=_utc(row.get("startTime")),_utc(row.get("endTime"))
        except ValueError:
            continue
        if start<=kickoff<end:
            return {
                "status":"ASOF_HOURLY_FORECAST_CAPTURED",
                "outdoor_forecast_applicable":True,
                "forecast_source":hourly,"coordinate_source":"TEAM_HOME_APPROXIMATE_STADIUM",
                "issued_utc":issued.isoformat(),
                "period_start_utc":start.isoformat(),"period_end_utc":end.isoformat(),
                "temperature_f":row.get("temperature"),
                "wind_speed_text":row.get("windSpeed"),
                "wind_direction":row.get("windDirection"),
                "precipitation_probability_percent":(
                    (row.get("probabilityOfPrecipitation") or {}).get("value")
                ),
                "short_forecast":row.get("shortForecast"),
                "model_points_adjustment":None,
                "effect_coefficient_independently_validated":False,
            }
    return {"status":"KICKOFF_NOT_COVERED_BY_ISSUED_NWS_HOURLY_FORECAST",
            "source":hourly,"issued_utc":issued.isoformat(),
            "model_points_adjustment":None}

def collect_web_context(
    games, *, at, fetch_json=get_json, fetch_xml=get_xml
):
    if at.tzinfo is None:raise ValueError("ASOF_TIMEZONE_REQUIRED")
    at=at.astimezone(UTC)
    if not isinstance(games,list):
        raise ValueError("GAMES_MUST_BE_LIST")
    todo=[];seen=set()
    for game in games:
        gid=str(game.get("game_id") or "")
        if not gid or gid in seen:raise ValueError("INVALID_OR_DUPLICATE_GAME")
        seen.add(gid)
        kickoff=kickoff_utc(game)
        if at>=kickoff:
            continue
        todo.append(game)
    errors=[]
    injuries={}
    try:
        payload=fetch_json(INJURIES_URL)
        for team in {g[side] for g in todo for side in ("home_team","away_team")}:
            injuries[team]=parse_espn_injuries(payload,team)
    except (ValueError,KeyError,TypeError) as e:
        errors.append({"source":"ESPN_INJURIES","error":str(e)})
    team_news={}
    for team in sorted({g[side] for g in todo for side in ("home_team","away_team")}):
        try:
            team_news[team]=news_search(team,at=at,fetch_xml=fetch_xml)
        except (ValueError,KeyError,TypeError) as e:
            errors.append({"source":"NEWS_RSS","team":team,"error":str(e)})
            team_news[team]={"status":"NEWS_SOURCE_UNAVAILABLE","leads":[]}
    out=[]
    for game in todo:
        gid=str(game["game_id"])
        try:
            weather=_nws_forecast(game,at=at,fetch_json=fetch_json)
        except (ValueError,KeyError,TypeError) as e:
            weather={"status":"WEATHER_SOURCE_UNAVAILABLE","reason":str(e),
                     "model_points_adjustment":None}
            errors.append({"source":"NWS","game_id":gid,"error":str(e)})
        home,away=game["home_team"],game["away_team"]
        out.append({
            "game_id":gid,"home_team":home,"away_team":away,
            "kickoff_utc":kickoff_utc(game).isoformat(),
            "roof_as_reported":game.get("roof"),
            "venue_as_reported":game.get("stadium"),
            "weather":weather,
            "structured_injury_snapshot":{
                home:injuries.get(home),away:injuries.get(away),
                "captured_utc":at.isoformat(),
                "source_url":INJURIES_URL,
                "feed_records_not_confirmed_starters_or_final_inactives":True,
            },
            "news_search_leads":{home:team_news.get(home),away:team_news.get(away)},
            "starting_qb_verified":False,
            "player_activity_verified":False,
            "quantitative_forecast_adjustment":None,
            "betting_authorized":False,
            "source_used_to_refit_original_predictions":False,
        })
    return {
        "version":VERSION,"status":"RESEARCH_ONLY_WEB_CONTEXT_NOT_SCORED",
        "captured_utc":at.isoformat(),
        "source_descriptions":{
            "injuries":"ESPN PUBLIC NFL injury snapshot, not independently official week-specific final actives",
            "news":"Google News RSS search result headlines: leads, not article verification",
            "weather":"National Weather Service generated hourly forecast, available at capture"},
        "games":out,"source_errors":errors,"score_projections_changed":False,
        "wager_authorized":False,"estimated_ev":None,
    }
