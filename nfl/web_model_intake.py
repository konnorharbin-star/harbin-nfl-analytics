"""No-key public search/page intake. Retrieved text is evidence, never model instructions."""

from __future__ import annotations

import hashlib
import math
import re
from datetime import UTC, datetime
from html.parser import HTMLParser
from urllib.parse import urlencode, urljoin, urlparse
from urllib.request import Request, urlopen
from xml.etree import ElementTree
from zoneinfo import ZoneInfo

from .web_context_research import TEAM_NAMES

BOARD = "https://www.covers.com/sport/football/nfl/player-props"
SOURCE_DOMAINS = (
    "covers.com",
    "espn.com",
    "cbssports.com",
    "nfl.com",
    "weather.gov",
    "apnews.com",
    "reuters.com",
    "foxsports.com",
    "nbcsports.com",
)
SITUATIONAL_TOPICS = {
    "football": "roster depth chart injuries inactive quarterback team statistics",
    "quarterback": "starting quarterback availability backup injury practice status",
    "conditions": "game time stadium weather forecast wind gusts precipitation temperature",
    "venue": "stadium roof dome playing surface altitude home field venue change",
    "rest": "rest days short week bye week Thursday Monday schedule fatigue",
    "travel": "travel distance time zones international game road trip arrival",
    "kickoff_body_clock": "kickoff local time early start time zone body clock",
    "schedule_spot": "consecutive road games look ahead next opponent schedule sandwich spot",
    "matchup": "offensive defensive line injuries pressure protection coverage run pass matchup",
    "coaching": "head coach coordinator scheme changes play calling pace personnel tendencies",
    "player_workload": "player snaps routes targets carries red zone usage workload role changes",
    "recent_form": "recent form opponent strength efficiency turnovers fumble recovery variance",
    "motivation": "division rivalry playoff standings motivation coach player comments",
    "officiating": "assigned referee officiating crew penalties tendencies",
    "market_sentiment": "opening current line movement public betting tickets money percentages",
}
TOPICS = {
    **SITUATIONAL_TOPICS,
    "markets": "spread total player props odds receiving yards",
    "history": "historical player prop lines odds results",
}
SECTIONS = {
    "receiving-yards": "receiving_yards",
    "receptions-made": "receptions",
    "passing-yards": "passing_yards",
    "rushing-yards": "rushing_yards",
}


def approved_url(url, *, search=False):
    try:
        p = urlparse(url)
        if p.scheme != "https" or p.username or p.password or p.port not in (None, 443):
            return False
    except (ValueError, TypeError):
        return False
    host = (p.hostname or "").lower()
    return (
        host == "www.bing.com"
        if search
        else any(host == d or host.endswith("." + d) for d in SOURCE_DOMAINS)
    )


def fetch_page(url, *, search=False, exact_source=False):
    if not approved_url(url, search=search):
        raise ValueError("UNAPPROVED_PUBLIC_SOURCE")
    request = Request(
        url, headers={"User-Agent": "HarbinFootballResearch/1.0 (public pages, read-only)"}
    )
    with urlopen(request, timeout=6) as response:
        if not approved_url(response.geturl(), search=search):
            raise ValueError("UNAPPROVED_REDIRECT")
        if exact_source and response.geturl() != url:
            raise ValueError("UNEXPECTED_STRUCTURED_SOURCE_REDIRECT")
        raw = response.read(5_000_001)
    if len(raw) > 5_000_000:
        raise ValueError("SOURCE_TOO_LARGE")
    return raw


class Node:
    def __init__(self, tag="root", attrs=()):
        self.tag, self.attrs, self.children = tag, dict(attrs), []

    def text(self):
        if self.tag in ("script", "style", "noscript", "svg"):
            return ""
        return " ".join(c.text() if isinstance(c, Node) else c for c in self.children)

    def find(self, *, tag=None, cls=None, ident=None):
        result = []
        if (
            (tag is None or self.tag == tag)
            and (cls is None or cls in self.attrs.get("class", "").split())
            and (ident is None or self.attrs.get("id") == ident)
        ):
            result.append(self)
        for child in self.children:
            if isinstance(child, Node):
                result.extend(child.find(tag=tag, cls=cls, ident=ident))
        return result


class Page(HTMLParser):
    def __init__(self, html):
        super().__init__(convert_charrefs=True)
        self.root = Node()
        self.stack = [self.root]
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        node = Node(tag, attrs)
        self.stack[-1].children.append(node)
        if tag not in ("img", "meta", "input", "link", "br", "hr", "source", "area", "wbr"):
            self.stack.append(node)

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == tag:
                del self.stack[i:]
                break

    def handle_data(self, data):
        self.stack[-1].children.append(data)


def clean(value):
    return re.sub(r"\s+", " ", value).strip()


def name_key(value):
    return re.sub(r"[^a-z0-9]", "", value.lower())


def player_links(html):
    page = Page(html).root
    return sorted(
        {
            urljoin(BOARD, n.attrs["href"])
            for n in page.find(tag="a")
            if re.fullmatch(r"/sport/football/nfl/players/\d+/[a-z0-9-]+", n.attrs.get("href", ""))
        }
    )


def parse_props(html, players, games, *, url, observed):
    """Exact displayed matchup, kickoff and unique model-player name/team binding."""
    if (
        not approved_url(url)
        or urlparse(url).hostname != "www.covers.com"
        or observed.tzinfo is None
    ):
        raise ValueError("INVALID_COVERS_SOURCE_OR_OBSERVATION_CLOCK")
    page = Page(html).root
    heads, matchups = page.find(tag="h1"), page.find(cls="covers-CoversNextMatchup")
    if len(heads) != 1 or len(matchups) != 1:
        raise ValueError("NO_UNIQUE_PLAYER_MATCHUP")
    heading = clean(heads[0].text()).split(" Projections", 1)[0]
    matchup = matchups[0]
    time_nodes, date_nodes = matchup.find(cls="nextGame-time"), matchup.find(cls="nextGame-date")
    if len(time_nodes) != 1 or len(date_nodes) != 1:
        raise ValueError("MISSING_GAME_TIME")
    kickoff = (
        datetime.strptime(
            clean(date_nodes[0].text()) + " " + clean(time_nodes[0].text()), "%b %d, %Y %H:%M ET"
        )
        .replace(tzinfo=ZoneInfo("America/New_York"))
        .astimezone(UTC)
    )
    team_nodes = matchup.find(cls="nextGame-team")
    source_teams = []
    for node in team_nodes:
        logos = node.find(tag="img")
        found = (
            re.search(r"/nfl/([a-z]+)\.svg", logos[0].attrs.get("src", ""))
            if len(logos) == 1
            else None
        )
        if found is None:
            raise ValueError("UNVERIFIED_MATCHUP_TEAMS")
        source_teams.append(found[1].upper())
    aliases = {"JAC": "JAX", "LAR": "LA"}
    source_teams = [aliases.get(t, t) for t in source_teams]
    matches = [
        g
        for g in games
        if [g["away_team"], g["home_team"]] == source_teams
        and datetime.fromisoformat(g["kickoff_utc"]) == kickoff
    ]
    if len(matches) != 1 or observed >= kickoff:
        raise ValueError("NO_UNIQUE_PREGAME_MODEL_MATCHUP")
    game = matches[0]
    matched = [
        p
        for p in players
        if p["game_id"] == game["game_id"]
        and not p["is_unallocated_bucket"]
        and name_key(p["player_name"]) == name_key(heading)
    ]
    if len(matched) != 1:
        raise ValueError("NO_UNIQUE_MODEL_PLAYER_IDENTITY")
    player = matched[0]
    contracts, excluded = [], 0
    for section, metric in SECTIONS.items():
        areas = page.find(ident=section)
        if len(areas) != 1 or metric not in player["projections"]:
            continue
        for row in areas[0].find(cls="other-odds-row"):
            labels = row.find(cls="other-odds-label")
            logos = labels[0].find(tag="img") if len(labels) == 1 else []
            book = logos[0].attrs.get("alt") if len(logos) == 1 else None
            sides = []
            for cls, side in (("other-over-odds", "over"), ("other-under-odds", "under")):
                nodes = row.find(cls=cls)
                text = clean(nodes[0].text()) if len(nodes) == 1 else ""
                text = text.replace("\u2212", "-")
                match = re.fullmatch(r"([ou])(\d+(?:\.\d+)?)\s+([+-]\d+|\d+(?:\.\d+)?%)", text)
                if match is None or match[1] != side[0]:
                    break
                if not math.isfinite(float(match[2])):
                    break
                percent = match[3].endswith("%")
                value = float(match[3][:-1]) / 100 if percent else int(match[3])
                if (percent and not 0 < value < 1) or (not percent and abs(value) < 100):
                    break
                sides.append((side, float(match[2]), value, percent))
            if (
                not book
                or len(sides) != 2
                or sides[0][1] != sides[1][1]
                or sides[0][3] != sides[1][3]
            ):
                excluded += 1
                continue
            for index, (side, line, odds, percent) in enumerate(sides):
                contracts.append(
                    {
                        "game_id": game["game_id"],
                        "team": player["team"],
                        "player_id": player["player_id"],
                        "player_name": heading,
                        "market": metric,
                        "side": side,
                        "line": line,
                        "quote_format": "DISPLAYED_PERCENT_REFERENCE"
                        if percent
                        else "AMERICAN_ODDS",
                        "american_odds": None if percent else odds,
                        "opposite_american_odds": None if percent else sides[1 - index][2],
                        "displayed_probability": odds if percent else None,
                        "opposite_displayed_probability": sides[1 - index][2] if percent else None,
                        "fees_known": False,
                        "bid_ask_side_known": False,
                        "book": book,
                        "observed_utc": observed.isoformat(),
                        "source_url": url,
                        "snapshot_sha256": hashlib.sha256(html.encode()).hexdigest(),
                        "offer_origin_utc": None,
                        "offer_origin_verified": False,
                        "executable_price_verified": False,
                        "settlement_verified": False,
                        "player_identity_basis": "UNIQUE_NAME_AND_GAME; GSIS_SOURCE_UNVERIFIED",
                    }
                )
    return contracts, excluded


def search_game(game, topic, *, fetch=fetch_page):
    game_date = (
        datetime.fromisoformat(game["kickoff_utc"])
        .astimezone(ZoneInfo("America/New_York"))
        .date()
        .isoformat()
    )
    query = " ".join(
        (
            "NFL",
            '"' + TEAM_NAMES[game["away_team"]] + '"',
            '"' + TEAM_NAMES[game["home_team"]] + '"',
            game_date,
            TOPICS[topic],
            "(" + " OR ".join("site:" + domain for domain in SOURCE_DOMAINS) + ")",
        )
    )
    url = "https://www.bing.com/search?" + urlencode({"format": "rss", "q": query})
    raw = fetch(url, search=True)
    root = ElementTree.fromstring(raw)
    leads = [
        {
            "title": clean(n.findtext("title") or ""),
            "url": n.findtext("link"),
            "snippet": clean(n.findtext("description") or ""),
        }
        for n in root.findall("./channel/item")
        if approved_url(n.findtext("link") or "")
    ]
    return {
        "topic": topic,
        "query": query,
        "search_url": url,
        "search_sha256": hashlib.sha256(raw).hexdigest(),
        "leads": leads[:3],
        "search_results_are_verified_facts": False,
    }
