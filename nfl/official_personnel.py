"""Conservative parsing of explicit, dated league/club personnel announcements."""

import re
import unicodedata
from collections import defaultdict
from datetime import timedelta
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

from nfl.official_sources import CLUB_DOMAINS, official_publisher
from nfl.public_article_dates import jsonld_articles
from nfl.public_game_inputs import clock
from nfl.web_context_research import TEAM_NAMES
from nfl.web_model_intake import Node, Page, clean


def key(text):
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def article_content(html):
    """Keep main article blocks, not navigation, related headlines or search snippets."""
    metadata = jsonld_articles(html)
    if len(metadata) != 1:
        return None
    root = Page(html).root
    bodies = root.find(cls="nfl-c-body-part--text")
    if not bodies:
        return None
    segments = []

    def visit(node):
        if node.tag in ("p", "li", "h2", "h3", "h4"):
            segments.append({"tag": node.tag, "text": clean(node.text())})
            return
        for child in node.children:
            if isinstance(child, Node):
                visit(child)

    for body in bodies:
        visit(body)
    return {
        "headline": str(metadata[0].get("headline", "")),
        "segments": segments[:256],
        "truncated": len(segments) > 256,
    }


def _owner(url, teams):
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or parsed.username
        or parsed.password
        or parsed.port not in (None, 443)
    ):
        return None
    domain = official_publisher(url)
    if domain == "nfl.com":
        return "NFL"
    for team in teams:
        if domain == CLUB_DOMAINS[team]:
            return team
    return None


def _team_heading(text, teams):
    normalized = key(text)
    for team in teams:
        nickname = key(TEAM_NAMES[team].split()[-1])
        names = {nickname, key(TEAM_NAMES[team]), nickname + " inactives"}
        if team == "SF":
            names.add("niners")
        if normalized in names:
            return team
    return None


def official_article_records(articles, game):
    fields = (
        "source_url",
        "captured_utc",
        "snapshot_sha256",
        "source_publication_utc",
        "source_modified_utc",
        "official_article",
    )
    teams = (game["away_team"], game["home_team"])
    return [
        {k: a.get(k) for k in fields}
        for a in articles
        if a.get("official_article") and _owner(a.get("source_url", ""), teams)
    ]


def _eligible(article, game, asof):
    teams = (game["away_team"], game["home_team"])
    owner = _owner(article["source_url"], teams)
    if owner is None:
        raise ValueError("NOT_MATCHING_OFFICIAL_LEAGUE_OR_CLUB_SOURCE")
    published, captured, at, kickoff = map(
        clock,
        (
            article.get("source_publication_utc"),
            article.get("captured_utc"),
            asof,
            game["kickoff_utc"],
        ),
    )
    modified = clock(article.get("source_modified_utc") or article["source_publication_utc"])
    if not published <= modified <= captured <= at < kickoff:
        raise ValueError("OFFICIAL_ARTICLE_NOT_CAPTURED_PREGAME")
    if at - captured > timedelta(hours=6) or kickoff - published > timedelta(days=7):
        raise ValueError("OFFICIAL_ARTICLE_STALE")
    if not re.fullmatch(r"[0-9a-f]{64}", article.get("snapshot_sha256", "")):
        raise ValueError("OFFICIAL_ARTICLE_SNAPSHOT_MISSING")
    content = article.get("official_article")
    if not isinstance(content, dict) or content.get("truncated") is not False:
        raise ValueError("OFFICIAL_ARTICLE_BODY_NOT_COMPLETE")
    headline = key(content.get("headline", ""))
    if re.search(r"\b(projected|expected|prediction|preview)\b", headline):
        raise ValueError("OFFICIAL_ARTICLE_IS_NOT_A_CONFIRMED_ANNOUNCEMENT")
    heading_teams = {_team_heading(s["text"], teams) for s in content["segments"]}
    dated_league_lists = (
        owner == "NFL"
        and "inactives" in headline
        and set(teams) <= heading_teams
        and published.astimezone(ZoneInfo("America/New_York")).date()
        == kickoff.astimezone(ZoneInfo("America/New_York")).date()
    )
    for team in teams:
        if team == owner:
            continue
        nickname = key(TEAM_NAMES[team].split()[-1])
        if " " + nickname + " " not in " " + headline + " " and not dated_league_lists:
            raise ValueError("OFFICIAL_HEADLINE_MATCHUP_NOT_EXPLICIT")
    week = game.get("week")
    if week is None:
        week = int(game["game_id"].split("_")[1])
    body = " ".join(s["text"] for s in content["segments"])
    if not re.search(rf"\bweek\s+{int(week)}\b", headline + " " + key(body)):
        raise ValueError("OFFICIAL_ARTICLE_WEEK_NOT_EXPLICIT")
    return owner, published, modified, captured, kickoff, content


def personnel_from_articles(articles, game, players, *, asof):
    """No starter inference from depth charts, pass leaders or the inactive list."""
    teams = (game["away_team"], game["home_team"])
    model = [
        p
        for p in players
        if p.get("game_id") == game["game_id"]
        and not p.get("is_unallocated_bucket")
        and p.get("team") in teams
    ]
    by_name = defaultdict(list)
    for player in model:
        aliases = set(player.get("source_player_names", [])) | {player.get("player_name", "")}
        for name in {key(n) for n in aliases if n}:
            by_name[(player["team"], name)].append(player)
    reports, candidates, explicit_active, rejections = {}, [], [], []
    for article in articles:
        if _owner(article.get("source_url", ""), teams) is None:
            continue
        try:
            owner, published, modified, captured, kickoff, content = _eligible(article, game, asof)
            headline = key(content["headline"])
            explicit_matchup = all(
                team == owner
                or " " + key(TEAM_NAMES[team].split()[-1]) + " " in " " + headline + " "
                for team in teams
            )
            inactive_article = bool(re.search(r"\binactives\b|\binactive list\b", headline))
            inactive_article &= not bool(
                re.search(r"\b(projected|expected|potential|preview)\b", headline)
            )
            eastern = ZoneInfo("America/New_York")
            game_day = published.astimezone(eastern).date() == kickoff.astimezone(eastern).date()
            provenance = {
                k: article[k]
                for k in ("source_url", "captured_utc", "snapshot_sha256", "source_publication_utc")
            }
            provenance["source_modified_utc"] = modified.isoformat()
            current_team = owner if owner != "NFL" else None
            lists = defaultdict(list)
            for segment in content["segments"]:
                text = segment["text"]
                heading = _team_heading(text, teams)
                if heading:
                    current_team = heading
                elif segment["tag"] in ("h2", "h3", "h4"):
                    current_team = owner if owner != "NFL" else None
                if inactive_article and game_day and segment["tag"] == "li" and current_team:
                    match = re.fullmatch(
                        r"(?:QB|RB|FB|WR|TE|OL|OT|G|C|OG|DL|DE|DT|NT|LB|ILB|OLB|CB|S|FS|SS|DB|K|P|LS|EDGE)\s+(.+?)(?:\s*\((.+)\))?",
                        text,
                        re.I,
                    )
                    if match:
                        name, annotation = match.groups()
                        identified = by_name.get((current_team, key(name)), [])
                        lists[current_team].append(
                            {
                                "source_name": name,
                                "source_annotation": annotation,
                                "player_id": identified[0]["player_id"]
                                if len(identified) == 1
                                else None,
                                "identity_binding": "EXACT_TEAM_AND_UNIQUE_DIRECTORY_NAME"
                                if len(identified) == 1
                                else "NOT_UNIQUELY_BOUND_TO_MODEL_PLAYER",
                            }
                        )
                if segment["tag"] != "p" or re.search(
                    r"\b(if|unless|expected|likely|could|may|might|would|not|won't|should)\b|\?",
                    text,
                    re.I,
                ):
                    continue
                if re.search(r"\b(next week|last week|next season|last season)\b", text, re.I):
                    continue
                mentioned_weeks = {int(w) for w in re.findall(r"\bweek\s+(\d+)\b", text, re.I)}
                target_week = int(game.get("week") or game["game_id"].split("_")[1])
                if mentioned_weeks and mentioned_weeks != {target_week}:
                    continue
                normalized = key(text)
                for player in model:
                    if owner not in ("NFL", player["team"]):
                        continue
                    aliases = set(player.get("source_player_names", [])) | {
                        player.get("player_name", "")
                    }
                    for alias in aliases:
                        name = key(alias)
                        if not name or len(by_name[(player["team"], name)]) != 1:
                            continue
                        prefix = r"\b" + re.escape(name) + r"\s+"
                        if (
                            explicit_matchup
                            and player.get("position") == "QB"
                            and re.search(
                                prefix + r"(?:will start|is the starting quarterback|"
                                r"has been named the starter)\b",
                                normalized,
                            )
                        ):
                            candidates.append(
                                {
                                    "team": player["team"],
                                    "player_id": player["player_id"],
                                    "source": provenance,
                                    "effective_utc": modified.isoformat(),
                                    "identity_binding": "EXACT_TEAM_AND_UNIQUE_DIRECTORY_NAME",
                                    "evidence_text": text[:1200],
                                }
                            )
                            break
                        if (
                            inactive_article
                            and game_day
                            and re.search(prefix + r"(?:is active|will play)\b", normalized)
                        ):
                            explicit_active.append(
                                {
                                    "team": player["team"],
                                    "player_id": player["player_id"],
                                    "source": provenance,
                                    "designation": "EXPLICITLY_REPORTED_ACTIVE",
                                }
                            )
                            break
            for team, rows in lists.items():
                if not 1 <= len(rows) <= 12 or len({key(r["source_name"]) for r in rows}) != len(
                    rows
                ):
                    continue
                report = {
                    "team": team,
                    "source": provenance,
                    "players": rows,
                    "status": "OFFICIAL_GAME_DAY_INACTIVE_LIST",
                }
                previous = reports.get(team)
                if previous is None or clock(previous["source"]["source_modified_utc"]) < modified:
                    reports[team] = report
                elif (
                    clock(previous["source"]["source_modified_utc"]) == modified
                    and previous["players"] != rows
                ):
                    reports[team] = {
                        "team": team,
                        "status": "CONFLICTING_OFFICIAL_LISTS",
                        "source": provenance,
                        "players": [],
                    }
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            rejections.append({"source_url": article.get("source_url"), "reason": str(error)[:150]})
    inactive_ids = {
        (team, row["player_id"])
        for team, report in reports.items()
        if report["status"] == "OFFICIAL_GAME_DAY_INACTIVE_LIST"
        for row in report["players"]
        if row["player_id"]
    }
    starters = {}
    for team in teams:
        choices = [c for c in candidates if c["team"] == team]
        if not choices:
            continue
        latest = max(clock(c["effective_utc"]) for c in choices)
        choices = [c for c in choices if clock(c["effective_utc"]) == latest]
        if len({c["player_id"] for c in choices}) == 1:
            choice = choices[0]
            if (team, choice["player_id"]) not in inactive_ids:
                starters[team] = choice
    active_ids = {(p["team"], p["player_id"]) for p in explicit_active}
    conflicts = inactive_ids & active_ids
    inactive_ids -= conflicts
    active = [p for p in explicit_active if (p["team"], p["player_id"]) not in conflicts]
    return {
        "status": "EXPLICIT_OFFICIAL_PERSONNEL_EVIDENCE",
        "game_id": game["game_id"],
        "starters": starters,
        "inactive_reports": reports,
        "explicit_active_players": active,
        "confirmed_player_designations": len(
            inactive_ids | {(p["team"], p["player_id"]) for p in active}
        ),
        "model_players_reviewed": len(model),
        "rejections": rejections,
        "conflicting_player_designations": sorted(conflicts),
        "not_listed_does_not_prove_active": True,
        "model_adjustment": None,
    }
