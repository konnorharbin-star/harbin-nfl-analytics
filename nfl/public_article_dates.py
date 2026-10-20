"""Conservative published-at metadata capture for public football pages.

A publisher metadata timestamp is an as-of research clue, not proof a page
has not been silently edited since publication. Never infer it from a crawl
time, headline, URL, undated article body or model projection timestamp.
"""

import json
from datetime import UTC, datetime, timedelta

from nfl.web_model_intake import Page

_NAMES = frozenset({"article:published_time", "datepublished", "pubdate", "publishdate"})


def jsonld_articles(html_text):
    articles = []
    for node in Page(html_text).root.find(tag="script"):
        if node.attrs.get("type", "").lower() != "application/ld+json":
            continue
        try:
            data = json.loads("".join(c for c in node.children if isinstance(c, str)))
        except (ValueError, TypeError):
            continue
        pending = data if isinstance(data, list) else [data]
        for item in pending:
            if not isinstance(item, dict):
                continue
            graph = item.get("@graph")
            candidates = graph if isinstance(graph, list) else [item]
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                kinds = candidate.get("@type")
                kinds = kinds if isinstance(kinds, list) else [kinds]
                if any(k in ("NewsArticle", "Article") for k in kinds):
                    articles.append(candidate)
    return articles


def publisher_issue_time(html_text, *, observed):
    if not isinstance(observed, datetime) or observed.tzinfo is None:
        raise ValueError("OBSERVATION_TIME_MUST_BE_ZONED")
    observed = observed.astimezone(UTC)
    timestamps = []
    modified = None
    try:
        nodes = Page(html_text).root.find(tag="meta")
    except (ValueError, TypeError):
        return {"source_publication_utc": None, "publication_metadata_status": "INVALID_PAGE"}
    for node in nodes:
        meta = (node.attrs.get("property") or node.attrs.get("name") or "").strip().lower()
        if meta not in _NAMES:
            continue
        value = (node.attrs.get("content") or "").strip()
        try:
            issued = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            continue
        if issued.tzinfo is None:
            continue  # Never infer a timezone from a date-only string.
        timestamps.append(issued.astimezone(UTC))
    articles = jsonld_articles(html_text)
    if len(articles) > 1:
        return {
            "source_publication_utc": None,
            "publication_metadata_status": "AMBIGUOUS_ARTICLE_METADATA",
        }
    if articles:
        article = articles[0]
        try:
            issued = datetime.fromisoformat(
                str(article.get("datePublished", "")).replace("Z", "+00:00")
            )
            if issued.tzinfo is not None:
                timestamps.append(issued.astimezone(UTC))
            if article.get("dateModified") is not None:
                modified = datetime.fromisoformat(article["dateModified"].replace("Z", "+00:00"))
                if modified.tzinfo is None or modified > observed or modified < issued:
                    raise ValueError("INVALID_MODIFIED_TIME")
        except (ValueError, TypeError):
            return {
                "source_publication_utc": None,
                "publication_metadata_status": "INVALID_ARTICLE_METADATA_CLOCK",
            }
    if not timestamps:
        return {
            "source_publication_utc": None,
            "publication_metadata_status": "NO_DATED_PUBLISHER_META",
        }
    earliest, latest = min(timestamps), max(timestamps)
    if latest - earliest > timedelta(minutes=5):
        return {
            "source_publication_utc": None,
            "publication_metadata_status": "CONFLICTING_PUBLISHER_META",
        }
    if latest > observed:
        return {
            "source_publication_utc": None,
            "publication_metadata_status": "FUTURE_PUBLISHER_META",
        }
    result = {
        "source_publication_utc": earliest.isoformat(),
        "publication_metadata_status": "DATED_PUBLISHER_META_NOT_IMMUTABLE_ORIGINAL",
    }
    if modified is not None:
        result["source_modified_utc"] = modified.astimezone(UTC).isoformat()
    return result


def bind_publication_to_game(article, *, kickoff_utc):
    """Do not label a postgame or post-capture page observation pregame."""
    if not article:
        return {"source_publication_utc": None, "publication_evidence_status": "PAGE_NOT_READ"}
    status = article.get("publication_metadata_status", "NO_DATED_PUBLISHER_META")
    published = article.get("source_publication_utc")
    if published is None:
        return {"source_publication_utc": None, "publication_evidence_status": status}
    try:
        kickoff = datetime.fromisoformat(kickoff_utc.replace("Z", "+00:00"))
        observed = datetime.fromisoformat(article["captured_utc"].replace("Z", "+00:00"))
        issued = datetime.fromisoformat(published.replace("Z", "+00:00"))
        if any(x.tzinfo is None for x in (kickoff, observed, issued)):
            raise ValueError("UNZONED")
    except (KeyError, TypeError, ValueError):
        return {"source_publication_utc": None, "publication_evidence_status": "INVALID_GAME_TIME"}
    if observed >= kickoff:
        return {
            "source_publication_utc": None,
            "publication_evidence_status": "PAGE_CAPTURE_AFTER_KICKOFF",
        }
    if issued >= kickoff:
        return {
            "source_publication_utc": None,
            "publication_evidence_status": "PUBLISHER_META_AFTER_KICKOFF",
        }
    return {
        "source_publication_utc": issued.astimezone(UTC).isoformat(),
        "publication_evidence_status": "PUBLISHER_META_PREKICKOFF_NOT_FACT_VERIFICATION",
    }
