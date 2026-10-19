"""Conservative published-at metadata capture for public football pages.

A publisher metadata timestamp is an as-of research clue, not proof a page
has not been silently edited since publication. Never infer it from a crawl
time, headline, URL, undated article body or model projection timestamp.
"""
from datetime import UTC, datetime, timedelta

from nfl.web_model_intake import Page

_NAMES = frozenset({"article:published_time", "datepublished", "pubdate", "publishdate"})


def publisher_issue_time(html_text, *, observed):
    if not isinstance(observed, datetime) or observed.tzinfo is None:
        raise ValueError("OBSERVATION_TIME_MUST_BE_ZONED")
    observed = observed.astimezone(UTC)
    timestamps = []
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
    if not timestamps:
        return {"source_publication_utc": None, "publication_metadata_status": "NO_DATED_PUBLISHER_META"}
    earliest, latest = min(timestamps), max(timestamps)
    if latest - earliest > timedelta(minutes=5):
        return {"source_publication_utc": None, "publication_metadata_status": "CONFLICTING_PUBLISHER_META"}
    if latest > observed:
        return {"source_publication_utc": None, "publication_metadata_status": "FUTURE_PUBLISHER_META"}
    return {
        "source_publication_utc": earliest.isoformat(),
        "publication_metadata_status": "DATED_PUBLISHER_META_NOT_IMMUTABLE_ORIGINAL",
    }


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
        return {"source_publication_utc": None, "publication_evidence_status": "PAGE_CAPTURE_AFTER_KICKOFF"}
    if issued >= kickoff:
        return {"source_publication_utc": None, "publication_evidence_status": "PUBLISHER_META_AFTER_KICKOFF"}
    return {
        "source_publication_utc": issued.astimezone(UTC).isoformat(),
        "publication_evidence_status": "PUBLISHER_META_PREKICKOFF_NOT_FACT_VERIFICATION",
    }
