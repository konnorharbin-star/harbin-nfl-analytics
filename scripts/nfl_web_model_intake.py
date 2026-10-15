"""Search/read public football pages and import same-book player prop pairs. No keys."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock

from nfl.web_model_intake import (
    BOARD,
    SITUATIONAL_TOPICS,
    TOPICS,
    Page,
    clean,
    fetch_page,
    name_key,
    parse_props,
    player_links,
    search_game,
)


def collect(
    predictions,
    *,
    fetch=fetch_page,
    max_players=60,
    max_articles=32,
    save_raw=None,
    now=lambda: datetime.now(UTC),
):
    if not 1 <= max_players <= 100 or not 0 <= max_articles <= 32:
        raise ValueError("public request limits exceeded")
    games = [
        g for g in predictions["games"] if g.get("status") == "INDEPENDENT_CONDITIONAL_FORECAST"
    ]
    players = predictions["player_props"]
    report = {
        "spec": "NFL_PUBLIC_WEB_MODEL_INTAKE_V1",
        "status": "PUBLIC_WEB_RESEARCH_ONLY",
        "capture_utc": now().isoformat(),
        "api_keys_required": False,
        "paid_sources_used": False,
        "model_coefficients_changed": False,
        "betting_authorized": False,
        "articles": [],
        "games": [],
        "source_errors": [],
        "prop_contracts": [],
        "required_situational_factors": list(SITUATIONAL_TOPICS),
        "situational_search_is_mandatory": True,
    }

    snapshot_lock = Lock()

    def get(url, **kwargs):
        raw = fetch(url, **kwargs)
        if save_raw is not None:
            with snapshot_lock:
                save_raw(hashlib.sha256(raw).hexdigest(), raw)
        return raw

    def run_search(game, topic):
        result = search_game(game, topic, fetch=get)
        result["observed_utc"] = now().isoformat()
        return result

    article_queues = []
    aliases = {"LA": "lar", "WAS": "wsh"}
    fallback_urls = [
        "https://www.espn.com/nfl/team/_/name/" + aliases.get(team, team.lower())
        for game in games
        for team in (game["away_team"], game["home_team"])
    ]
    for game in games:
        view = {
            "game_id": game["game_id"],
            "searches": [],
            "paired_prop_sides": 0,
            "prop_price_coverage": "MISSING",
            "structured_context_verified": False,
            "situational_factors": {},
        }
        game_urls = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            pending = {topic: executor.submit(run_search, game, topic) for topic in TOPICS}
            for topic, future in pending.items():
                try:
                    search = future.result()
                    view["searches"].append(search)
                    game_urls.extend(lead["url"] for lead in search["leads"])
                except Exception as error:
                    report["source_errors"].append(
                        {"game_id": game["game_id"], "topic": topic, "error": type(error).__name__}
                    )
                    view["searches"].append(
                        {
                            "topic": topic,
                            "status": "SEARCH_UNAVAILABLE",
                            "observed_utc": now().isoformat(),
                            "error": type(error).__name__,
                        }
                    )
        game_urls.extend(
            "https://www.espn.com/nfl/team/_/name/" + aliases.get(team, team.lower())
            for team in (game["away_team"], game["home_team"])
        )
        article_queues.append(list(dict.fromkeys(game_urls)))
        report["games"].append(view)
    article_urls = []
    while any(article_queues):
        for queue in article_queues:
            if queue:
                article_urls.append(queue.pop(0))
    for url in list(dict.fromkeys(article_urls + fallback_urls))[:max_articles]:
        try:
            raw = get(url)
            body = clean(Page(raw.decode("utf-8", errors="replace")).root.text())
            report["articles"].append(
                {
                    "source_url": url,
                    "captured_utc": now().isoformat(),
                    "snapshot_sha256": hashlib.sha256(raw).hexdigest(),
                    "body_excerpt": body[:16000],
                    "body_excerpt_truncated": len(body) > 16000,
                    "structured_facts_verified": False,
                }
            )
        except Exception as error:
            report["source_errors"].append({"source_url": url, "error": type(error).__name__})
    articles = {a["source_url"]: a for a in report["articles"]}
    for game in report["games"]:
        by_topic = {s["topic"]: s for s in game["searches"]}
        for topic in SITUATIONAL_TOPICS:
            search = by_topic[topic]
            evidence = [
                {
                    "source_url": lead["url"],
                    "title": lead["title"],
                    "snippet": lead["snippet"],
                    "page_read": lead["url"] in articles,
                    "page_capture_utc": articles.get(lead["url"], {}).get("captured_utc"),
                    "page_sha256": articles.get(lead["url"], {}).get("snapshot_sha256"),
                    "source_publication_utc": None,
                }
                for lead in search.get("leads", [])
            ]
            status = (
                "SEARCH_UNAVAILABLE"
                if search.get("status") == "SEARCH_UNAVAILABLE"
                else "NO_TRUSTED_RESULTS"
                if not evidence
                else "SOURCE_PAGE_READ_UNVERIFIED"
                if any(e["page_read"] for e in evidence)
                else "TRUSTED_LEADS_UNREAD"
            )
            game["situational_factors"][topic] = {
                "search_attempted": True,
                "status": status,
                "search_observed_utc": search.get("observed_utc"),
                "search_sha256": search.get("search_sha256"),
                "evidence": evidence,
                "verified_value": None,
                "model_adjustment": None,
                "used_as_calibrated_feature": False,
            }
        game["situational_search_checklist_complete"] = len(game["situational_factors"]) == len(
            SITUATIONAL_TOPICS
        )
        game["situational_facts_verified"] = False
    try:
        links = player_links(get(BOARD).decode("utf-8", errors="replace"))
        candidates = []
        for player in players:
            if player["is_unallocated_bucket"]:
                continue
            for url in links:
                if name_key(url.rsplit("/", 1)[-1]) == name_key(player["player_name"]):
                    candidates.append(
                        (
                            player["game_id"],
                            -player["projections"].get("receiving_yards", {}).get("mean", 0),
                            url,
                        )
                    )
        buckets = {}
        for gid, _mean, url in sorted(set(candidates)):
            buckets.setdefault(gid, []).append(url)
        ordered = []
        while any(buckets.values()):
            for gid in sorted(buckets):
                if buckets[gid]:
                    ordered.append(buckets[gid].pop(0))
        urls = list(dict.fromkeys(ordered))[:max_players]
        report.update(
            discovered_player_pages=len(links),
            requested_player_pages=len(urls),
            request_limit_players=max_players,
        )
        excluded = 0
        for url in urls:
            try:
                raw = get(url)
                props, invalid = parse_props(
                    raw.decode("utf-8", errors="replace"),
                    players,
                    games,
                    url=url,
                    observed=now(),
                )
                for prop in props:
                    prop["snapshot_sha256"] = hashlib.sha256(raw).hexdigest()
                report["prop_contracts"].extend(props)
                excluded += invalid
            except Exception as error:
                report["source_errors"].append(
                    {"source_url": url, "error": type(error).__name__, "reason": str(error)[:150]}
                )
        report["incomplete_or_mismatched_book_pairs_excluded"] = excluded
    except Exception as error:
        report["source_errors"].append({"source_url": BOARD, "error": type(error).__name__})
    for game in report["games"]:
        count = sum(p["game_id"] == game["game_id"] for p in report["prop_contracts"])
        american = sum(
            p["game_id"] == game["game_id"]
            and p.get("quote_format") != "DISPLAYED_PERCENT_REFERENCE"
            for p in report["prop_contracts"]
        )
        game.update(
            paired_prop_sides=count,
            american_price_sides=american,
            displayed_reference_sides=count - american,
            prop_price_coverage="PARTIAL"
            if american
            else ("DISPLAYED_REFERENCE_ONLY" if count else "MISSING"),
        )
    report["collection_started_utc"] = report["capture_utc"]
    report["capture_utc"] = now().isoformat()
    report["situational_factor_status_counts"] = dict(
        Counter(
            factor["status"]
            for game in report["games"]
            for factor in game["situational_factors"].values()
        )
    )
    report["situational_searches_attempted"] = sum(
        len(game["situational_factors"]) for game in report["games"]
    )
    report["verified_situational_values"] = 0
    return report


def coverage_markdown(report):
    lines = [
        "## Situational web research",
        "",
        "Every forecast game receives the required factor searches. Page reads and search hits "
        "are unverified evidence; unknown factors do not become numerical adjustments.",
        "",
        "| Game | Searches attempted | Factors with read leads | Missing results | Verified |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for game in report["games"]:
        factors = list(game["situational_factors"].values())
        pages = sum(f["status"] == "SOURCE_PAGE_READ_UNVERIFIED" for f in factors)
        unavailable = sum(
            f["status"] in ("SEARCH_UNAVAILABLE", "NO_TRUSTED_RESULTS") for f in factors
        )
        lines.append(f"| {game['game_id']} | {len(factors)} | {pages} | {unavailable} | 0 |")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, default=Path("outputs/independent_forecasts.json"))
    p.add_argument("--out", type=Path, default=Path("reports/nfl_web_model_intake.json"))
    p.add_argument(
        "--prop-contracts", type=Path, default=Path("reports/nfl_web_prop_contracts.json")
    )
    p.add_argument("--max-players", type=int, default=60)
    p.add_argument("--max-articles", type=int, default=32)
    a = p.parse_args()
    snapshots = a.out.parent / "web_source_snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)

    def save_raw(digest, raw):
        path = snapshots / (digest + ".gz")
        if not path.exists():
            path.write_bytes(gzip.compress(raw, mtime=0))

    report = collect(
        json.loads(a.predictions.read_text()),
        max_players=a.max_players,
        max_articles=a.max_articles,
        save_raw=save_raw,
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    a.out.with_suffix(".md").write_text(coverage_markdown(report))
    a.prop_contracts.parent.mkdir(parents=True, exist_ok=True)
    a.prop_contracts.write_text(
        json.dumps(report["prop_contracts"], indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "games": len(report["games"]),
                "public_prop_sides": len(report["prop_contracts"]),
                "read_articles": len(report["articles"]),
                "source_errors": len(report["source_errors"]),
                "api_keys_required": False,
                "situational_searches_attempted": report["situational_searches_attempted"],
                "validated_edges": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
