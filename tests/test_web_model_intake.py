from datetime import UTC, datetime

import pytest

from nfl.web_model_intake import approved_url, parse_props, player_links, search_game
from scripts.nfl_web_model_intake import collect

URL = "https://www.covers.com/sport/football/nfl/players/196870/jaxon-smith-njigba"
AT = datetime(2026, 10, 14, 12, tzinfo=UTC)
GAME = {
    "game_id": "2026_06_SEA_DEN",
    "home_team": "DEN",
    "away_team": "SEA",
    "kickoff_utc": "2026-10-16T00:15:00+00:00",
    "status": "INDEPENDENT_CONDITIONAL_FORECAST",
}
PLAYER = {
    "game_id": GAME["game_id"],
    "team": "SEA",
    "player_id": "00-0038543",
    "player_name": "Jaxon Smith-Njigba",
    "is_unallocated_bucket": False,
    "projections": {"receiving_yards": {"mean": 100}},
}
HTML = """<h1>Jaxon Smith-Njigba Projections, Player Props and Stats</h1>
<div class="covers-CoversNextMatchup">
<span class="nextGame-team"><img src="https://img.covers.com/nfl/sea.svg">Seattle</span>
<span class="nextGame-team"><img src="https://img.covers.com/nfl/den.svg">Denver</span>
<span class="nextGame-time">20:15 ET</span><span class="nextGame-date">Oct 15, 2026</span></div>
<section id="receiving-yards"><div class="other-odds-row">
<div class="other-odds-label"><img alt="FanDuel"></div>
<div class="other-over-odds">o91.5 <span>-114</span></div>
<div class="other-under-odds">u91.5 <span>-114</span></div>
</div></section><script>o1.5 +999 fake prices</script>"""


def parse(html=HTML, players=None, games=None):
    return parse_props(html, players or [PLAYER], games or [GAME], url=URL, observed=AT)


def test_exact_game_player_same_book_both_sides_are_required():
    contracts, excluded = parse()
    assert len(contracts) == 2 and excluded == 0
    assert {c["side"] for c in contracts} == {"over", "under"}
    assert {c["line"] for c in contracts} == {91.5}
    assert all(c["player_id"] == PLAYER["player_id"] for c in contracts)
    assert all(
        c["offer_origin_utc"] is None and not c["executable_price_verified"] for c in contracts
    )


def test_percentage_reference_retains_literal_quote_without_inventing_odds():
    html = HTML.replace("-114", "55%").replace("FanDuel", "Kalshi Sports")
    contracts, excluded = parse(html)
    assert excluded == 0 and len(contracts) == 2
    assert all(
        c["american_odds"] is None and c["opposite_american_odds"] is None for c in contracts
    )
    assert all(c["displayed_probability"] == 0.55 and not c["fees_known"] for c in contracts)


def test_mixed_quote_formats_are_not_complementary_prices():
    html = HTML.replace("u91.5 <span>-114</span>", "u91.5 <span>55%</span>")
    assert parse(html)[0] == []


@pytest.mark.parametrize("replacement", ["", "u90.5 -114", "u91.5 +50"])
def test_missing_mismatched_invalid_other_side_cannot_create_pair(replacement):
    html = HTML.replace("u91.5 <span>-114</span>", replacement)
    contracts, excluded = parse(html)
    assert contracts == [] and excluded == 1


def test_ambiguous_name_wrong_game_and_postkickoff_are_blocked():
    with pytest.raises(ValueError, match="IDENTITY"):
        parse(players=[PLAYER, PLAYER])
    with pytest.raises(ValueError, match="MATCHUP"):
        parse(games=[{**GAME, "home_team": "KC"}])
    with pytest.raises(ValueError, match="MATCHUP"):
        parse_props(HTML, [PLAYER], [GAME], url=URL, observed=datetime(2026, 10, 17, tzinfo=UTC))


@pytest.mark.parametrize(
    "url",
    [
        "http://espn.com/news",
        "https://espn.com.evil.test/news",
        "https://user:secret@espn.com/news",
        "https://127.0.0.1/news",
        "https://espn.com:bad/news",
    ],
)
def test_source_allowlist_rejects_unsafe_or_spoofed_hosts(url):
    assert not approved_url(url)


def test_search_keeps_trusted_leads_without_declaring_verified_facts():
    rss = b"<rss><channel><item><title>Report</title><link>https://www.espn.com/nfl/story</link></item><item><link>https://evil.test</link></item></channel></rss>"
    result = search_game(GAME, "football", fetch=lambda *a, **k: rss)
    assert len(result["leads"]) == 1
    assert result["search_results_are_verified_facts"] is False
    assert "2026-10-15" in result["query"]


def test_all_source_failures_leave_game_coverage_and_empty_contracts():
    def fail(*a, **k):
        raise OSError("unavailable")

    result = collect({"games": [GAME], "player_props": [PLAYER]}, fetch=fail)
    assert len(result["games"][0]["searches"]) == 4
    assert result["games"][0]["prop_price_coverage"] == "MISSING"
    assert result["prop_contracts"] == []
    assert result["api_keys_required"] is False


def test_collect_imports_page_price_and_records_raw_snapshots():
    def fetch(url, **kwargs):
        if kwargs.get("search"):
            return b"<rss><channel/></rss>"
        if url.endswith("/player-props"):
            return (
                '<a href="' + URL.removeprefix("https://www.covers.com") + '">Player</a>'
            ).encode()
        return HTML.encode()

    captures = []
    # Test clock is current pregame in this fixture's live development date.
    result = collect(
        {"games": [GAME], "player_props": [PLAYER]},
        fetch=fetch,
        save_raw=lambda sha, raw: captures.append((sha, raw)),
        now=lambda: AT,
    )
    assert len(result["prop_contracts"]) == 2
    assert result["games"][0]["prop_price_coverage"] == "PARTIAL"
    assert len(captures) == 8
    assert player_links('<a href="/go/book">Bet</a>') == []
