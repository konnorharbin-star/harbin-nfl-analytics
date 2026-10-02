from nfl.book_identity import canonical_book_identity


def test_draftkings_feed_aliases_collapse_to_one_book() -> None:
    assert canonical_book_identity("Draft Kings") == "draftkings"
    assert canonical_book_identity("Draft Kings - Live Odds") == "draftkings"
    assert canonical_book_identity("DraftKings") == "draftkings"


def test_distinct_sportsbooks_remain_distinct() -> None:
    identities = {
        canonical_book_identity("Draft Kings"),
        canonical_book_identity("FanDuel"),
        canonical_book_identity("BetMGM"),
        canonical_book_identity("Caesars Sportsbook"),
    }
    assert identities == {"draftkings", "fanduel", "betmgm", "caesars"}
