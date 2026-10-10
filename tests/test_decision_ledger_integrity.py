import csv

import polars as pl

from nfl.decision_ledger import append_portfolio_decisions


def row(market="spread", odds=-110):
    return {
        "game_id": "g1",
        "quant_market": market,
        "quant_side": "home",
        "quant_odds": odds,
        "portfolio_candidate_units": 0.5,
        "portfolio_action": "PAPER",
    }


def test_append_preserves_prefix_and_all_market_ids(tmp_path):
    path = tmp_path / "ledger.csv"
    rows = [row(), row("total")]
    assert append_portfolio_decisions(pl.DataFrame(rows), path=path)["appended_rows"] == 2
    original = path.read_bytes()
    assert append_portfolio_decisions(pl.DataFrame(rows), path=path)["appended_rows"] == 0
    assert path.read_bytes() == original
    rows = [row(odds=-115)]
    assert append_portfolio_decisions(pl.DataFrame(rows), path=path)["appended_rows"] == 1
    assert path.read_bytes().startswith(original)
    # Replaying an older recommendation cannot duplicate it after price changes.
    rows = [row()]
    assert append_portfolio_decisions(pl.DataFrame(rows), path=path)["appended_rows"] == 0
    with path.open() as handle:
        assert len(list(csv.DictReader(handle))) == 3


def test_legacy_header_kept_with_complete_immutable_sidecar(tmp_path):
    path = tmp_path / "ledger.csv"
    original = b"decision_at,game_id,quant_market,decision_signature\n"
    path.write_bytes(original)
    rows = [row()]
    result = append_portfolio_decisions(pl.DataFrame(rows), path=path)
    assert result["appended_rows"] == 1
    assert path.read_bytes().startswith(original)
    sidecars = list(path.with_suffix(".events").glob("*.json"))
    assert len(sidecars) == 1
    import json
    assert json.loads(sidecars[0].read_text())["portfolio_candidate_units"] == .5
